import os.path as osp
from typing import Callable, List, Optional
import torch.nn.functional as F
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, HeteroData
from torch_geometric.io import fs
from tqdm import tqdm


class MyHeteroDataset(InMemoryDataset):
    def __init__(
            self,
            root: str,
            name: str,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None,
            force_reload: bool = False,
            use_node_attr: bool = False,
            use_edge_attr: bool = False,
            cleaned: bool = False,
    ) -> None:
        self.name = name
        self.cleaned = cleaned
        self.use_node_attr = use_node_attr
        self.use_edge_attr = use_edge_attr
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        names = ["A", "edge_labels", "graph_indicator", "graph_labels", "node_attributes", "node_labels"]
        return [f"{self.name}_{n}.txt" for n in names]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        pass  # Not needed, files are assumed to be already downloaded.

    def process(self):
        raw_path = self.raw_dir

        with open(osp.join(raw_path, f'{self.name}_node_ids.txt'), encoding='utf-8') as f:
            all_node_ids = [line.strip() for line in f]

        # Load all data at once
        edges = np.loadtxt(osp.join(raw_path, f'{self.name}_A.txt'), dtype=int, delimiter=',') - 1
        edge_types = np.loadtxt(osp.join(raw_path, f'{self.name}_edge_labels.txt'), dtype=int)
        node_graph = np.loadtxt(osp.join(raw_path, f'{self.name}_graph_indicator.txt'), dtype=int) - 1
        graph_labels = np.loadtxt(osp.join(raw_path, f'{self.name}_graph_labels.txt'), dtype=int)
        node_attributes = np.loadtxt(osp.join(raw_path, f'{self.name}_node_attributes.txt'), dtype=float, delimiter=',')
        node_labels = np.loadtxt(osp.join(raw_path, f'{self.name}_node_labels.txt'), dtype=int)

        num_graphs = graph_labels.shape[0]
        node_graph = torch.from_numpy(node_graph)
        node_attributes = torch.from_numpy(node_attributes)
        node_labels = torch.from_numpy(node_labels)
        edges = torch.from_numpy(edges)
        edge_types = torch.from_numpy(edge_types)

        data_list = []
        error_count = 0

        graph_node_indices = [[] for _ in range(num_graphs)]
        for idx, g_id in enumerate(node_graph):
            graph_node_indices[g_id.item()].append(idx)

        graph_edge_indices = [[] for _ in range(num_graphs)]
        for idx, (src, tgt) in enumerate(edges):
            src_graph = node_graph[src]
            tgt_graph = node_graph[tgt]
            if src_graph == tgt_graph:
                graph_edge_indices[src_graph.item()].append(idx)

        for i in tqdm(range(num_graphs), desc=f"Building HeteroData for {self.name}"):
            try:
                node_idx = graph_node_indices[i]
                edge_idx = graph_edge_indices[i]

                if len(node_idx) == 0:
                    continue

                node_id_map = {old_id: new_id for new_id, old_id in enumerate(node_idx)}

                node_id = [all_node_ids[j] for j in node_idx]

                edge_pairs = edges[edge_idx]
                src = [node_id_map[s.item()] for s in edge_pairs[:, 0]]
                tgt = [node_id_map[t.item()] for t in edge_pairs[:, 1]]

                x = node_attributes[node_idx, :-1].float()
                node_type = node_attributes[node_idx, -1].long()
                y_node = node_labels[node_idx].long()
                y_graph = torch.tensor([graph_labels[i]], dtype=torch.long)
                edge_index = torch.tensor([src, tgt], dtype=torch.long)
                edge_attr = edge_types[edge_idx].long()

                data = HeteroData()
                data["node"].x = x
                data["node"].y = y_node
                data["node"].type = node_type
                data["node"].batch = torch.zeros(x.size(0), dtype=torch.long)
                data["node", "edge", "node"].edge_index = edge_index
                data["node", "edge", "node"].edge_type = edge_attr
                data["node"].node_id = node_id

                if edge_attr.numel() > 0:
                    data["node", "edge", "node"].edge_attr = F.one_hot(edge_attr, num_classes=4).float()
                else:
                    data["node", "edge", "node"].edge_attr = torch.empty((0, 4), dtype=torch.float)

                data.y = y_graph
                data_list.append(data)

            except Exception as e:
                error_count += 1
                print(f"Skipped graph {i} due to error: {e}")

        print(f"Finished processing {self.name}:")
        print(f"   Total Graphs   : {num_graphs}")
        print(f"   Valid Graphs   : {len(data_list)}")
        print(f"   Skipped Graphs : {error_count}")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({len(self)})'
