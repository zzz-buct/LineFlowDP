import os
import time
import numpy as np
import warnings
import glob
from script.my_util import *

warnings.filterwarnings('ignore')

source_code_path = '../../sourcecode/'
model_path = './model/'
used_file_path = '../../datasets/used_file_data/'
save_path = './corpus/'

os.makedirs(model_path, exist_ok=True)
os.makedirs(save_path, exist_ok=True)


def readData(path, file):
    data = pd.read_csv(path + file, encoding='latin', keep_default_na=False)
    return data


def get_line_flow(project, release):
    data_all = readData(path=used_file_path, file=release + '.csv')
    files_list = data_all.drop_duplicates('filename', keep='first')
    files_list = list(files_list['filename'])
    for files_list_index in range(len(files_list)):
        file_name = files_list[files_list_index]
        print('file_name:', release + '_', file_name, 'progress:', files_list_index + 1, '/', len(files_list))
        folder = (source_code_path + project + '/' + release + '/' + file_name).replace('.java', '')
        file_df = data_all.loc[data_all['filename'] == file_name, :]

        line2code = dict(zip(file_df['line_number'], file_df['code_line']))

        edges = np.loadtxt(folder + '_pdg.txt', delimiter=' ')
        edge_labels = np.loadtxt(folder + '_edge_label.txt')
        try:
            sources = [node[0] for node in edges]
            targets = [node[1] for node in edges]
        except:
            sources = [edges[0]]
            targets = [edges[1]]
            edge_labels = [edge_labels]
        nodes = np.unique(edges.flatten())

        for file_line in range(len(file_df)):
            node = int(file_line + 1)

            code_extend = str(line2code.get(node, ""))

            if node in nodes:
                control_forward = []
                data_forward = []
                control_backward = []
                data_backward = []

                for source_index in range(len(sources)):
                    if (sources[source_index] == node) and (edge_labels[source_index] == 1):
                        tgt = targets[source_index]
                        if tgt in line2code:
                            control_forward.append(line2code[tgt])
                    if (sources[source_index] == node) and (edge_labels[source_index] == 2):
                        tgt = targets[source_index]
                        if tgt in line2code:
                            data_forward.append(line2code[tgt])
                    if (targets[source_index] == node) and (edge_labels[source_index] == 1):
                        src = sources[source_index]
                        if src in line2code:
                            control_backward.append(line2code[src])
                    if (targets[source_index] == node) and (edge_labels[source_index] == 2):
                        src = sources[source_index]
                        if src in line2code:
                            data_backward.append(line2code[src])

                if len(control_forward) > 0:
                    code_extend = str(code_extend) + '\n' + str(control_forward[0])
                if len(data_forward) > 0:
                    code_extend = str(code_extend) + '\n' + str(data_forward[0])
                if len(control_backward) > 0:
                    code_extend = str(control_backward[0]) + '\n' + str(code_extend)
                if len(data_backward) > 0:
                    code_extend = str(data_backward[0]) + '\n' + str(code_extend)
            data_all.loc[
                (data_all['filename'] == file_name) & (data_all['line_number'] == node), 'code_line'
            ] = str(code_extend)
    data_all.to_csv(save_path + release + '_line_flow.csv', encoding='latin', na_rep=False, index=False)


def combine_code_line(project, path):
    l = []
    for release in all_releases[project]:
        file_path = os.path.join(path, f"{release}_line_flow.csv")
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        df = pd.read_csv(file_path, encoding='latin', keep_default_na=False)
        df['release'] = release
        l.append(df)
    if l:
        demomerge = pd.concat(l, axis=0, ignore_index=True, join='inner')
        demomerge.to_csv(save_path + f"{project}.csv", index=False, encoding='latin')
        print(f"Combined csv saved to: {save_path}{project}.csv")
    else:
        print("No files merged.")


def main():
    total_time = 0
    count = 0
    path = '../doc2vec/corpus'
    for project in all_releases.keys():
        for release in all_releases[project]:
            start_time = time.time()
            get_line_flow(project=project, release=release)
            end_time = time.time()
            elapsed_time = end_time - start_time
            total_time += elapsed_time
            count += 1
            print(f"Project: {project}, Release: {release}, Time: {elapsed_time} seconds")
        combine_code_line(project, path)
    average_time = total_time / count
    print(f"Average Time: {average_time} seconds")


if __name__ == '__main__':
    main()
