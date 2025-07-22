from script.my_util import *
from MyHeteroDataset import MyHeteroDataset
import warnings

warnings.filterwarnings("ignore")

data_path = './data/'


def tudataset(project, version):
    MyHeteroDataset(root=(data_path + project), name=version, use_node_attr=True, force_reload=True)


if __name__ == '__main__':
    for project in list(all_releases.keys()):
        cur_releases = all_releases[project]
        for release in cur_releases:
            tudataset(project=project, version=release)
