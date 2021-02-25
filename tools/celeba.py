from zipfile import ZipFile
from pathlib import Path
import pandas as pd

from torchvision.datasets.utils import verify_str_arg


class CelebALoader(dict):
    '''载入百度网盘下载的 CelebA 数据'''
    split_map = {
        "train": 0,
        "valid": 1,
        "test": 2,
        "all": None,
    }

    def __init__(self, root, *args, **kw):
        super().__init__(*args, **kw)
        self.__dict__ = self
        root = Path(root)
        self._init_load(root)

    def _mask(self, split):
        split_ = self.split_map[verify_str_arg(split.lower(), "split",
                                               ("train", "valid", "test", "all"))]
        _mask = slice(None) if split_ is None else (self.splits[1] == split_)
        return _mask

    def _init_load(self, root):
        self.splits = pd.read_csv(root/"Eval/list_eval_partition.txt",
                                  delim_whitespace=True, header=None, index_col=0)
        self.identity = pd.read_csv(root/"Anno/identity_CelebA.txt",
                                    delim_whitespace=True, header=None, index_col=0)
        self.bbox = pd.read_csv(root/"Anno/list_bbox_celeba.txt",
                                delim_whitespace=True, header=1, index_col=0)
        self.landmarks_align = pd.read_csv(root/"Anno/list_landmarks_align_celeba.txt",
                                           delim_whitespace=True, header=1)
        self.attr = pd.read_csv(root/"Anno/list_attr_celeba.txt",
                                delim_whitespace=True, header=1)
        self.Z = ZipFile(root/'Img/img_align_celeba.zip')
