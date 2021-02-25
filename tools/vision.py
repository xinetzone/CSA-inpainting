from typing import Any, Callable, List, Optional, Union, Tuple
from PIL import Image
import numpy as np

from torch import as_tensor

from .loader import VisionLoader


class VisionDataset(VisionLoader):
    def __init__(self, loader: dict, split: str = "train",
                 target_type: Union[List[str], str] = "attr",
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        super().__init__(loader, transform=transform,
                         target_transform=target_transform)
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError(
                'target_transform is specified but target_type is empty')
        self.split = split
        self._init_load()

    def _init_load(self):
        mask = self.loader._mask(self.split)
        self.filename = self.loader.splits[mask].index.values
        self.identity = as_tensor(self.loader.identity[mask].values)
        self.bbox = as_tensor(self.loader.bbox[mask].values)
        self.landmarks_align = as_tensor(
            self.loader.landmarks_align[mask].values)
        self.attr = as_tensor(self.loader.attr[mask].values)
        self.attr = (self.attr + 1) // 2  # map from {-1, 1} to {0, 1}
        self.attr_names = list(self.loader.attr.columns)

    def __len__(self) -> int:
        return len(self.attr)

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        target: Any = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            elif t == "identity":
                target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError(
                    "Target type \"{}\" is not recognized.".format(t))
        with self.loader.Z.open(f'img_align_celeba/{self.filename[index]}') as im:
            X = np.array(Image.open(im))

        X = as_tensor(X)
        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return X, target
