from torch.utils import data

from tools.celeba import CelebALoader
from dataset import Split
from tools.toml import load_option
from tools.mask import mask_iter


class Loader:
    def __init__(self, batch_size, fine_size, root, mask_root):
        self.batch_size = batch_size
        self._root = root
        self._mask_root = mask_root
        self.fine_size = fine_size
        self.loader = CelebALoader(self._root)
        self.dataset = Split(self.loader, self.fine_size)

    def trainset(self, alpha=1):
        _dataset = self.dataset.train(
            'bbox',  alpha=alpha)
        return data.DataLoader(_dataset,
                               batch_size=self.batch_size,
                               shuffle=True)

    def valset(self, alpha=1):
        _dataset = self.dataset.valid(
            'bbox',  alpha=alpha)
        return data.DataLoader(_dataset,
                               batch_size=self.batch_size,
                               shuffle=False)

    def testset(self, alpha=1):
        _dataset = self.dataset.test(
            'bbox',  alpha=alpha)
        return data.DataLoader(_dataset,
                               batch_size=self.batch_size,
                               shuffle=False)

    @property
    def maskset(self):
        return mask_iter(self._mask_root, self.fine_size)


_loader_opt = load_option('options/loader.toml')
loader = Loader(**_loader_opt)
