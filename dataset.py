from torchvision import transforms

from tools.vision import VisionDataset


class Transform:
    def __init__(self, fine_size):
        self.train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                         transforms.Resize(
                                             (fine_size, fine_size)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])
        self.test = transforms.Compose([transforms.Resize((fine_size, fine_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])


class Split:
    def __init__(self, loader, fine_size):
        self.loader = loader
        self.transforms = Transform(fine_size)

    def _dataset(self, split_type, target_type, transform=None, target_transform=None, alpha=1):
        '''
        target_type 是 ["attr", "identity", "bbox", "landmarks"] 的子集
        '''
        return VisionDataset(self.loader, split_type, target_type, transform, target_transform, alpha)

    def train(self, target_type, target_transform=None, alpha=1):
        return self._dataset('train', target_type, self.transforms.train, target_transform, alpha)

    def valid(self, target_type, target_transform=None, alpha=1):
        return self._dataset('valid', target_type, self.transforms.test, target_transform, alpha)

    def test(self, target_type, target_transform=None, alpha=1):
        return self._dataset('test', target_type, self.transforms.test, target_transform, alpha)
