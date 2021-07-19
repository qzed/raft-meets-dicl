from .adapter import TorchAdapter


class Collection:
    def __init__(self):
        pass

    def get_config(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def torch(self):
        return TorchAdapter(self)
