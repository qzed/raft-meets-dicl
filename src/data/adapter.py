import torch


class TorchAdapter:
    def __init__(self, source):
        self.source = source

    def get_config(self):
        return self.source.get_config()

    def __getitem__(self, index):
        img1, img2, flow, valid, key = self.source[index]

        img1 = torch.from_numpy(img1).float().permute(2, 0, 1)
        img2 = torch.from_numpy(img2).float().permute(2, 0, 1)
        flow = torch.from_numpy(flow).float().permute(2, 0, 1)
        valid = torch.from_numpy(valid).bool()

        return img1, img2, flow, valid, key

    def __len__(self):
        return len(self.source)
