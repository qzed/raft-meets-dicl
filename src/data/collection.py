class DataCollection:
    def __init__(self):
        pass

    def get_config(self):
        raise NotImplementedError

    def get_image_loader(self):
        raise NotImplementedError

    def get_flow_loader(self):
        raise NotImplementedError

    def get_files(self):
        raise NotImplementedError

    def __getitem__(self, index):
        img1, img2, flow, key = self.get_files()[index]

        img1 = self.get_image_loader().load(img1)
        img2 = self.get_image_loader().load(img2)
        flow, valid = self.get_flow_loader().load(flow)

        return img1, img2, flow, valid, key

    def __len__(self):
        return len(self.files)

    def validate_files(self):
        for img1, img2, flow in self.files:
            if not img1.exists():
                return False
            if not img2.exists():
                return False
            if not flow.exists():
                return False

        return True
