class DataCollection:
    def __init__(self):
        pass

    def get_config(self):
        raise NotImplementedError

    def get_files(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def validate_files(self):
        for img1, img2, flow in self.files:
            if not img1.exists():
                return False
            if not img2.exists():
                return False
            if not flow.exists():
                return False

        return True
