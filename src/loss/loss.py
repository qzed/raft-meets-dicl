class Loss:
    def __init__(self):
        pass

    def get_config(self):
        raise NotImplementedError

    def compute(self, result, target, valid):
        raise NotImplementedError

    def __call__(self, result, target, valid):
        return self.compute(result, target, valid)
