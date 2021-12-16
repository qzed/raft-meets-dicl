from ... import ModelAdapter, Result


class MultiLevelSequenceAdapter(ModelAdapter):
    def __init__(self, model):
        super().__init__(model)

    def wrap_result(self, result, original_shape) -> Result:
        return MultiLevelSequenceResult(result, original_shape)


class MultiLevelSequenceResult(Result):
    def __init__(self, output, shape):
        super().__init__()

        self.result = output        # list of lists (level, iteration)
        self.shape = shape

    def output(self, batch_index=None):
        if batch_index is None:
            return self.result

        return [[x[batch_index].view(1, *x.shape[1:]) for x in level] for level in self.result]

    def final(self):
        return self.result[-1][-1]

    def intermediate_flow(self):
        return self.result
