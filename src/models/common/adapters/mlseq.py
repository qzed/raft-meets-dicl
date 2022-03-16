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

        if not isinstance(self.result[0][0], tuple):
            return [[x[batch_index].view(1, *x.shape[1:]) for x in level] for level in self.result]
        else:
            return [[[x[batch_index].view(1, *x.shape[1:]) for x in tp] for tp in level] for level in self.result]

    def final(self):
        final = self.result[-1][-1]
        return final[-1] if isinstance(final, (list, tuple)) else final

    def intermediate_flow(self):
        return self.result
