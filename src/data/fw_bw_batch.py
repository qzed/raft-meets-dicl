import numpy as np

from . import config
from .collection import Collection


class ForwardsBackwardsBatch(Collection):
    type = 'forwards-backwards-batch'

    @classmethod
    def from_config(cls, path, cfg):
        cls._typecheck(cfg)

        forwards = config.load(path, cfg['forwards'])
        backwards = config.load(path, cfg['backwards'])

        return cls(forwards, backwards)

    def __init__(self, forwards, backwards):
        super().__init__()

        assert len(forwards) == len(backwards)

        self.forwards = forwards
        self.backwards = backwards

    def get_config(self):
        return {
            'type': self.type,
            'forwards': self.forwards.get_config(),
            'backwards': self.backwards.get_config(),
        }

    def __getitem__(self, index):
        # Images are ordered alphabetically by key which is derived from the
        # first frame, so it should be safe to simply get pairs via the index.
        img1_fw, img2_fw, flow_fw, valid_fw, meta_fw = self.forwards[index]
        img1_bw, img2_bw, flow_bw, valid_bw, meta_bw = self.backwards[index]

        assert img1_fw.shape[:3] == img2_fw.shape[:3] == valid_fw.shape[:3]
        assert img1_bw.shape[:3] == img2_bw.shape[:3] == valid_bw.shape[:3]
        assert img1_fw.shape[:3] == img1_bw.shape[:3]
        assert len(meta_fw) == len(meta_bw) == img1_fw.shape[0]

        # Make sure that the pairs actually match
        for mf, mb in zip(meta_fw, meta_bw):
            assert mf.sample_id.img1 == mb.sample_id.img2
            assert mf.sample_id.img2 == mb.sample_id.img1

        # Concat to batch. Note: Batches get shuffled internally when
        # collating.
        img1 = np.concatenate((img1_fw, img1_bw), axis=0)
        img2 = np.concatenate((img2_fw, img2_bw), axis=0)

        if flow_fw is not None:
            flow = np.concatenate((flow_fw, flow_bw), axis=0)
            valid = np.concatenate((valid_fw, valid_bw), axis=0)

        return img1, img2, flow, valid, meta_fw + meta_bw

    def __len__(self):
        return len(self.forwards)

    def description(self):
        return f"Forwards/Backwards batch: '{self.forwards.description()}'"
