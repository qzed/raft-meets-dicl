#!/usr/bin/env python

import sys
import types
import tempfile
import json

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import src


DIR_OUT = Path("out/flow")


@dataclass
class Stage:
    model: str
    checkpoint: str


@dataclass
class Model:
    stages: Dict[str, Stage]


#mask = {
#    "base": (),
#    "mask-3": (3,),
#    "mask-34": (3, 4),
#    "mask-345": (3, 4, 5),
#    "mask-3456": (3, 4, 5, 6),
#    "mask-456": (4, 5, 6),
#    "mask-56": (5, 6),
#    "mask-6": (6,),
#}

mask = {
    "base": (),
    "mask-3": (3,),
    "mask-34": (3, 4),
    "mask-4": (4,),
}


data = "cfg/data/mpi-sintel-clean.visual.yaml"


# dev-1/raft-baseline
# dev-1-prep1e4/raft+dicl-ml-l4
# dev-1-prep5e5/raft+dicl-ml-l2-pool-diclshared
# dev-1-prep5e5/raft+dicl-ml-l2-pool-diclshared-ifreg-nodap

models = {
#    "raft-original": Model(
#        stages={
#            "chairs": Stage(
#                model="cfg/model/raft-baseline.yaml",
#                checkpoint="chkpt/orig/raft/chkpt-chairs.pth",
#            ),
#            "things": Stage(
#                model="cfg/model/raft-baseline.yaml",
#                checkpoint="chkpt/orig/raft/chkpt-things.pth",
#            ),
#            "sintel": Stage(
#                model="cfg/model/raft-baseline.yaml",
#                checkpoint="chkpt/orig/raft/chkpt-sintel.pth",
#            ),
#        }
#    ),
    "raft+dicl-ml-l2-pool-diclshared-ifreg-nodap": Model(
        stages={
            "chairs-1": Stage(
                model="../results/runs/dev-1-prep5e5/raft+dicl-ml-l2-pool-diclshared-ifreg-nodap.flyingchairs2/config.json",
                checkpoint="../results/runs/dev-1-prep5e5/raft+dicl-ml-l2-pool-diclshared-ifreg-nodap.flyingchairs2/checkpoints/raft+dicl_ml-s0_e1_b7410-epe3.7037.pth",
            ),
            "chairs-15": Stage(
                model="../results/runs/dev-1-prep5e5/raft+dicl-ml-l2-pool-diclshared-ifreg-nodap.flyingchairs2/config.json",
                checkpoint="../results/runs/dev-1-prep5e5/raft+dicl-ml-l2-pool-diclshared-ifreg-nodap.flyingchairs2/checkpoints/raft+dicl_ml-s1_e14_b111150-epe1.6431.pth",
            ),
            "things-10": Stage(
                model="../results/runs/dev-1-prep5e5/raft+dicl-ml-l2-pool-diclshared-ifreg-nodap.flyingthings3d/config.json",
                checkpoint="../results/runs/dev-1-prep5e5/raft+dicl-ml-l2-pool-diclshared-ifreg-nodap.flyingthings3d/checkpoints/raft+dicl_ml-s0_e10_b100536-epe2.3690.pth",
            ),
            "sintel-250": Stage(
                model="../results/runs/dev-1-prep5e5/raft+dicl-ml-l2-pool-diclshared-ifreg-nodap.sintel/config.json",
                checkpoint="../results/runs/dev-1-prep5e5/raft+dicl-ml-l2-pool-diclshared-ifreg-nodap.sintel/checkpoints/raft+dicl_ml-s0_e250_b102000-epe2.9020.pth",
            ),
        }
    ),
}


def do_evaluate(model, checkpoint, data, flow_out):
    args = types.SimpleNamespace()

    args.device = None
    args.device_ids = None
    args.batch_size = 1

    args.model = model
    args.checkpoint = checkpoint
    args.data = data
    args.output = None
    args.metrics = None

    args.flow = flow_out
    args.flow_only = True
    args.flow_format = 'visual:intermediate:flow'
    args.flow_mrm = 60
    args.flow_gamma = None
    args.flow_transform = None
    args.epe_max = None
    args.epe_cmap = None

    src.cmd.eval.evaluate(args)


def path_validate(path):
    if not Path(path).is_file():
        raise RuntimeError(f"path does not exist: '{path}'")


def update_model(model_file, model_src, mask):
    model = src.utils.config.load(model_src)['model']
    # model = src.utils.config.load(model_src)
    model = src.models.load(model)

    model.model.arguments['mask_costs'] = mask
    model_cfg = json.dumps(model.get_config())

    model_file.seek(0)
    model_file.truncate(0)
    model_file.write(bytes(model_cfg, 'utf-8'))
    model_file.flush()


def main():
    # validate paths
    for model_name, model in models.items():
        for stage_name, stage in model.stages.items():
            path_validate(stage.model)
            path_validate(stage.checkpoint)

    with tempfile.NamedTemporaryFile(suffix='.yaml') as model_file:
        model_path = model_file.name

        for model_name, model in models.items():
            for stage_name, stage in model.stages.items():
                for mask_name, ms in mask.items():
                    output = f"{model_name}/{stage_name}/{mask_name}"  # TODO
                    output = DIR_OUT / output

                    # modify config
                    update_model(model_file, stage.model, ms)

                    # evaluate
                    do_evaluate(model_path, stage.checkpoint, data, output)


if __name__ == '__main__':
    main()
