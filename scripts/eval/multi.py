import sys
import types
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import src

from dataclasses import dataclass
from pathlib import Path
from typing import Dict


DIR_OUT = Path("multieval")


@dataclass
class Stage:
    model: str
    checkpoint: str
    data: Dict[str, str]


@dataclass
class Model:
    stages: Dict[str, Stage]


ranges = (32, 64, 128, 256)

data_chairs = {"chairs2": "cfg/data/ufreiburg-flyingchairs2.test.yaml"}
data_chairs |= {f"chairs2.r{rng}": f"cfg/data/ufreiburg-flyingchairs2.test-r{rng}.yaml" for rng  in ranges}

data_things = {
    "sintel-clean": "cfg/data/mpi-sintel-clean.train-full.yaml",
    "sintel-final": "cfg/data/mpi-sintel-final.train-full.yaml",
}
data_things |= {f"sintel-clean.r{rng}": f"cfg/data/mpi-sintel-clean.train-full-r{rng}.yaml" for rng  in ranges}
data_things |= {f"sintel-final.r{rng}": f"cfg/data/mpi-sintel-final.train-full-r{rng}.yaml" for rng  in ranges}

data_sintel = {
    "sintel-clean": "cfg/data/mpi-sintel-clean.val.yaml",
    "sintel-final": "cfg/data/mpi-sintel-final.val.yaml",
}
data_sintel |= {f"sintel-clean.r{rng}": f"cfg/data/mpi-sintel-clean.val-r{rng}.yaml" for rng  in ranges}
data_sintel |= {f"sintel-final.r{rng}": f"cfg/data/mpi-sintel-final.val-r{rng}.yaml" for rng  in ranges}

models = {
    "raft-sl-ctf2l": Model(
        stages={
            "chairs2": Stage(
                model="../results/runs/dev-1/raft-sl-ctf2l.flyingchairs2/config.json",
                checkpoint="../results/runs/dev-1/raft-sl-ctf2l.flyingchairs2/checkpoints/raft_sl.ctf.l2-s1_e14_b111150-epe1.1731.pth",
                data=data_chairs,
            ),
            "things": Stage(
                model="../results/runs/dev-1/raft-sl-ctf2l.flyingthings3d/config.json",
                checkpoint="../results/runs/dev-1/raft-sl-ctf2l.flyingthings3d/checkpoints/raft_sl.ctf.l2-s0_e10_b100516-epe1.7419.pth",
                data=data_things,
            ),
            "sintel": Stage(
                model="../results/runs/dev-1/raft-sl-ctf2l.sintel/config.json",
                checkpoint="../results/runs/dev-1/raft-sl-ctf2l.sintel/checkpoints/raft_sl.ctf.l2-s0_e249_b102000-epe2.0841.pth",
                data=data_sintel,
            ),
        }
    ),
    "raft-sl-ctf3l": Model(
        stages={
            "chairs2": Stage(
                model="../results/runs/dev-1/raft-sl-ctf3l.flyingchairs2/config.json",
                checkpoint="../results/runs/dev-1/raft-sl-ctf3l.flyingchairs2/checkpoints/raft_sl.ctf.l3-s1_e14_b111150-epe1.2092.pth",
                data=data_chairs,
            ),
            "things": Stage(
                model="../results/runs/dev-1/raft-sl-ctf3l.flyingthings3d/config.json",
                checkpoint="../results/runs/dev-1/raft-sl-ctf3l.flyingthings3d/checkpoints/raft_sl.ctf.l3-s0_e10_b100523-epe1.5909.pth",
                data=data_things,
            ),
            "sintel": Stage(
                model="../results/runs/dev-1/raft-sl-ctf3l.sintel/config.json",
                checkpoint="../results/runs/dev-1/raft-sl-ctf3l.sintel/checkpoints/raft_sl.ctf.l3-s0_e250_b102000-epe2.1264.pth",
                data=data_sintel,
            ),
        }
    ),
    "raft-sl-ctf4l": Model(
        stages={
            "chairs2": Stage(
                model="../results/runs/dev-1/raft-sl-ctf4l.flyingchairs2/config.json",
                checkpoint="../results/runs/dev-1/raft-sl-ctf4l.flyingchairs2/checkpoints/raft_sl.ctf.l4-s1_e14_b111150-epe1.2723.pth",
                data=data_chairs,
            ),
            "things": Stage(
                model="../results/runs/dev-1/raft-sl-ctf4l.flyingthings3d/config.json",
                checkpoint="../results/runs/dev-1/raft-sl-ctf4l.flyingthings3d/checkpoints/raft_sl.ctf.l4-s0_e10_b100517-epe1.6752.pth",
                data=data_things,
            ),
            "sintel": Stage(
                model="../results/runs/dev-1/raft-sl-ctf4l.sintel/config.json",
                checkpoint="../results/runs/dev-1/raft-sl-ctf4l.sintel/checkpoints/raft_sl.ctf.l4-s0_e249_b102000-epe2.0409.pth",
                data=data_sintel,
            ),
        }
    ),
    "raft+dicl-ctf2l": Model(
        stages={
            "chairs2": Stage(
                model="../results/runs/dev-1-prep1e4/raft+dicl-ctf2l.flyingchairs2/config.json",
                checkpoint="../results/runs/dev-1-prep1e4/raft+dicl-ctf2l.flyingchairs2/checkpoints/raft+dicl_ctf.l2-s1_e14_b111150-epe1.3773.pth",
                data=data_chairs,
            ),
            "things": Stage(
                model="../results/runs/dev-1-prep1e4/raft+dicl-ctf2l.flyingthings3d/config.json",
                checkpoint="../results/runs/dev-1-prep1e4/raft+dicl-ctf2l.flyingthings3d/checkpoints/raft+dicl_ctf.l2-s0_e10_b100524-epe1.8688.pth",
                data=data_things,
            ),
            "sintel": Stage(
                model="../results/runs/dev-1-prep1e4/raft+dicl-ctf2l.sintel/config.json",
                checkpoint="../results/runs/dev-1-prep1e4/raft+dicl-ctf2l.sintel/checkpoints/raft+dicl_ctf.l2-s0_e249_b102000-epe2.2251.pth",
                data=data_sintel,
            ),
        }
    ),
    "raft+dicl-ctf3l": Model(
        stages={
            "chairs2": Stage(
                model="../results/runs/dev-1-prep5e5/raft+dicl-ctf3l.flyingchairs2/config.json",
                checkpoint="../results/runs/dev-1-prep5e5/raft+dicl-ctf3l.flyingchairs2/checkpoints/raft+dicl_ctf.l3-s1_e14_b111150-epe1.2745.pth",
                data=data_chairs,
            ),
            "things": Stage(
                model="../results/runs/dev-1-prep5e5/raft+dicl-ctf3l.flyingthings3d/config.json",
                checkpoint="../results/runs/dev-1-prep5e5/raft+dicl-ctf3l.flyingthings3d/checkpoints/raft+dicl_ctf.l3-s0_e10_b100514-epe1.7405.pth",
                data=data_things,
            ),
            "sintel": Stage(
                model="../results/runs/dev-1-prep5e5/raft+dicl-ctf3l.sintel/config.json",
                checkpoint="../results/runs/dev-1-prep5e5/raft+dicl-ctf3l.sintel/checkpoints/raft+dicl_ctf.l3-s0_e249_b102000-epe2.0433.pth",
                data=data_sintel,
            ),
        }
    ),
    "raft+dicl-ctf4l": Model(
        stages={
            "chairs2": Stage(
                model="../results/runs/dev-1-prep5e6/raft+dicl-ctf4l.flyingchairs2/config.json",
                checkpoint="../results/runs/dev-1-prep5e6/raft+dicl-ctf4l.flyingchairs2/checkpoints/raft+dicl_ctf.l4-s1_e14_b111150-epe1.3097.pth",
                data=data_chairs,
            ),
            "things": Stage(
                model="../results/runs/dev-1-prep5e6/raft+dicl-ctf4l.flyingthings3d/config.json",
                checkpoint="../results/runs/dev-1-prep5e6/raft+dicl-ctf4l.flyingthings3d/checkpoints/raft+dicl_ctf.l4-s0_e10_b100517-epe1.7926.pth",
                data=data_things,
            ),
            "sintel": Stage(
                model="../results/runs/dev-1-prep5e6/raft+dicl-ctf4l.sintel/config.json",
                checkpoint="../results/runs/dev-1-prep5e6/raft+dicl-ctf4l.sintel/checkpoints/raft+dicl_ctf.l4-s0_e250_b102000-epe2.2615.pth",
                data=data_sintel,
            ),
        }
    ),
}


def do_evaluate(model, checkpoint, data, output):
    args = types.SimpleNamespace()

    args.device = None
    args.device_ids = None
    args.batch_size = 1

    args.model = model
    args.checkpoint = checkpoint
    args.data = data
    args.output = output
    args.metrics = None

    args.flow = None
    args.flow_only = False
    args.flow_format = None
    args.flow_mrm = None
    args.flow_gamma = None
    args.flow_transform = None
    args.epe_max = None
    args.epe_cmap = None

    src.cmd.eval.evaluate(args)


def path_validate(path):
    if not Path(path).is_file():
        raise RuntimeError(f"path does not exist: '{path}'")


def main():
    table = dict()

    # validate paths
    for model_name, model in models.items():

        for stage_name, stage in model.stages.items():
            path_validate(stage.model)
            path_validate(stage.checkpoint)

            for data_name, data in stage.data.items():
                path_validate(data)

    try:
        for model_name, model in models.items():
            table[model_name] = dict()

            for stage_name, stage in model.stages.items():
                table[model_name][stage_name] = dict()

                for data_name, data in stage.data.items():
                    output = f"{model_name}.{stage_name}.{data_name}.json"
                    output = DIR_OUT / output

                    # evaluate
                    do_evaluate(stage.model, stage.checkpoint, data, output)

                    # read back results
                    results = src.utils.config.load(output)
                    epe = results['summary']['mean']['EndPointError/mean']

                    table[model_name][stage_name][data_name] = epe

    finally:
        summary = json.dumps(table, indent=4, sort_keys=True)
        print(summary)

        with open(DIR_OUT / "summary.json", "w") as fd:
            fd.write(summary)


if __name__ == '__main__':
    main()
