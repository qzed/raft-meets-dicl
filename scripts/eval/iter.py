#!/usr/bin/env python

import sys
import types
import tempfile
import pandas
import types
import json

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import src

basepath = Path(__file__).parent.parent.parent


CONFIGS = {
#    'raft-sl-ctf3l.433': {
#        'things': basepath / '../results/runs/dev-1/raft-sl-ctf3l.flyingthings3d/config.json',
#        'sintel': basepath / '../results/runs/dev-1/raft-sl-ctf3l.sintel/config.json',
#    },
#    'raft+dicl-ctf3l.433': {
#        'things': basepath / '../results/runs/dev-1-prep5e5/raft+dicl-ctf3l.flyingthings3d/config.json',
#        'sintel': basepath / '../results/runs/dev-1-prep5e5/raft+dicl-ctf3l.sintel/config.json',
#    },
    'raft+dicl-ctf3l.888': {
        'things': basepath / '../results/runs/dev-1-prep5e5/raft+dicl-ctf3l-i8.flyingthings3d/config.json',
#        'sintel': basepath / '../results/runs/dev-1-prep5e5/raft+dicl-ctf3l.sintel/config.json',
    },
}

DATASETS = {
    'things': {
        'sintel-clean': basepath / 'cfg' / 'data' / 'mpi-sintel-clean.train-full.yaml',
        'sintel-final': basepath / 'cfg' / 'data' / 'mpi-sintel-final.train-full.yaml',
    },
    'sintel': {
        'sintel-clean': basepath / 'cfg' / 'data' / 'mpi-sintel-clean.val.yaml',
        'sintel-final': basepath / 'cfg' / 'data' / 'mpi-sintel-final.val.yaml',
    },
}

CHECKPOINTS = {
#    'raft-sl-ctf3l.433': {
#        'things': basepath / '../results/runs/dev-1/raft-sl-ctf3l.flyingthings3d/checkpoints/raft_sl.ctf.l3-s0_e10_b100523-epe1.5909.pth',
#        'sintel': basepath / '../results/runs/dev-1/raft-sl-ctf3l.sintel/checkpoints/raft_sl.ctf.l3-s0_e250_b102000-epe2.1264.pth',
#    },
#    'raft+dicl-ctf3l.433': {
#        'things': basepath / '../results/runs/dev-1-prep5e5/raft+dicl-ctf3l.flyingthings3d/checkpoints/raft+dicl_ctf.l3-s0_e10_b100514-epe1.7405.pth',
#        'sintel': basepath / '../results/runs/dev-1-prep5e5/raft+dicl-ctf3l.sintel/checkpoints/raft+dicl_ctf.l3-s0_e249_b102000-epe2.0433.pth',
#    },
    'raft+dicl-ctf3l.888': {
        'things': basepath / '../results/runs/dev-1-prep5e5/raft+dicl-ctf3l-i8.flyingthings3d/checkpoints/raft+dicl_ctf.l3-s0_e10_b100514-epe1.6867.pth',
#        'sintel': basepath / 'TODO',
    },
}

ITERATIONS = list(range(1, 24 + 1))

OUTPUT_DIR = Path('eval-iter')


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


def write_model(model_file, cfg_file, iter):
    model = src.utils.config.load(cfg_file)['model']
    model = src.models.load(model)

    model.model.arguments['iterations'] = [iter, iter, iter]
    model_cfg = json.dumps(model.get_config())

    model_file.seek(0)
    model_file.truncate(0)
    model_file.write(bytes(model_cfg, 'utf-8'))
    model_file.flush()


def main():
    # evaluate
    with tempfile.NamedTemporaryFile(suffix='.yaml') as model_file:
        model_path = model_file.name

        for iter in ITERATIONS:
            for model, stages in CHECKPOINTS.items():
                for stage, checkpoint in stages.items():
                    cfg_file = CONFIGS[model][stage]

                    for dataset_name, dataset_path in DATASETS[stage].items():
                        basename = f"{model}.{stage}.{dataset_name}.{iter}"
                        output = OUTPUT_DIR / f"{basename}.json"

                        print(f"== EVALUATING {basename} ============")

                        write_model(model_file, cfg_file, iter)
                        do_evaluate(model_path, checkpoint, dataset_path, output)

                        print("")

    # collect
    data = {'iterations': ITERATIONS}

    for model, stages in CHECKPOINTS.items():
        for stage in stages.keys():
            for dataset_name in DATASETS[stage].keys():
                identifier = f"{model}.{stage}.{dataset_name}"

                collected_epe = list()
                for iter in ITERATIONS:
                    output = f"{model}.{stage}.{dataset_name}.{iter}.json"
                    output = OUTPUT_DIR / output

                    results = src.utils.config.load(output)
                    epe = results['summary']['mean']['EndPointError/mean']

                    collected_epe.append(epe)

                data[identifier] = collected_epe

    df = pandas.DataFrame(data)
    df.to_csv(OUTPUT_DIR / 'summary.csv', index=False)


if __name__ == '__main__':
    main()
