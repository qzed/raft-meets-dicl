import sys
import types
import tempfile
import pandas
import types

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import src

from pathlib import Path


MODEL_CFG = """
name: RAFT baseline config
id: raft/baseline

model:
  type: raft/baseline

  parameters:
    droput: 0.0
    mixed-precision: false

  arguments:
    iterations: {iter}

loss:
  type: raft/sequence

  arguments:
    ord: 1
    gamma: 0.85

input:
  clip: [0, 1]
  range: [-1, 1]

  padding:
    type: modulo
    mode: zeros
    size: [8, 8]

"""

DATASETS = {
    'things': {
        'sintel-clean': Path(__file__).parent / 'cfg' / 'data' / 'mpi-sintel-clean.train-full.yaml',
        'sintel-final': Path(__file__).parent / 'cfg' / 'data' / 'mpi-sintel-final.train-full.yaml',
    },
    'sintel': {
        'sintel-clean': Path(__file__).parent / 'cfg' / 'data' / 'mpi-sintel-clean.val.yaml',
        'sintel-final': Path(__file__).parent / 'cfg' / 'data' / 'mpi-sintel-final.val.yaml',
    },
}

CHECKPOINTS = {
    'raft.04': {
        'things': Path(__file__).parent / 'raft-baseline-i04.flyingthings3d.pth',
        'sintel': Path(__file__).parent / 'raft-baseline-i04.sintel.pth',
    },
    'raft.12': {
        'things': Path(__file__).parent / 'raft-baseline-i12.flyingthings3d.pth',
        'sintel': Path(__file__).parent / 'raft-baseline-i12.sintel.pth',
    },
}

ITERATIONS = list(range(1, 48 + 1))

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


def write_model(model_file, iter):
    model_file.seek(0)
    model_file.truncate(0)
    model_file.write(bytes(MODEL_CFG.format(iter=iter), 'utf-8'))
    model_file.flush()


def main():
    # evaluate
    with tempfile.NamedTemporaryFile(suffix='.yaml') as model_file:
        model_path = model_file.name

        for iter in ITERATIONS:
            for model, stages in CHECKPOINTS.items():
                for stage, checkpoint in stages.items():
                    for dataset_name, dataset_path in DATASETS[stage].items():
                        basename = f"{model}.{stage}.{dataset_name}.{iter}"
                        output = OUTPUT_DIR / f"{basename}.json"

                        print(f"== EVALUATING {basename} ============")

                        write_model(model_file, iter)
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
