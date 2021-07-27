from pathlib import Path

from ..strategy import Checkpoint
from ..strategy.checkpoint import load_directory


def checkpoint(args):
    commands = {
        'info': info,
        'trim': trim,
    }

    commands[args.subcommand](args)


def info(args):
    # set up comparator for sorting
    compare = args.sort
    if not compare:
        compare = '{n_stage}, {n_epoch}, {n_steps}'

    compare = [expr.strip() for expr in compare.split(',')]

    # print info
    for path in args.file:
        path = Path(path)

        if path.is_file():
            chkpt = Checkpoint.load(path, map_location='cpu').to_entry(path)

            info = [
                f"stage: {chkpt.idx_stage}",
                f"epoch: {chkpt.idx_epoch}",
                f"step: {chkpt.idx_step}",
            ]
            info += [f"{k}: {v:.04f}" for k, v in chkpt.metrics.items()]

            print()
            print(f"File: '{path}', Model: {chkpt.model}")
            print(f"  {', '.join(info)}")

        else:
            for mgr in load_directory(path, compare):
                print()
                print(f"Directory: '{path}', Model: {mgr.model_id}")

                checkpoints = sorted(mgr.checkpoints, key=mgr._chkpt_sort_key_best)
                for chkpt in checkpoints:
                    info = [
                        f"stage: {chkpt.idx_stage}",
                        f"epoch: {chkpt.idx_epoch}",
                        f"step: {chkpt.idx_step}",
                    ]
                    info += [f"{k}: {v:.04f}" for k, v in chkpt.metrics.items()]

                    print(f"  {', '.join(info)}")


def trim(args):
    if args.keep_best and not args.compare:
        raise ValueError('option --compare must be specified when --keep-best is specified')

    if not args.keep_best and not args.keep_latest:
        raise ValueError('need to specify --keep-best or --keep-latest (or both)')

    # set up comparator for sorting
    compare = args.compare
    if not compare:
        compare = '{n_stage}, {n_epoch}, {n_steps}'

    compare = [expr.strip() for expr in compare.split(',')]

    # trim
    for path in args.directory:
        for mgr in load_directory(path, compare):
            mgr.trim(args.keep_best, args.keep_latest)
