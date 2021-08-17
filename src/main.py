import argparse

from . import cmd


# example usage:
# - basic training
#   {prog} train --data strategy.yaml --model model.yaml
#   {prog} train --data strategy.yaml --modelm model.yaml -inspect metrics.yaml
#   {prog} train --config config.yaml
#
# - override parts of config
#   {prog} train --config config.yaml --strategy data.yaml
#
# - training with warm start from previous checkpoint (only model weights)
#   {prog} train --data data.yaml --model model.yaml --checkpoint checkpoint.pth
#
# - continue previous run from checkpoint (only epoch granularity, includes optimizer state etc.)
#   {prog} train --config config.yaml --continue checkpoint.pth
#   {prog} train --data data.yaml --model model.yaml --continue checkpoint.pth
#
# - reproduce previous training run
#   {prog} train --config config.yaml --reproduce
#
# - trim checkpoints (keep only best and/or latest checkpoints)
#   [prog] checkpoint trim ./directory/ --compare "{m_EndPointError_mean}" --keep-best 5
#   [prog] checkpoint trim ./directory/ --compare "{m_EndPointError_mean}" --keep-latest 5
#
# - show info for checkpoints
#   [prog] checkpoint info checkpoint.pth
#   [prog] checkpoint info ./directory/ --sort "{m_EndPointError_mean}"


def main():
    def fmtcls(prog): return argparse.HelpFormatter(prog, max_help_position=42)

    # main parser
    parser = argparse.ArgumentParser(description='Optical Flow Estimation', formatter_class=fmtcls)
    subp = parser.add_subparsers(dest='command', help='help for command')

    # subcommand: train
    train = subp.add_parser('train', aliases=['t'], formatter_class=fmtcls, help='train model')
    train.add_argument('-c', '--config', help='full training configuration')
    train.add_argument('-d', '--data', help='training strategy and data')
    train.add_argument('-m', '--model', help='specification of the model')
    train.add_argument('-s', '--seeds', help='seed config for initializing RNGs')
    train.add_argument('-i', '--inspect', help='specification of metrics')
    train.add_argument('-o', '--output', default='runs', help='base output directory [default: %(default)s]')
    train.add_argument('--device', help='device to use [default: cuda:0 if available]')
    train.add_argument('--device-ids', help='device IDs to use with DataParallel')
    train.add_argument('--checkpoint', help='start with pre-trained model state from checkpoint')
    train.add_argument('--resume', help='resume trainig from checkpoint (full state)')
    train.add_argument('--start-stage', type=int, help='start with sepcified stage and skip previous')
    train.add_argument('--start-epoch', type=int, help='start with sepcified epoch and skip previous')
    train.add_argument('--reproduce', action='store_true', help='use seeds from config')
    train.add_argument('--debug', action='store_true', help='enter debugger on exception')
    train.add_argument('--detect-anomaly', action='store_true', help='enable pytorch anomaly detection')
    train.add_argument('--suffix', '--sfx', dest='suffix', help='suffix for output directory')

    # subcommand: eval[uate]
    eval = subp.add_parser('evaluate', aliases=['e', 'eval'], formatter_class=fmtcls, help='evaluate model')
    eval.add_argument('-d', '--data', required=True, help='evaluation dataset')
    eval.add_argument('-m', '--model', required=True, help='the model to ue')
    eval.add_argument('-c', '--checkpoint', required=True, help='the checkpoint to load')
    eval.add_argument('-b', '--batch-size', type=int, default=1, help='batch-size to use for evaluation')
    eval.add_argument('-x', '--metrics', help='specification of metrics to use for evaluation')
    eval.add_argument('-o', '--output', help='write detailed output to this file (json or yaml)')
    eval.add_argument('-f', '--flow', help='compute and write flow images to specified directory')
    eval.add_argument('--flow-format', default='visual:flow', help='output format for flow images [default: visual:flow]')
    eval.add_argument('--flow-mrm', type=float, help='maximum range of motion for visual flow image output')
    eval.add_argument('--flow-gamma', type=float, help='gamma for visual:flow image output')
    eval.add_argument('--flow-transform', help='transform for visual:flow:dark image output')
    eval.add_argument('--flow-only', action='store_true', help='only compute flow images, do not evaluate metrics')
    eval.add_argument('--epe-cmap', default='gray', help='colormap for end-point-error visualization')
    eval.add_argument('--epe-max', type=float, default=None, help='maximum end point error for visualization')
    eval.add_argument('--device', help='device to use [default: cuda:0 if available]')
    eval.add_argument('--device-ids', help='device IDs to use with DataParallel')

    # subcommand: checkpoint
    chkpt = subp.add_parser('checkpoint', formatter_class=fmtcls, help='inspect and manage checkpoints')
    chkpt_sub = chkpt.add_subparsers(dest='subcommand', help='help for subcommand')

    chkpt_info = chkpt_sub.add_parser('info', formatter_class=fmtcls, help='show info on checkpoint(s)')
    chkpt_info.add_argument('file', nargs='+', help='checkpoint file or directory to search for checkpoints')
    chkpt_info.add_argument('--sort', help='expression(s) for sorting checkpoints (separated by comma)')

    chkpt_trim = chkpt_sub.add_parser('trim', formatter_class=fmtcls, help='remove bad and/or outdated checkpoints')
    chkpt_trim.add_argument('directory', nargs='+', help='directory to search for checkpoints')
    chkpt_trim.add_argument('--compare', help='expression(s) for comparing checkpoints (separated by comma)')
    chkpt_trim.add_argument('--keep-latest', type=int, help='keep specified number of latest checkpoints')
    chkpt_trim.add_argument('--keep-best', type=int, help='keep specified number of best checkpoints')

    # parse arguments
    args = parser.parse_args()

    # run subcommand
    commands = {
        'train': cmd.train,
        'evaluate': cmd.evaluate,
        'checkpoint': cmd.checkpoint,
    }

    commands[args.command](args)
