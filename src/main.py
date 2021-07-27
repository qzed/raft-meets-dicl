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
    ptrain = subp.add_parser('train', formatter_class=fmtcls, help='train model')
    ptrain.add_argument('-c', '--config', help='full training configuration')
    ptrain.add_argument('-d', '--data', help='training strategy and data')
    ptrain.add_argument('-m', '--model', help='specification of the model')
    ptrain.add_argument('-i', '--inspect', help='specification of metrics')
    ptrain.add_argument('-o', '--output', default='runs', help='base output directory [default: %(default)s]')
    ptrain.add_argument('--device', help='device to use [default: cuda:0 if available]')
    ptrain.add_argument('--device-ids', help='device IDs to use with DataParallel')
    ptrain.add_argument('--checkpoint', help='start with pre-trained model state from checkpoint')
    ptrain.add_argument('--resume', help='resume trainig from checkpoint (full state)')
    ptrain.add_argument('--start-stage', type=int, help='start with sepcified stage and skip previous')
    ptrain.add_argument('--start-epoch', type=int, help='start with sepcified epoch and skip previous')
    ptrain.add_argument('--reproduce', action='store_true', help='use seeds from config')

    # subcommand: checkpoint
    pchkpt = subp.add_parser('checkpoint', formatter_class=fmtcls, help='inspect and manage checkpoints')
    pchkpt_sub = pchkpt.add_subparsers(dest='subcommand', help='help for subcommand')

    pchkpt_info = pchkpt_sub.add_parser('info', formatter_class=fmtcls, help='show info on checkpoint(s)')
    pchkpt_info.add_argument('file', nargs='+', help='checkpoint file or directory to search for checkpoints')
    pchkpt_info.add_argument('--sort', help='expression(s) for sorting checkpoints (separated by comma)')

    pchkpt_trim = pchkpt_sub.add_parser('trim', formatter_class=fmtcls, help='remove bad and/or outdated checkpoints')
    pchkpt_trim.add_argument('directory', nargs='+', help='directory to search for checkpoints')
    pchkpt_trim.add_argument('--compare', help='expression(s) for comparing checkpoints (separated by comma)')
    pchkpt_trim.add_argument('--keep-latest', type=int, help='keep specified number of latest checkpoints')
    pchkpt_trim.add_argument('--keep-best', type=int, help='keep specified number of best checkpoints')

    # parse arguments
    args = parser.parse_args()

    # run subcommand
    commands = {
        'train': cmd.train,
        'checkpoint': cmd.checkpoint,
    }

    commands[args.command](args)
