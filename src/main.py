import argparse

from . import train


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
    ptrain.add_argument('--start-stage', type=int, default=1, help='start with sepcified stage and skip previous')
    ptrain.add_argument('--start-epoch', type=int, default=1, help='start with sepcified epoch and skip previous')

    # parse arguments
    args = parser.parse_args()

    # run subcommand
    if args.command == 'train':
        train.train(args)
