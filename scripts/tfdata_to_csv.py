#!/usr/bin/env python3

import argparse
import sys

from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import src.utils.tfdata as tfdata


def main():
    # handle command-line input
    def fmtcls(prog): return argparse.HelpFormatter(prog, max_help_position=42)

    parser = argparse.ArgumentParser(description='Convert tensorboard scalar data to CSV', formatter_class=fmtcls)
    parser.add_argument('-d', '--data', required=True, help='the tensorboard log file')
    parser.add_argument('-t', '--tag', required=True, action='append', help='the tag to export')
    parser.add_argument('-o', '--output', required=True, action='append', help='output file')
    parser.add_argument('-a', '--ewm', type=float, help='alpha for (optional) exponential weighted moving average')

    args = parser.parse_args()

    if len(args.output) != len(args.tag):
        raise ValueError("must have one output file per tag")

    # load data
    print("loading data...")
    df = tfdata.tfdata_scalars_to_pandas(args.data, args.tag)

    # extract data for each tag and write it to CSV
    print("converting...")
    for (output, tag) in zip(args.output, args.tag):
        steps = df.loc[df.tag == tag].step
        values = df.loc[df.tag == tag].value

        df_out = pd.DataFrame()
        df_out['step'] = steps
        df_out['value'] = values

        if args.ewm is not None:
            values_ewm = values.ewm(alpha=args.ewm)
            df_out['value'] = values_ewm.mean()
            df_out['std'] = values_ewm.std().fillna(value=0.0)

        print(f"writing CSV data to '{output}'")
        df_out.to_csv(output, index=False)


if __name__ == '__main__':
    main()
