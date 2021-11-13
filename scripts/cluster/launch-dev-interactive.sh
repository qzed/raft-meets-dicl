#!/usr/bin/env bash
salloc -p dev_gpu_4 --nodes=1 --ntasks=10 --mem=24gb --gres=gpu:2 "${@}"
