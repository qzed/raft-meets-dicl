#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --mem=32gb
#SBATCH --gres=gpu:2

# Example usage:
#
# sbatch -p dev_gpu_4 --time=00:15:00 ./scripts/cluster/train.sh    \
#        --config cfg/full/dev/dicl-baseline.flyingchairs2.json     \
#        --reproduce --suffix testing --comment "Some test run"

echo "============================== SETTING UP =============================="
echo ""

export PIPENV_VENV_IN_PROJECT="enabled"

module load devel/python/3.9.5_gnu_11.1
module load devel/cuda/10.2
module load devel/cudnn/10.2

echo "executing: pipenv run ./main.py train --env cfg/env/cluster.yaml ${@}"

echo ""
echo "============================= STARTING JOB ============================="
echo ""
pipenv run ./main.py train --env "cfg/env/cluster.yaml" "${@}"
