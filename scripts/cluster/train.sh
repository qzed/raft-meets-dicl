#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --mem=24gb
#SBATCH --gres=gpu:2

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
