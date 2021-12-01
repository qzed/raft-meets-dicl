#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --mem=48gb
#SBATCH --gres=gpu:2

# Example usage:
#
# sbatch -p dev_gpu_4 --time=00:15:00 ./scripts/cluster/train.sh    \
#        --config cfg/full/dev/dicl-baseline.flyingchairs2.json     \
#        --reproduce --suffix testing --comment "Some test run"

echo "============================== SETTING UP =============================="
echo ""

export PATH="$HOME/.pyenv/bin:$HOME/.local/bin:$HOME/bin:$PATH"
export PIPENV_VENV_IN_PROJECT="enabled"

eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"

module load devel/cuda/10.2
module load devel/cudnn/10.2

echo "executing: pipenv run ./main.py train --env cfg/env/cluster.yaml ${@}"

echo ""
echo "============================= STARTING JOB ============================="
echo ""
python -m pipenv run ./main.py train --env "cfg/env/cluster.yaml" "${@}"
