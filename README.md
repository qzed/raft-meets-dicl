# Optical Flow Estimation Framework - Work in Progress

Framework for developing and evaluating optical flow estimation methods based on neural networks.


## Setup

This project uses `pipenv`, i.e. in terms of code, setup is as easy installing `pipenv` (e.g. via `python -m pip install --user pipenv`) and running `pipenv install` (or `pipenv install -d` if you want to also install development dependencies).
Commands and scripts of this project can then be run either in a shell opened with `pipenv shell` or directly via `pipenv run <command>`.
The `pipenv` command can also be used to generate a `requirements.txt` file for integration into other python environments.


### Datasets

Datasets are expected to be placed in `../datasets`, i.e. a `datasets` folder next to the folder containing this code.
The dataset folder should follow the specification below:
```
../datasets
├── <some-dataset-id>
│  └── data                         # the dataset root directory
```
For all supported datasets see `cfg/data/dataset/`.
The `path` attribute of these config files specifies the path local to the config file and may be adapted if needed.

A subset of supported datasets would look like this:
```
../datasets
├── hci-hd1k                        # HCI HD1K
│  └── data
│     ├── hd1k_challenge
│     ├── hd1k_flow_gt
│     ├── hd1k_flow_uncertainty
│     └── hd1k_input
├── kitti-flow-2012                 # KITTI 2012
│  └── data
│     ├── testing
│     └── training
├── mpi-sintel-flow                 # MPI Sintel
│  └── data
│     ├── bundler
│     ├── flow_code
│     ├── test
│     ├── training
│     └── README.txt
├── ufreiburg-flyingchairs          # Uni Freiburg FlyingChairs
│  ├── data
│  │  ├── data
│  │  └── README.txt
│  └── train_val.txt
└── ufreiburg-flyingthings3d        # Uni Freiburg FlyingThings3D
   └── data
      ├── frames_cleanpass
      ├── frames_finalpass
      └── optical_flow
```


## Training

Training is done via
```
./main.py train -d <strategy> -m <model>
```
where `<strategy>` is a strategy config specifying training stages, i.e. data and optimizer parameters, and `<model>` is a model specification file, describing the model parameters.
For strategy config files see `cfg/strategy/`, for model config files see `cfg/model/`.
Config files are documented in `cfg/_doc/`.

By default, training will store tensorboard and other log files in the `runs/<timestamp>/` directory.
If specified in the training strategy and stages, checkpoints will be stored in `runs/<timestamp>/checkpoints/`.
Metrics etc. can be customized via an inspection config

See `./main.py train -h` for more information.


## Evaluation

Saved checkpoints can be evaluated via
```
./main.py evaluate -d <data> -m <model> -c <checkpoint>
```
where `<data>` is a data source specification (`cfg/data/`), `<model>` is a model specification (`cfg/model/`), and `<checkpoint>` is the checkpoint to evaluate.
Note that the model specification should match the one used for training the model (i.e. generating the checkpoint).

By default, this prints a set of metrics per sample and a summary collecting means.
This can e.g. be adapted via a custom evaluation config file.
The `eval` sub-command can also be used to visualize flow, visualize error metrics, and generate flow files for submission.

See `./main.py eval -h` for more information.


### Using original RAFT/DICL checkpoints

Original RAFT and DICL checkpoints can be used for evaluation with this framework.
To do this, the checkpoints have to be converted via
```
./scripts/chkpt_convert.py -i <original> -o <output> -f <format>
```
where `<original>` is the original checkpoint file, `<output>` is the output file (to be generated) and `<format>` is the format of the original checkpoint, i.e. one of `dicl` or `raft`.
