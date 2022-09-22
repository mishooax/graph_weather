#!/bin/bash -x

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=64
#SBATCH --mem=128Gb
#SBATCH --exclude=ac6-301
#SBATCH --time=06:00:00
#SBATCH --output=job_out.%j
#SBATCH --error=job_error.%j

# debugging flags (optional)
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=ib0,lo

# Name and notes optional
export WANDB_NAME="gnn-r1-bs2-mp9-lr2em3-gpu2-mlpnorm-none-newlatlons"
export WANDB_NOTES="DDP (2GPUs), 2x the learning rate (lr=2e-3), Huber loss, 9 message passing layers, MLP w/o output normalization, reindexed lat-lons."

# generic settings
CONDA_ENV=dev-gnn-gpu
GITDIR=/home/syma/dask/codes/graph_weather
WORKDIR=/home/syma/dask/codes/graph_weather

cd $WORKDIR
module load conda
conda activate $CONDA_ENV
srun gnn-wb-train --config $GITDIR/graph_weather/config/wb_config_atos.yaml
