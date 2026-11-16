#!/bin/bash
#SBATCH --account=project_462001175
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --partition=dev-g

#module load LUMI/24.03
module load LUMI/25.03

CONTAINER=/flash/project_462001175/structuralizer/structuralizer-amd-rocm63.sif

NUM_PROCESSES=$(expr $SLURM_NNODES \* $SLURM_GPUS_PER_NODE)
MAIN_PROCESS_IP=$(hostname -i)

# MIOPEN needs some initialisation for the cache as the default location
# does not work on LUMI as Lustre does not provide the necessary features.
export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-$SLURM_NODEID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH

if [ $SLURM_LOCALID -eq 0 ] ; then
    rm -rf $MIOPEN_USER_DB_PATH
    mkdir -p $MIOPEN_USER_DB_PATH
fi
sleep 2

export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3  # allow using NICs up to topology level 3, given by CSC

#export NCCL_P2P_LEVEL=SYS  # whether to only allow using within-node P2P through fastest possible interconnect or some slower paths as well (=AMD Infinity Fabric, equivalent to NVLink)

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT

export TRANSFORMERS_OFFLINE=1  # Disable this for the first run to download checkpoints


RUN_CMD="trap 'rm -rf -- /tmp/archxai-latvian-split' EXIT INT TERM; rm -rf -- /tmp/archxai-latvian-split && \
source /uv/env && \
source /code/train-trocr-main/.venv/bin/activate && \
cd \$HOME/tests/train-trocr-main && \
export HF_HOME=/flash/project_462001175/trocr/hf && \
mkdir /tmp/archxai-latvian-split && \
tar -xf /scratch/project_462001175/paddle/archxai-latvian-split-november2026.tar -C /tmp/archxai-latvian-split/ && \
echo \"Finished extracting\" && \
python test_simple.py --root_path /tmp/archxai-latvian-split/ \
    --test_data_path /tmp/archxai-latvian-split/test.csv --experiment_name latvian-november2026 \
    --model_path /scratch/project_462001175/trocr/outputs/latvian-large-november2026 \
    --batch_size 8 \
    --img_height 192 --img_width 1024 --no_binning \
    --output_path /users/makelajo/metrics/latvian-extremenumeralist-july2026"


srun singularity exec -B /flash/project_462001175,/scratch/project_462001175,/opt/cray $CONTAINER bash -c "$RUN_CMD"
