#!/bin/bash
#SBATCH --account=project_462001175
#SBATCH --nodes=16
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=48:00:00
#SBATCH --partition=standard-g

#module load LUMI/24.03
module load LUMI/25.03

CONTAINER=/flash/project_462001175/structuralizer/structuralizer-amd-rocm63.sif

NUM_PROCESSES=$(expr $SLURM_NNODES \* $SLURM_GPUS_PER_NODE)
MAIN_PROCESS_IP=$(hostname -i)

# MIOPEN needs some initialisation for the cache as the default location
# does not work on LUMI as Lustre does not provide the necessary features.
export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH

export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3  # allow using NICs up to topology level 3, given by CSC. "Remove this setting if you encounter a hang/crash." according to Alps training

#export NCCL_P2P_LEVEL=SYS  # whether to only allow using within-node P2P through fastest possible interconnect or some slower paths as well (=AMD Infinity Fabric, equivalent to NVLink)

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT

export TRANSFORMERS_OFFLINE=1  # Disable this for the first run to download checkpoints

# Awesome CXI defaults from Alps and Frontier supercomputers
# https://www.olcf.ornl.gov/wp-content/uploads/OLCF_AI_Training_0417_2024.pdf
# https://docs.cscs.ch/software/communication/nccl/#using-nccl
# Also refer to HPE documentation
# https://support.hpe.com/hpesc/public/docDisplay?docId=dp00005991en_us&page=user/completion_queue_size_attribute.html
export NCCL_CROSS_NIC=1
#export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_MR_CACHE_MONITOR=userfaultfd     # Required to avoid a deadlock.
export FI_CXI_DEFAULT_CQ_SIZE=131072  # Ask the network stack to allocate additional space to process message completions.
#export FI_CXI_DEFAULT_TX_SIZE=2048    # 2048 Ask the network stack to allocate additional space to hold pending outgoing messages.
export FI_CXI_RX_MATCH_MODE=hybrid    # Allow the network stack to transition to software mode if necessary.
export FI_CXI_DEFAULT_TX_SIZE=4096
# Starting with nccl 2.27 a new protocol (LL128) was enabled by default, which
# typically performs worse on Slingshot. The following disables that protocol.
export NCCL_PROTO=^LL128
export PYTHONUNBUFFERED=1

# latvian-large-may2026-200: lr=1e-05
# latvian-large-may2026-200-5e05: lr=5e-05

RUN_CMD="trap 'rm -rf -- /tmp/archxai-latvian-split /tmp/gen /tmp/sd19' EXIT INT TERM; rm -rf -- /tmp/archxai-latvian-split /tmp/gen /tmp/sd19 && \
rm -rf $MIOPEN_USER_DB_PATH && \
mkdir -p $MIOPEN_USER_DB_PATH && \
TROCR_LOG=/flash/project_462001175/trocr/trocr-logs/${SLURM_JOB_ID}-\$SLURM_NODEID.log && \
mkdir -p \$(dirname \$TROCR_LOG) && \
source /uv/env && \
source /code/train-trocr-main/.venv/bin/activate && \
cd \$HOME/tests/train-trocr-main && \
export HF_HOME=/flash/project_462001175/trocr/hf && \
mkdir /tmp/archxai-latvian-split && \
tar -xf /scratch/project_462001175/paddle/archxai-latvian-split-november2026.tar -C /tmp/archxai-latvian-split/ && \
mkdir /tmp/gen && \
tar -xf /scratch/project_462001175/trocr/archxai-2026/synthgen_data.tar -C /tmp/gen/ && \
tar -xf /scratch/project_462001175/trocr/archxai-2026/dida_segmented.tar -C /tmp/gen/ && \
echo \"Finished extracting\" && \
cp -r /flash/project_462001175/sd19 /tmp/ && \
accelerate launch \
    --config_file="\$HF_HOME/accelerate/multinode_config.yaml" \
    --num_processes=$NUM_PROCESSES \
    --num_machines=$SLURM_NNODES \
    --machine_rank=\$SLURM_NODEID \
    --main_process_ip=$MAIN_PROCESS_IP \
    train_simple.py --root_path /tmp/archxai-latvian-split/ --tr_data_path /tmp/archxai-latvian-split/train.csv \
    --val_data_path /tmp/archxai-latvian-split/val.csv --model_name latvian-large-november2026 \
    --model_path microsoft/trocr-large-handwritten \
    --output_path /scratch/project_462001175/trocr/outputs/ --batch_size 8 --eval_batch_size 1 --epochs 400 --lr 5e-05 --augment True --seed 41 --warmup_ratio 0.05 --numeric_proportion 0.1 > \$TROCR_LOG"


srun singularity exec -B /flash/project_462001175,/scratch/project_462001175,/opt/cray $CONTAINER bash -c "$RUN_CMD"
