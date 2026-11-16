#!/bin/bash
#SBATCH --account=project_462001175
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=36:00:00
#SBATCH --partition=standard-g
#SBATCH --exclusive

#module load LUMI/24.03
#module load LUMI/25.03
module purge
module load Local-LAIF lumi-aif-singularity-bindings

CONTAINER=/scratch/project_462001175/paddle/lumi-multitorch-latest.sif

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

export PADDLE_DISTRI_BACKEND=nccl
export PADDLE_WITH_GLOO=0
export NCCL_DEBUG_SUBSYS=INIT

# Awesome CXI defaults from Alps and Frontier supercomputers
# https://www.olcf.ornl.gov/wp-content/uploads/OLCF_AI_Training_0417_2024.pdf
# https://docs.cscs.ch/software/communication/nccl/#using-nccl
# Also refer to HPE documentation
# https://support.hpe.com/hpesc/public/docDisplay?docId=dp00005991en_us&page=user/completion_queue_size_attribute.html
export NCCL_CROSS_NIC=1
export FI_MR_CACHE_MONITOR=userfaultfd     # Required to avoid a deadlock.
export FI_CXI_DEFAULT_CQ_SIZE=131072  # Ask the network stack to allocate additional space to process message completions.
#export FI_CXI_DEFAULT_CQ_SIZE=524288  # Ask the network stack to allocate additional space to process message completions.
export FI_CXI_RX_MATCH_MODE=hybrid    # software (safe) or hybrid to allow the network stack to transition to software mode only if necessary.
export FI_CXI_DEFAULT_TX_SIZE=4096
#export FI_CXI_DEFAULT_TX_SIZE=8192
# Starting with nccl 2.27 a new protocol (LL128) was enabled by default, which
# typically performs worse on Slingshot. The following disables that protocol.
export NCCL_PROTO=^LL128
export PYTHONUNBUFFERED=1

# latvian-large-may2026-200: lr=1e-05
# latvian-large-may2026-200-5e05: lr=5e-05

# John note:
# IMPORTANTLY, we must run rocm-smi before starting the job to initialize some mutexes
# Otherwise, Paddlepaddle will start multiple processes that try access ROCm SMI and initialize that mutex
# leading to a race condition

RUN_CMD="trap 'rm -rf -- /tmp/archxai-latvian-split-paddle /tmp/synthgen' EXIT INT TERM; rm -rf -- /tmp/archxai-latvian-split-paddle /tmp/synthgen && \
rm -rf -- /tmp/archxai-latvian-split-paddle /tmp/synthgen && \
rocm-smi || true && \
rocm-smi --showid || true && \
ulimit -s unlimited; ulimit -n 65536 && \
rm -rf $MIOPEN_USER_DB_PATH && \
mkdir -p $MIOPEN_USER_DB_PATH && \
tar xf /scratch/project_462001175/paddle/synthgen.tar -C /tmp/ && \
mkdir /tmp/archxai-latvian-split-paddle && \
tar xf /scratch/project_462001175/paddle/archxai-latvian-split-november2026.tar -C /tmp/archxai-latvian-split-paddle && \
tar xf /scratch/project_462001175/paddle/latvian-national-library.tar -C /tmp/archxai-latvian-split-paddle && \
export WORKON_HOME=/root/.virtualenvs && \
source /opt/venv/bin/virtualenvwrapper.sh && \
workon paddle-venv && \
cd ~/paddle/PaddleOCR && \
../launch_wider_multinode_latvian.sh $MAIN_PROCESS_IP $SLURM_NNODES
"


srun singularity exec --overlay /scratch/project_462001175/paddle/overlay2.img:ro -B /flash/project_462001175,/scratch/project_462001175,/opt/cray $CONTAINER bash -c "$RUN_CMD"
