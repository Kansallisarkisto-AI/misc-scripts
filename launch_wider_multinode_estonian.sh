python -m paddle.distributed.launch --log_dir log_wider_multinode_estonian_v4 --enable_gpu_log false --with_gloo=0 --devices 0,1,2,3,4,5,6,7 --master=$1:8090 --nnodes $2  tools/train.py -c ../PP-OCRv6_medium_rec_estonian_wider.yml \
        -o Global.pretrained_model=/scratch/project_462001175/paddle/PP-OCRv6_medium_rec_pretrained.pdparams
