export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=$PYTHONPATH:/home/xiaojian/clipvip

python -m torch.distributed.launch --nproc_per_node=2 --master_port 29665 \
src/tasks/msrvtt_eval.py \
--config src/configs/msrvtt_retrieval/msrvtt_32_eval.json \
--blob_mount_dir /home/xiaojian/clipvip 
