export CUDA_VISIBLE_DEVICES=0,1
PROJ_PATH=[YOUR PROJECT DIRECTORY]
export PYTHONPATH=$PYTHONPATH:${PROJ_PATH}

python -m torch.distributed.launch --nproc_per_node=2 --master_port 29664 \
src/tasks/run_video_retrieval.py \
--config src/configs/msrvtt_retrieval/msrvtt_retrieval_vip_base_32.json \
--blob_mount_dir ${PROJ_PATH}