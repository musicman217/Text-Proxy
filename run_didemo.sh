export CUDA_VISIBLE_DEVICES=0
PROJ_PATH=[YOUR PROJECT DIRECTORY]
export PYTHONPATH=$PYTHONPATH:${PROJ_PATH}

python -m torch.distributed.launch --nproc_per_node=1 --master_port 29664 \
src/tasks/run_video_retrieval.py  \
--config src/configs/didemo_retrieval/didemo_retrieval_vip_base_32.json \
--blob_mount_dir ${PROJ_PATH}