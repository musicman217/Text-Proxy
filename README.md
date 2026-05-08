<div align="center">
  
# 【AAAI'2025】Text Proxy: Decomposing Retrieval from a 1-to-N Relationship into N 1-to-1 Relationships for Text-Video Retrieval
[![Conference](https://img.shields.io/badge/AAAI-2025-ff69b4.svg)](https://aaai.org/Conferences/AAAI-25/)
[![Paper](http://img.shields.io/badge/Paper-arxiv.2410.06618-FF6B6B.svg)](https://arxiv.org/abs/2410.06618)
</div>

## 📚 Abstract
Text-video retrieval (TVR) has seen substantial advancements in recent years, fueled by the utilization of pre-trained models and large language models (LLMs). Despite these advancements, achieving accurate matching in TVR remains challenging due to inherent disparities between video and textual modalities and irregularities in data representation. In this paper, we propose Text-Video-ProxyNet (TV-ProxyNet), a novel framework designed to decompose the conventional 1-to-N relationship of TVR into N distinct 1-to-1 relationships. By replacing a single text query with a series of text proxies, TV-ProxyNet not only broadens the query scope but also achieves a more precise expansion. Each text proxy is crafted through a refined iterative process, controlled by mechanisms we term as the director and dash, which regulate the proxy's direction and distance relative to the original text query. This setup not only facilitates more precise semantic alignment but also effectively manages the disparities and noise inherent in multimodal data. Our experiments on three representative video-text retrieval benchmarks, MSRVTT, DiDeMo, and ActivityNet Captions, demonstrate the effectiveness of TV-ProxyNet. The results show an improvement of 2.0% to 3.3% in R@1 over the baseline. TV-ProxyNet achieved state-of-the-art performance on MSRVTT and ActivityNet Captions, and a 2.0% improvement on DiDeMo compared to existing methods, validating our approach's ability to enhance semantic mapping and reduce error propensity.

## 🚀 Finetune on MSR-VTT

### Setup code environment
```shell
conda create -n TextProxy python=3.8
conda activate TextProxy
pip install -r requirements.txt
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
```

### Download CLIP-ViP Model
```shell
cd tvr/models
wget https://hdvila.blob.core.windows.net/dataset/pretrain_clipvip_base_32.pt?sp=r&st=2023-03-16T05:02:41Z&se=2027-05-31T13:02:41Z&spr=https&sv=2021-12-02&sr=b&sig=91OEG2MuszQmr16N%2Bt%2FLnvlwY3sc9CNhbyxYT9rupw0%3D
# wget https://hdvila.blob.core.windows.net/dataset/pretrain_clipvip_base_16.pt?sp=r&st=2023-03-16T05:02:05Z&se=2026-07-31T13:02:05Z&spr=https&sv=2021-12-02&sr=b&sig=XNd7fZSsUhW7eesL3hTfYUMiAvCCN3Bys2TadXlWzFU%3D
```

### Download Datasets
You could download the MSR-VTT, DiDeMo and ActivityNet Captions from [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip).

### Compress Video
```sh
python preprocess/compress_video.py --input_root [raw_video_path] --output_root [compressed_video_path]
```
This script will compress the video to *3fps* with width *224* (or height *224*). Modify the variables for your customization.

###  Train on MSR-VTT
```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
DATA_PATH=/home/username/textproxy/data/MST-VTT
python -m torch.distributed.launch \
--master_port 29510 \
--nproc_per_node=4 \
main_retrieval.py \
--do_train 1 \
--workers 8 \
--n_display 50 \
--epochs 5 \
--lr 1e-4 \
--coef_lr 1e-3 \
--batch_size 128 \
--batch_size_val 128 \
--anno_path data/MSR-VTT/anns \
--video_path ${DATA_PATH}/3fps_videos \
--datatype msrvtt \
--max_words 32 \
--max_frames 12 \
--video_framerate 1 \
--output_dir ckpts/ckpt_msrvtt_retrieval \
--temp 3
```

## 🎗️ Acknowledgments
* This code implementation are adopted from [CLIP-ViP](https://github.com/microsoft/XPretrain/tree/main/CLIP-ViP), [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip), and [NVIDIA apex](https://github.com/NVIDIA/apex).
We sincerely appreciate for their contributions.

## 📝 Citation
If you find this code useful for your research, please consider citing our paper:

```bibtex
@inproceedings{xiao2025text,
  title={Text proxy: Decomposing retrieval from a 1-to-n relationship into n 1-to-1 relationships for text-video retrieval},
  author={Xiao, Jian and Hu, Zhenzhen and Li, Jia and Hong, Richang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={8},
  pages={8655--8663},
  year={2025}
}
```
```
