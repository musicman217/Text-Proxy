# 【AAAI'2025】Text Proxy: Decomposing Retrieval from a 1-to-N Relationship into N 1-to-1 Relationships for Text-Video Retrieval
The implementation of AAAI 2025 paper [Text Proxy: Decomposing Retrieval from a 1-to-N Relationship into N 1-to-1 Relationships for Text-Video Retrieval](https://arxiv.org/abs/2410.06618).

# Getting Started

1. Download pretrained models.

   You could learn about CLIP-ViP (our baseline) and download CLIP-ViP pre-trained weights [here](https://github.com/microsoft/XPretrain/tree/main/CLIP-ViP), and we provide links to download:

   CLIP-ViP-B/32: [Azure Blob Link](https://hdvila.blob.core.windows.net/dataset/pretrain_clipvip_base_32.pt?sp=r&st=2023-03-16T05:02:41Z&se=2027-05-31T13:02:41Z&spr=https&sv=2021-12-02&sr=b&sig=91OEG2MuszQmr16N%2Bt%2FLnvlwY3sc9CNhbyxYT9rupw0%3D)

   CLIP-ViP-B/16: [Azure Blob Link](https://hdvila.blob.core.windows.net/dataset/pretrain_clipvip_base_16.pt?sp=r&st=2023-03-16T05:02:05Z&se=2026-07-31T13:02:05Z&spr=https&sv=2021-12-02&sr=b&sig=XNd7fZSsUhW7eesL3hTfYUMiAvCCN3Bys2TadXlWzFU%3D)

2. Download Datasets.

   You could download the MSR-VTT, DiDeMo and ActivityNet Captions [here](https://github.com/ArrowLuo/CLIP4Clip).

3. Compress Video.

   ```bash
   python preprocess/compress_video.py --input_root [raw_video_path] --output_root [compressed_video_path]
   ```

4. Setup code environment.

   ```
   conda create -n TextProxy python=3.8
   conda activate TextProxy
   pip install -r requirements.txt
   pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
   ```

5. Fine-tuning for text-video retrieval.

   ```
   bash run_{DATASET}.sh
   ```

# Acknowledgments
Our code is based on [CLIP-ViP](https://github.com/microsoft/XPretrain/tree/main/CLIP-ViP). We sincerely appreciate for their contributions. 

   
