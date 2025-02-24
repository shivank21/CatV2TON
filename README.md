# CatV2TON: Taming Diffusion Transformers for Vision-Based Virtual Try-On with Temporal Concatenation

<div style="display: flex; justify-content: center; align-items: center;">
  <a href="https://arxiv.org/abs/2501.11325v1" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/arXiv-2501.11325-red?style=flat&logo=arXiv&logoColor=red' alt='arxiv'>
  </a>
  <a href='https://huggingface.co/zhengchong/CatV2TON' style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/HuggingFace-ckpts-orange?style=flat&logo=HuggingFace&logoColor=orange' alt='huggingface'>
  </a>
  <a href="https://github.com/Zheng-Chong/CatV2TON" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/GitHub-Repo-blue?style=flat&logo=GitHub' alt='GitHub'>
  </a>
  <!-- <a href="https://huggingface.co/spaces/zhengchong/CatVTON" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Demo-Gradio-gold?style=flat&logo=Gradio&logoColor=red' alt='Demo'>
  </a> -->
  <!-- <a href="https://huggingface.co/spaces/zhengchong/CatVTON" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Space-ZeroGPU-orange?style=flat&logo=Gradio&logoColor=red' alt='Demo'>
  </a> -->
  <!-- <a href='https://zheng-chong.github.io/CatVTON/' style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Webpage-Project-silver?style=flat&logo=&logoColor=orange' alt='webpage'>
  </a> -->
  <a href="https://github.com/Zheng-Chong/CatV2TON/LICENCE" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/License-CC BY--NC--SA--4.0-lightgreen?style=flat&logo=Lisence' alt='License'>
  </a>
</div>


## Updates

- **`2025/02/24`**: We have released both 256 and 512 model weights, and provided [inference scripts](#inference). Check out our [HuggingFace repo](https://huggingface.co/zhengchong/CatV2TON) for the weights.
- **`2025/01/20`**: Our paper has been published on **[ArXiv](http://arxiv.org/abs/2501.11325v1)**.

## Overview

<div align="center">
  <img src="resource/img/overview.png" width="100%" height="100%"/>
</div>


- [CatV2TON: Taming Diffusion Transformers for Vision-Based Virtual Try-On with Temporal Concatenation](#catv2ton-taming-diffusion-transformers-for-vision-based-virtual-try-on-with-temporal-concatenation)
  - [Updates](#updates)
  - [Overview](#overview)
  - [Evaluation](#evaluation)
    - [Evaluation for Image Try-On](#evaluation-for-image-try-on)
    - [Evaluation for Video Try-On](#evaluation-for-video-try-on)
  - [Inference](#inference)
    - [Inference for Image Try-On](#inference-for-image-try-on)
    - [Inference for Video Try-On](#inference-for-video-try-on)


## Evaluation

### Evaluation for Image Try-On

We provide the evaluation script for VITONHD and DressCode datasets.
You can download our generated [VITONHD](https://drive.google.com/file/d/1ol2M6x918lDH6bawpsiJea6DNbTNf0Cc/view?usp=share_link) and [DressCode](https://drive.google.com/file/d/1kSjofynJM13ccxn-t69z4WuqO333ahXX/view?usp=share_link) results to evaluate the performance of our method.
Or you can infer your own results following the [Inference](#inference) section which may be slightly different due to the randomness of the inference process.

```bash
CUDA_VISIBLE_DEVICES=0 python eval_image_metrics.py \
--gt_folder YOUR_GT_FOLDER \
--pred_folder YOUR_PRED_FOLDER \
--batch_size 16 \
--num_workers 16 \
--paired
```


### Evaluation for Video Try-On  

We provide the evaluation script for ViViD-S-Test and VVT-Test datasets.
You can download our generated [ViViD-S-Test](https://drive.google.com/file/d/1tvcDe3Z4ES6VGtpS_OI1EG155xZgjfC5/view?usp=share_link) and [VVT-Test](https://drive.google.com/file/d/1Gh8YRBsdV3BeKEXR91CNI-UU2j1fMYnt/view?usp=share_link) results to evaluate the performance of our method.
Or you can infer your own results following the [Inference](#inference) section which may be slightly different due to the randomness of the inference process.

```bash
CUDA_VISIBLE_DEVICES=0 python eval_video_metrics.py \
--gt_folder YOUR_GT_FOLDER \
--pred_folder YOUR_PRED_FOLDER \
--num_workers 16 \
--paired
```

`YOUR_GT_FOLDER` is the path to the ground truth video folder which includes only `mp4` files.
`YOUR_PRED_FOLDER` is the path to the predicted video folder which includes only `mp4` files.


## Inference

### Inference for Image Try-On

We provide the inference script for VITONHD and DressCode datasets.\
The datasets can be downloaded from [VITONHD](https://github.com/shadow2496/VITON-HD) and [DressCode](https://github.com/aimagelab/dress-code).
You can run the following command to do inference with some edited parameters for your own settings.

```bash
CUDA_VISIBLE_DEVICES=0 python eval_image_try_on.py \
--dataset vitonhd | dresscode \
--data_root_path YOUR_DATASET_PATH \
--output_dir OUTPUT_DIR_TO_SAVE_RESULTS \
--dataloader_num_workers 8 \
--batch_size 8 \
--seed 42 \
--mixed_precision bf16 \
--allow_tf32 \
--repaint \
--eval_pair  
```

### Inference for Video Try-On

The Video Try-On Test datasets are provided: [ViViD-S-Test](https://drive.google.com/file/d/12QDkjn30P9EiIqZhtCFL4pEi7oZj2psQ/view?usp=share_link) and [VVT](https://drive.google.com/file/d/1mQaHP99c4CWLrVjPZEL_07OnW26z8xs2/view?usp=share_link).
You can run the following command to do inference with some edited parameters for your own settings.

```bash
CUDA_VISIBLE_DEVICES=0 python eval_video_try_on.py \
--dataset vivid | vvt \
--data_root_path YOUR_DATASET_PATH \
--output_dir OUTPUT_DIR_TO_SAVE_RESULTS \
--dataloader_num_workers 8 \
--batch_size 8 \
--seed 42 \
--mixed_precision bf16 \
--allow_tf32 \
--repaint \
--eval_pair  
```