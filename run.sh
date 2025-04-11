git clone https://github.com/shivank21/CatV2TON.git
cd CatV2TON

conda create -n catv2ton python=3.11

conda activate catv2ton

pip install -r requirements.txt

gdown 12QDkjn30P9EiIqZhtCFL4pEi7oZj2psQ

tar -xvzf /content/CatV2TON/ViViD-CLIPED-512x384-Test.tar.gz

Data_Path="./ViViD-CLIPED-512x384-Test"
Output="./results"

CUDA_VISIBLE_DEVICES=0 python eval_video_try_on.py \
--dataset vivid \
--data_root_path "$Data_Path" \
--output_dir "$Output" \
--dataloader_num_workers 1 \
--batch_size 1 \
--seed 42 \
--mixed_precision bf16 \
--allow_tf32 \
--repaint \
--eval_pair

