# CUDA_VISIBLE_DEVICES=0 CUDA_CACHE_PATH='~/.cudacache'    
python train_hx.py \
        --data config/wsodd.yaml \
        --epochs 2 \
        --batch-size 2 \
        --img-size 512 \
        --device 0 