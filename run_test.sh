python denoiser.py \
         -o ./results/image/image_BBb \
         -m ./results/pre_trained/model_BBb_epoch40.sav \
         -s 128 \
         -d 0 \
         --patch-padding 32 \
         --batch-size 2 \
         ./dataset/Centriole/noisy/BBb_1024.rec
