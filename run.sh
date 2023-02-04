python denoiser.py \
          --save-prefix ./results/models/model_BBb \
          --save-interval 5 \
          --N-train 270 \
          --N-test 50 \
          -a ./dataset/Centriole/synthetic_noisy \
          -b ./dataset/Centriole/noisy \
          -spa ./dataset/Centriole/gradient_guidance \
          -gui ./dataset/Centriole/filtered_guidance \
          -smooth 0.05 \
          -recon 1.0 \
          -guide 0.5 \
          -loss_choice nmsg \
          -c 128 \
          -p 32 \
          -o ./results/image/image_BBb \
          --num-epochs 40 \
          -d 1 \
          --batch-size 2

python denoiser.py \
         -o ./results/image/image_BBb \
         -m ./results/pre_trained/model_BBb_epoch40.sav \
         -s 128 \
         -d 0 \
         --patch-padding 32 \
         --batch-size 2 \
         ./dataset/Centriole/noisy/BBb_1024.rec
