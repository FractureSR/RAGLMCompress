#python evaluation/eval_rac_bgpt.py \
#    --database results/rac_img_db \
#    --model pretrained/bgpt/weights-image.pth \
#    --m 16 \
#    --device cuda:0 \
#    --no-decompress

# Audio:
 python evaluation/eval_rac_bgpt.py \
     --database results/rac_peoples_speech \
     --model pretrained/bgpt/weights-audio.pth \
     --m 4 \
     --device cuda:2 \
     --n-samples 100 \
     --cascade \
     --cascade-max-cond 2 \
     --cascade-retriever \
     --calibrate \
     --calib-samples 20 \
     --no-decompress
