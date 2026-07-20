#python evaluation/eval_rac_bgpt.py \
#    --database results/rac_img_db \
#    --model pretrained/bgpt/weights-image.pth \
#    --m 16 \
#    --device cuda:0 \
#    --no-decompress

# Audio:
 python evaluation/eval_rac_bgpt.py \
     --database results/rac_vtck_p225/ \
     --model pretrained/bgpt/weights-audio.pth \
     --m 4 \
     --device cuda:3 \
     --n-samples 50 \
     --cascade \
     --cascade-max-cond 2 \
     --cascade-retriever \
     --calibrate \
     --calib-samples 10 \
     --no-decompress
