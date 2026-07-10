python evaluation/eval_rac_bgpt.py \
    --database results/rac_img_db \
    --model pretrained/bgpt/weights-image.pth \
    --m 16 \
    --device cuda:0 \
    --no-decompress

# Audio:
# python evaluation/eval_rac_bgpt.py \
#     --database results/rac_audio_db \
#     --model pretrained/bgpt/weights-audio.pth \
#     --m 16 \
#     --device cuda:0 \
#     --no-decompress
