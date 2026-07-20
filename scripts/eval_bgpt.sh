# Image
#python evaluation/eval_bgpt.py \
#    --modality image \
#    --dataset  datasets/clic2024/bmp \
#    --model    pretrained/bgpt/weights-image.pth \
#    --n-samples 100 \
#    --device cuda:0 \
#    --output results/bgpt_image.csv

# Audio
python evaluation/eval_bgpt.py \
    --modality audio \
    --dataset  results/rac_vtck_p225/eval_samples.pkl \
    --model    pretrained/bgpt/weights-audio.pth \
    --n-samples 50 \
    --device cuda:3 \
    --audio-chunk-bytes 8160 \
    --no-decompress \
    