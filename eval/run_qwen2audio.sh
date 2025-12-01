export NCCL_P2P_DISABLE=1
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# VLLM_USE_V1=0 \
python ./audio_language_ccfqa.py \
    -m qwen2_audio \
    --task xsqa \
    --input-file ../data/test/test_x.jsonl \
    --output-dir ../output/ \
    --num-audios 1 \
    --seed 443 \
    --batch 1000 \