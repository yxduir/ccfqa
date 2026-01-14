# [AAAI 2026] CCFQA: A Benchmark for Cross-Lingual and Cross-Modal Speech and Text Factuality Evaluation
CCFQA is a speech and text factuality evaluation benchmark that measures language models’ ability to answer short, fact-seeking questions and assess their cross-lingual and cross-modal consistency. It consists of speech and text in 8 languages, containing 1,800 n-way parallel sentences and a total of 14,400 speech samples.
- **Language**: Mandarin Chinese, English, French, Japanese, Korean, Russian, Spanish, Cantonese(HK)
- **ISO-3 Code**: cmn, eng, fra, jpn, kor, rus, spa, yue
- **Data Size**: 14,400 sample 
- **Data Split**: Test
- **Data Source**: Native speakers  (6 males and 6 females)
- **Domain**: Factuality Evaluation
- **Task**: Spoken Question Answering(SQA)
- **License**: CC BY-NC-SA-4.0


📄Paper：[https://arxiv.org/abs/2508.07295](https://arxiv.org/abs/2508.07295)

## How to use
```python
from datasets import load_dataset
ccfqa = load_dataset("yxdu/ccfqa")
print(ccfqa)
```


## Installation
```
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/yxduir/ccfqa
cd ccfqa

uv venv --python 3.10
source .venv/bin/activate

sudo apt update
sudo apt install ffmpeg
uv pip install -r requirements.txt
```

## Download Model 
```
cd models/

#Total 16G of storage space
hf download Qwen/Qwen2-Audio-7B-Instruct --local-dir Qwen2-Audio-7B-Instruct

#Total 52G of storage space for eval model
#Access to the Gemma-3 models is required before downloading.
hf download google/gemma-3-27b-it --local-dir gemma-3-27b-it
cd ..
```

## Download Demo Data
```
cd data
hf download yxdu/ccfqa_test --repo-type dataset --local-dir ./ccfqa_test
tar -zxvf "./ccfqa_test/audio.tar.gz" -C "./ccfqa_test/"
cd ..
```


## VLLM Inference Demo

```
cd eval
bash run_qwen2audio.sh
cd ..
```

## Eval

```
cd output
python vllm_eval.py
```



# 🖊Citation
```
@misc{du2025ccfqabenchmarkcrosslingualcrossmodal,
      title={{CCFQA}: A Benchmark for Cross-Lingual and Cross-Modal Speech and Text Factuality Evaluation}, 
      author={Yexing Du and Kaiyuan Liu and Youcheng Pan and Zheng Chu and Bo Yang and Xiaocheng Feng and Ming Liu and Yang Xiang},
      year={2025},
      eprint={2508.07295},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.07295}, 
}
```
