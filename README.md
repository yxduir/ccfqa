# CCFQA
CCFQA is a speech and text factuality evaluation benchmark that measures language models’ ability to answer short, fact-seeking questions and assess their cross-lingual and cross-modal consistency. It consists of speech and text in 8 languages, containing 1,800 n-way parallel sentences and a total of 14,400 speech samples.
- **Language**: Mandarin Chinese, English, French, Japanese, Korean, Russian, Spanish, Cantonese(HK)
- **ISO-3 Code**: cmn, eng, fra, jpn, kor, rus, spa, yue 

📄Paper：[https://arxiv.org/abs/2508.07295](https://arxiv.org/abs/2508.07295)

## How to use



```python
from datasets import load_dataset
ccfqa = load_dataset("yxdu/ccfqa")
print(ccfqa)
```

## ⚖️ Evals

please visit [github page](https://github.com/yxduir/ccfqa).


## License

All datasets are licensed under the [Creative Commons Attribution-NonCommercial license (CC-BY-NC)](https://creativecommons.org/licenses/), which allows use, sharing, and adaptation for **non-commercial** purposes only, with proper attribution.

# 🖊Citation

```
@misc{du2025ccfqabenchmarkcrosslingualcrossmodal,
      title={CCFQA: A Benchmark for Cross-Lingual and Cross-Modal Speech and Text Factuality Evaluation}, 
      author={Yexing Du and Kaiyuan Liu and Youcheng Pan and Zheng Chu and Bo Yang and Xiaocheng Feng and Yang Xiang and Ming Liu},
      year={2025},
      eprint={2508.07295},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.07295}, 
}
```
