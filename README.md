# CCFQA
CCFQA is a speech and text factuality evaluation benchmark that measures language models’ ability to answer short, fact-seeking questions and assess their cross-lingual and cross-modal consistency. It consists of speech and text in 8 languages, containing 1,800 n-way parallel sentences and a total of 14,000 speech samples.
- **Language**: Mandarin Chinese, English, French, Japanese, Korean, Russian, Spanish, Cantonese(HK)
- **ISO-3 Code**: cmn, eng, fra, jpn, kor, rus, spa, yue 

📄Paper：[https://arxiv.org/abs/2411.17666](https://arxiv.org/abs/2411.17666)

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
@misc{muhlgay2024generatingbenchmarksfactualityevaluation,
      title={Generating Benchmarks for Factuality Evaluation of Language Models}, 
      author={Dor Muhlgay and Ori Ram and Inbal Magar and Yoav Levine and Nir Ratner and Yonatan Belinkov and Omri Abend and Kevin Leyton-Brown and Amnon Shashua and Yoav Shoham},
      year={2024},
      eprint={2307.06908},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2307.06908}, 
}
```