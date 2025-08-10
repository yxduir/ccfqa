
# CCFQA
CCFQA is a speech and text factuality evaluation benchmark that measures language models’ ability to answer short, fact-seeking questions and assess their cross-lingual and cross-modal consistency. It consists of speech and text in 8 languages, containing 1,800 n-way parallel sentences and a total of 14,000 speech samples.
- **Language**: Mandarin Chinese, English, French, Japanese, Korean, Russian, Spanish, Cantonese(HK)
- **ISO-3 Code**: cmn, eng, fra, jpn, kor, rus, spa, yue 

<!-- 📄Paper：[https://arxiv.org/abs/2503.07010](https://arxiv.org/abs/2503.07010) -->

## How to use



```python
from datasets import load_dataset

ccfqa = load_dataset("yxdu/ccfqa")
print(ccfqa)
```

## License

All datasets are licensed under the [Creative Commons Attribution-NonCommercial license (CC-BY-NC)](https://creativecommons.org/licenses/), which allows use, sharing, and adaptation for **non-commercial** purposes only, with proper attribution.

<!-- # 🖊Citation

```
@misc{liu2025projectevalbenchmarkprogrammingagents,
      title={ProjectEval: A Benchmark for Programming Agents Automated Evaluation on Project-Level Code Generation}, 
      author={Kaiyuan Liu and Youcheng Pan and Yang Xiang and Daojing He and Jing Li and Yexing Du and Tianrun Gao},
      year={2025},
      eprint={2503.07010},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2503.07010}, 
}
``` -->


