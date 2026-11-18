# PromptBERT: Word Dropout Extension (fork)

This is a fork of [princeton-nlp/PromptBERT](<link to the original repo>) with an added
augmentation experiment. The original README follows below the divider.

Original Paper
[PromptBERT: Improving BERT Sentence Embeddings with Prompts](https://aclanthology.org/2022.emnlp-main.603/) (Jiang et al., EMNLP 2022)

## What I changed
- Added word deletion on one view of each training pair (`--word_dropout_prob`, default 0.0 = original behavior)
- Edits are in `train.py` (flag + helper, marked `SHANECHANGE`) and `run.sh` (flag passthrough, `--overwrite_cache`)
- Trained on a 100k-sentence subset of the 1M training file, 1 epoch, BERT-base

## Results (100k subset, Avg. STS, mean ± std over seeds)
| p | Avg. | Δ vs baseline | Seeds |
|---|---|---|---|
| 0.0 (baseline) | 77.87 ± 0.18 | n/a | 5 |
| 0.05 | 78.25 ± 0.17 | +0.37 (paired t-test p = 0.013, uncorrected) | 5 |
| 0.1 | 78.23 ± 0.26 | +0.29 | 3 |
| 0.2 | 77.32 ± 0.18 | -0.55 | 3 |
| 0.3 | 76.15 ± 0.31 | -1.72 | 3 |

All Δ values are relative to the 5-seed baseline mean (77.87). Rows with 3 seeds ran seeds 0 to 2 only.

## How to reproduce
```bash
P=0.05 ./run.sh unsup-bert 0 0   # p=0.05, seed 0, GPU 0
```
Baseline: `P=0.0`. 
Training data: the first 100,000 lines of `wiki1m_for_simcse.txt`
```bash
head -n 100000 wiki1m_for_simcse.txt > wiki100k_for_simcse.txt
```
## Limitations
- 100k sentences only (full-data run pending or single-seed), one template pair, BERT-base
- p chosen by sweeping on the same test numbers; several p values tried
- Per-task gains not reliable; only the Avg. column is claimed

## Other experiments
- Mean-pooling the last 3 layers: 76.26 (single seed), worse than baseline


## Citation

If you use this work, please cite the original paper:

```bibtex
@inproceedings{jiang-etal-2022-promptbert,
    title = "{P}rompt{BERT}: Improving {BERT} Sentence Embeddings with Prompts",
    author = "Jiang, Ting  and
      Jiao, Jian  and
      Huang, Shaohan  and
      Zhang, Zihan  and
      Wang, Deqing  and
      Zhuang, Fuzhen  and
      Wei, Furu  and
      Huang, Haizhen  and
      Deng, Denvy  and
      Zhang, Qi",
    editor = "Goldberg, Yoav  and
      Kozareva, Zornitsa  and
      Zhang, Yue",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.603/",
    doi = "10.18653/v1/2022.emnlp-main.603",
    pages = "8826--8837",
    abstract = "We propose PromptBERT, a novel contrastive learning method for learning better sentence representation. We firstly analysis the drawback of current sentence embedding from original BERT and find that it is mainly due to the static token embedding bias and ineffective BERT layers. Then we propose the first prompt-based sentence embeddings method and discuss two prompt representing methods and three prompt searching methods to make BERT achieve better sentence embeddings .Moreover, we propose a novel unsupervised training objective by the technology of template denoising, which substantially shortens the performance gap between the supervised and unsupervised settings. Extensive experiments show the effectiveness of our method. Compared to SimCSE, PromptBert achieves 2.29 and 2.58 points of improvement based on BERT and RoBERTa in the unsupervised setting."
}
```



# Original PromptBERT paper's readme:
# PromptBERT: Prompt makes BERT Better at Sentence Embeddings

Update: We have extended our prompt-based method to LLMs in [scaling_sentemb](https://github.com/kongds/scaling_sentemb). [2023/08/01]

## Overview
We propose PromptBERT, a novel contrastive learning method for learning better sentence representation. We firstly analyze the drawback of current sentence embedding from original BERT and find that it is mainly due to the static token embedding bias and ineffective BERT layers. Then we propose the first prompt-based sentence embeddings method and discuss two prompt representing methods and three prompt searching methods to make BERT achieve better sentence embeddings. Moreover, we propose a novel unsupervised training objective by the technology of template denoising, which substantially shortens the performance gap between the supervised and unsupervised settings. Extensive experiments show the effectiveness of our method. Compared to SimCSE, PromptBert achieves 2.29 and 2.58 points of improvement based on BERT and RoBERTa in the unsupervised setting.
    
## Results on STS Tasks

| Model                                                                                                                    | STS12      | STS13      | STS14      | STS15      | STS16      | STSb       | SICK-R     | Avg.       |
|--------------------------------------------------------------------------------------------------------------------------|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| [royokong/unsup-PromptBERT](https://huggingface.co/royokong/unsup-PromptBERT)   | 71.56±0.18 | 84.58±0.22 | 76.98±0.26 | 84.47±0.24 | 80.60±0.21 | 81.60±0.22 | 69.87±0.40 | 78.54±0.15 |
| [royokong/unsup-PromptRoBERTa](https://huggingface.co/royokong/unsup-PromptRoBERTa) | 73.94±0.90 | 84.74±0.36 | 77.28±0.41 | 84.99±0.25 | 81.74±0.29 | 81.88±0.37 | 69.50±0.57 | 79.15±0.25 |
| [royokong/sup-PromptBERT](https://huggingface.co/royokong/sup-PromptBERT)      | 75.48      | 85.59      | 80.57      | 85.99      | 81.08      | 84.56      | 80.52      | 81.97      |
| [royokong/sup-PromptRoBERTa](https://huggingface.co/royokong/sup-PromptRoBERTa)   | 76.75      | 85.93      | 82.28      | 86.69      | 82.80      | 86.14      | 80.04      | 82.95      |
    
To evaluate the above models, please run the following script, 
```sh
bash eval_only.sh [unsup-bert|unsup-roberta|sup-bert|sup-roberta]
```
## Setup
Install Dependencies

``` sh
pip install -r requirements.txt
```

Download Data

``` sh
cd SentEval/data/downstream/
bash download_dataset.sh
cd -
cd ./data
bash download_wiki.sh
bash download_nli.sh
cd -
```

## Prompt with unfine-tuned BERT

bert-base-uncased with prompt

``` sh
./run.sh bert-prompt
```

bert-base-uncased with optiprompt

``` sh
./run.sh bert-optiprompt
```

## Train PromptBERT and PromptRoBERTa
unsupervised

``` sh
SEED=0
./run.sh unsup-roberta $SEED
```

``` sh
SEED=0
./run.sh unsup-bert $SEED
```
supervised

``` sh
./run.sh sup-roberta 
```

``` sh
./run.sh sup-bert
```

 
## Static token embedding with removing embedding biases
robert-base, bert-base-cased and robert-base-uncased
```sh
./run.sh roberta-base-static-embedding-remove-baises
./run.sh bert-base-cased-static-embedding-remove-baises
./run.sh bert-base-uncased-static-embedding-remove-baises
```
## Calculation of anisotropy
To calculate the sentence anisotropy

```sh
EXP=bert-base-uncased | bert-base-uncased-static-embedding | bert-base-uncased-static-embedding-remove-baises 
./run.sh calc-anisotropy-${EXP}
```

## Acknowledgement
Our Code is based on SimCSE
