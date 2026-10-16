# Accurate Fairness Criterion
Code for Accurate Fairness: Improving Individual Fairness without Trading Accuracy (AAAI-23)

**Xuran Li, Peng Wu\*, Jing Su** · State Key Laboratory of Computer Science, Institute of Software, Chinese Academy of Sciences · University of Chinese Academy of Sciences

📄 **AAAI-23** · [arXiv:2205.08704](https://arxiv.org/abs/2205.08704) · *\*Corresponding author: Peng Wu*

> **TL;DR** — Accuracy and individual fairness are both crucial for trustworthy machine learning, but these two aspects are often incompatible with each other so that enhancing one aspect may sacrifice the other inevitably with side effects of true bias or false fairness. We propose in this paper a new fairness criterion, accurate fairness, to align individual fairness with accuracy. Informally, it requires the treatments of an individual and the individual’s similar counterparts to conform to a uniform target, i.e., the ground truth of the individual.

---

## Table of contents

1. [Motivation](#1-motivation)
2. [Accurate Fairness Criterion](#2-Accurate-Fairness-Criterions)
3. [Siamese Fairness Imerprovment](#3-siamese-Fairness-Imerprovment)
4. [Experiments Results](#4-experiments)
5. [Code](#5-code)
6. [Citation and Acknowledgments](#6-citation-and-acknowledgments)

---

## 1. Motivation

Accuracy and individual fairness are usually evaluated separately, and neither can tell on its own whether a prediction is *both* accurate and fair. For example, take one loan application and two similar applicants, same income, credit history and job, differing only in gender, whose ideal application *should* be both approved:

<p align="center"><img src="figures/accurate_fairness_overview.png" width="900" alt="Accurate fairness explained with two similar loan applicants: the four cases Accurate Fair, Accurate Bias, Inaccurate Fair and Inaccurate Bias"></p>

Only the top-left case, **Accurate Fair**, is what we want. 

The other three are the ways a model can *look* good on one metric while failing the other:

- **Accurate Bias** — the individual is predicted correctly, but the counterpart who differs only in the sensitive attribute gets a different answer. An accuracy-only evaluation never notices.
- **Inaccurate Fair** — the similar individuals gets the same results, so individual-fairness metrics are satisfied, but the output answers are wrong. A fairness-only evaluation never notices.
- **Inaccurate Bias** — both wrong *and* biased.

Existing fairness or accuracy- improvement methods optimise fairness or accuracy *separately*, causing a trade-off: Accurate Bias or Inaccurate Fair. Accurate fairness is designed to rule out both.

## 2. Accurate Fairness Criterion

For an input `(x, a)` with non-sensitive attributes `x`, sensitive attributes `a` and ground truth `y`, its **similar counterparts** is every version of the same person with a different sensitive value:

```
I(x, a) = { (x, a') | a' ∈ A }
```

**Definition.** A classifier `f` is **accurately fair** to `(x, a)` if, for every `(x, a') ∈ I(x, a)`,

```
D( y, f(x, a') )  ≤  K · d( (x, a), (x, a') )
```

where `D` and `d` are distance metrics and `K ≥ 0` is a Lipschitz constant. One inequality captures both requirements:

| Taking `a' = a` | Taking `a' ≠ a` |
|---|---|
| The right-hand side is 0, so `f(x, a) = y`: **accuracy**. | Predictions for similar individuals must stay within `K·d` of the ground truth, and therefore of each other: **individual fairness** (the Lipschitz condition of Dwork et al.). |

<p align="center"><img src="figures/accuracy_vs_fairness.png" width="520" alt="Accurate fairness sits between accuracy and individual fairness"></p>

Two consequences proved in the paper:

- **Theorem 1.** Accurate fairness implies the usual individual-fairness Lipschitz condition over the similar counterparts — it is a refinement of individual fairness built on top of accuracy.
- **Theorems 2–3.** Accurate fairness implies **statistical parity** and **confusion-matrix-based group fairness** over the union of similar group populations. Improving individual fairness this way improves group fairness as well.

## 3. Siamese fairness Imerprovment

To train a model under the accurate-fairness constraints we solve

```
min_f   Σ_(xᵢ,aᵢ)∈V  Σ_(xᵢ,aᵢⱼ)∈I(xᵢ,aᵢ)  L( yᵢ, f(xᵢ, aᵢⱼ) )
s.t.    D( yᵢ, f(xᵢ, aᵢⱼ) ) ≤ K · d( (xᵢ,aᵢ), (xᵢ,aᵢⱼ) )     for all i, j
```

which, by the KKT conditions, becomes a max–min problem with one Lagrange multiplier `λᵢⱼ` per similar individual. A **Siamese network** is the natural way to optimise it: it feeds every member of a similar sub-population through `m` copies of the model with **shared parameters θ**, so the whole sub-population is treated uniformly in every back-propagation step.

<p align="center"><img src="figures/siamese_fairness.png" width="640" alt="Siamese fairness in-processing architecture"></p>

1. **Fair augmentation** — for each training input `vᵢ = (xᵢ, aᵢ)`, generate its similar counterparts `(xᵢ, aᵢ₁), …, (xᵢ, aᵢₘ)`.
2. **Parallel forward pass** — each counterpart goes through its own copy of the model, all sharing `θ`.
3. **Accurate-fairness loss** — the task loss `L` plus the Lagrangian penalty for the constraint, summed over the sub-population.
4. **Update** — gradient ascent on `λ`, gradient descent on `θ`, until convergence (Algorithm 1 in the paper).

To the best of our knowledge this is the first time a Siamese approach has been adapted for bias mitigation. It works with any differentiable classifier; the repo includes fully-connected networks (FCNN), logistic regression (LR) and SVM.

## 4. Experiments

### Fairness confusion matrix

Because accurate fairness has two axes — *is the prediction correct?* and *is it consistent across the sub-population?* — every prediction falls into one of the four cells seen in the motivation figure. The paper names them True Fair / True Biased / False Fair / False Biased:

<p align="center"><img src="figures/fairness_confusion_matrix.png" width="480" alt="Fairness confusion matrix"></p>

| Figure name | Paper name | Rate | Meaning |
|---|---|---|---|
| Accurate Fair | True Fair | TFR | Correct, and all similar counterparts agree — the goal. |
| Accurate Bias | True Biased | TBR | Correct for the individual, but some counterpart gets a different answer. |
| Inaccurate Fair | False Fair | FFR | Wrong, but consistently wrong across the sub-population. |
| Inaccurate Bias | False Biased | FBR | Wrong *and* inconsistent. |

From these rates we define three metrics that summarise the accuracy–fairness trade-off in one number each:

```
Fair-Precision  F-P  = TFR / (TFR + TBR)     # how many accurate predictions are also fair
Fair-Recall     F-R  = TFR / (TFR + FFR)     # how many fair predictions are also accurate
Fair-F1         F-F1 = 2 · F-P · F-R / (F-P + F-R)
```

### Datasets

| Dataset | Size | Models | Sensitive attributes |
|---|---|---|---|
| Adult (Census Income) | 45,222 | FCNN, LR, SVM | gender, age, race |
| German Credit | 1,000 | FCNN, LR, SVM | gender, age |
| ProPublica Recidivism (COMPAS) | 6,172 | FCNN, LR, SVM | gender, age, race |
| Ctrip | 68,191 | FCNN (3 and 5 layers) | 6 customer consumption habits |

### Benchmark results

<p align="center"><img src="figures/fcm_results.png" width="720" alt="Fairness confusion matrix performance of each method on FCNN classifiers"></p>

Share of each prediction type for a fully-connected classifier, averaged over the three benchmark datasets and ten runs. **BL** = baseline; **iFair**, **LFR**, **SSI** (SenSeI), **SSR** (SenSR) = state-of-the-art bias mitigation; **SF** / **SF_3** = our Siamese fairness (SF_3 augments with only the min/max sensitive values to save computation).

Compared with the state-of-the-art individual bias mitigation techniques, Siamese fairness achieves on average:

| Metric | Improvement |
|---|---|
| Individual fairness (fairness through awareness) | **+1.02 % – 8.78 %** |
| Accuracy | **+8.38 % – 13.69 %** |
| True fair rate | **+10.09 % – 20.57 %** |
| Fair-F1 score | **+5.43 % – 10.01 %** |

The Fairea benchmark places only the SF and SF_3 models in the *win-win* region; every other method improves fairness at the cost of accuracy.

### Real-world case: service discrimination on Ctrip

Should customers who pay the same price for the same room receive the same room service, regardless of their consumption habits? Treating six consumption-habit attributes as sensitive, the baseline FCNNs do discriminate: only **34.68 %** of customers are served both accurately and fairly (TFR). With Siamese fairness the true fair rate rises to **62.85 %** — close to its ceiling, the models' accuracy of 65.77 % — and **93.00 %** of customers are served fairly irrespective of their habits.

| Metric | BL (3-layer) | SF_3 (3-layer) | BL (5-layer) | SF_3 (5-layer) |
|---|---|---|---|---|
| Accuracy | 0.664 | 0.660 | 0.666 | 0.655 |
| Fairness through awareness | 0.524 | **0.902** | 0.352 | **0.958** |
| True fair rate | 0.412 | **0.620** | 0.281 | **0.637** |
| True biased rate | 0.251 | **0.040** | 0.385 | **0.018** |
| Fair-F1 | 0.689 | **0.794** | 0.516 | **0.790** |

## 5. Code

### Repository layout

```
AccurateFairness/
├── prepare_<dataset>_data.py          # clean + fair-augment a dataset  (adult | credit | compas | ctrip)
├── train_<dataset>_siamese_fair*.py   # Siamese fairness in-processing
├── get_<dataset>_result.py            # evaluate baseline vs. SF models, print tables, draw charts
├── <dataset>_data/                    # train / test splits
├── <dataset>_model/                   # pre-trained baseline (BL_*) and Siamese fairness (FS_*) models
├── utils/
│   ├── utils_data_augmentation.py     # builds similar sub-populations I(x, a)
│   ├── utils_siamese_fair.py          # Siamese architecture + accurate-fairness loss
│   └── util_result.py                 # metrics: ACC, SPD/EOD/AVOD, CON, FTA, fairness confusion matrix
└── pic/                               # charts written by get_<dataset>_result.py
```

Pre-trained models are included, so the evaluation can be reproduced without training.

### Requirements

- Python 3.8
- `tensorflow==2.4.1`
- `keras==2.4.3`

### Reproduce the evaluation (no training needed)

```bash
cd AccurateFairness
python prepare_ctrip_data.py      # builds the test set and its similar counterparts
python get_ctrip_result.py        # evaluates the pre-trained BL and SF models
```

This prints accuracy, individual fairness, group fairness and accurate fairness for every model, and writes to `pic/`:

- a statistics table in the format of **Table 3** of the paper,
- `<dataset>_<method>.pdf` — fairness-confusion-matrix pie charts (**Figure 2**), where `method` is `bl` (baseline) or `sf` (Siamese fairness),
- `<dataset>_Fairea.pdf` — the Fairea trade-off evaluation (**Figure 3**).

Swap `ctrip` for `adult`, `credit` or `compas` to run the other datasets.

### Train a Siamese fairness model

```bash
cd AccurateFairness
python prepare_ctrip_data.py             # fair-augments the training set
python train_ctrip_siamese_fair.py       # Siamese fairness in-processing
```

The other datasets use `train_adult_siamese_fairness.py`, `train_credit_siamese_fairness.py` and `train_compas_siamese_fairness.py`.

## 6. Citation and acknowledgments

```bibtex
@inproceedings{li2023accurate,
  title     = {Accurate Fairness: Improving Individual Fairness without Trading Accuracy},
  author    = {Li, Xuran and Wu, Peng and Su, Jing},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2023}
}
```

This work is partially supported by the National Key R&D Program of China (Grants 2022YFA1005100, 2022YFA1005101, 2022YFA1005104), the CAS-INRIA Joint Research Program (Grant GJHZ1844), the CAS Project for Young Scientists in Basic Research (Grant YSBR-040), the Beijing Academy of Artificial Intelligence (BAAI), and the Joint Laboratory on Cyberspace Security, China Southern Power Grid.

Questions or issues: open a GitHub issue or contact `lixr@ios.ac.cn`.




# Accurate Fairness Criterion
Code for Accurate Fairness: Improving Individual Fairness without Trading Accuracy (AAAI-23)
**************************************************************************************
Package Requirements:
- tensorflow 2.4.1
- keras 2.4.3 
**************************************************************************************
Siamese fairness in-processing for improving the accurate fairness of a machine learning model.

For example: to improve the accurate fairness on the Ctrip dataset,
- First run prepare_ctrip_data.py to generate the augmentated training dataset,
- Then run train_ctrip_siamese_fairness.py to improve the accurate fairness.
**************************************************************************************
How to check whether the predications of a machine learning model are accurately fair?

For example: to check the accurate fairness on the Ctrip dataset,
- First run prepare_ctrip_data.py to generate the test dataset,
- Then run get_ctrip_result.py to print out the accuracy, individual fairness, group fairness and accurate fairness measurements of the baseline model and the corresponding Siamese fairness model, in the following formats
  - a table, similar to Table 3 in our main paper, for statistical results of each model with various sensitive attributes
  - a chart X_Y.pdf in .\pic, similar to Figure 2 in our main paper, for fairness confusion matrix performances of dataset X under method Y (e.g., bl=baseline, sf=Siamese fairness) 
  - a chart X_Fairea.pdf in .\pic, similar to Figure 3 in our main paper, for Fairea evaluation of dataset X
