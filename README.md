# Deep Learning for Medical Imaging and RNA–Protein Binding

Deep learning project exploring neural architectures across two biological domains: **blood cell image classification** and **RNA–protein binding affinity prediction**.

The project combines convolutional neural networks for image classification with CNN, BiLSTM, and attention-based architectures for RNA sequence modeling. It also explores automated hyperparameter optimization, sequence masking, and architecture design for biological data.

## Project Overview

The project is divided into two main tasks:

### BloodMNIST Image Classification

The first task addresses multi-class blood cell image classification using the **BloodMNIST** dataset.

Two convolutional architectures were evaluated:

- **Baseline CNN** using stacked convolutional layers followed by a fully connected classifier.
- **Improved CNN** incorporating MaxPooling layers to reduce spatial dimensionality and computational cost.

The experiments also investigated the effect of explicitly applying Softmax before the cross-entropy loss. Since PyTorch's `CrossEntropyLoss` already applies LogSoftmax internally, using the raw logits is the standard formulation.

Adding MaxPooling improved generalization while substantially reducing the dimensionality of the representation passed to the fully connected layers.

### RNA–Protein Binding Prediction

The second task addresses the prediction of RNA–protein binding affinity for the **RBFOX1** protein using RNAcompete data.

Three architectures were implemented and compared:

- **1D CNN** — learns local sequence motifs and uses masked global max pooling.
- **Bidirectional LSTM (BiLSTM)** — captures dependencies across the complete RNA sequence.
- **BiLSTM + Attention** — extends the recurrent architecture with self-attention and learned attention pooling.

RNA sequences are one-hot encoded and padded to a fixed maximum length. Sequence masks prevent padded positions from influencing the models, while missing binding measurements are handled through masked loss and evaluation functions.

Hyperparameters were optimized automatically using **Optuna**, and model performance was evaluated using **Spearman rank correlation**.

## Models

| Task | Model | Main Components |
| --- | --- | --- |
| BloodMNIST | Baseline CNN | Convolutional layers, ReLU, fully connected classifier |
| BloodMNIST | Improved CNN | Convolutional layers, MaxPooling, fully connected classifier |
| RNA Binding | 1D CNN | 1D convolutions, masked global max pooling |
| RNA Binding | BiLSTM | Bidirectional LSTM, masked sequence pooling |
| RNA Binding | BiLSTM + Attention | Bidirectional LSTM, self-attention, attention pooling |

## Results

### BloodMNIST

Introducing MaxPooling improved test accuracy while considerably reducing the size of the representation entering the fully connected classifier.

| Architecture | Output | Test Accuracy |
| --- | --- | ---: |
| Baseline CNN | Logits | 92.98% |
| Baseline CNN | Softmax | 93.16% |
| CNN + MaxPooling | Logits | 94.68% |
| CNN + MaxPooling | Softmax | **94.74%** |

The pooled architecture reduced the feature representation entering the first fully connected layer from **100,352 to 1,152 features**, reducing the corresponding parameter count from approximately **25.7M to 0.3M**.

Training time on a Kaggle P100 GPU also decreased from approximately **15 minutes to 10 minutes**.

### RNA–Protein Binding

| Model | Spearman Correlation |
| --- | ---: |
| 1D CNN | 0.566 |
| BiLSTM | 0.662 |
| BiLSTM + Attention | **0.666** |

The BiLSTM substantially outperformed the 1D CNN, suggesting that modeling the broader sequence context was beneficial compared with relying primarily on local motif detection.

Adding attention produced only a small improvement over the BiLSTM (**0.662 → 0.666**). Given the relatively short RNA sequences, with a maximum length of 41 nucleotides, the additional attention mechanism provided limited benefit.

Training curves and additional experimental results are available in [`results/`](results/).

## Hyperparameter Optimization

Optuna was used to optimize the RNA models.

The selected configurations included:

| Parameter | 1D CNN | BiLSTM | BiLSTM + Attention |
| --- | ---: | ---: | ---: |
| Kernel Size | 6 | — | — |
| Filters | 126 | — | — |
| Pooling (k) | 2 | — | — |
| Layers | — | 2 | 2 |
| Hidden Dimension | 42 | 83 | 97 |
| Dropout | 0.12 | 0.28 | 0.28 |
| Learning Rate | 1.2 × 10⁻³ | 1.3 × 10⁻³ | 9.7 × 10⁻⁴ |

Interestingly, Optuna selected a kernel size of **6** for the CNN, matching the length of the RBFOX1 consensus motif considered in the project.

## Repository Structure

```text
deep-learning-medical-imaging-rna/
├── src/
│   ├── bloodmnist/
│   │   ├── cnn_baseline.py
│   │   └── cnn_improved.py
│   │
│   └── rna_binding/
│       ├── cnn.py
│       ├── bilstm.py
│       ├── bilstm_attention.py
│       ├── config.py
│       └── utils.py
│
├── results/
│   ├── bloodmnist/
│   └── rna_binding/
│
├── report/
│   └── project_report.pdf
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

Clone the repository:

```bash
git clone https://github.com/catarinassgoncalves/deep-learning-medical-imaging-rna.git
cd deep-learning-medical-imaging-rna
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Data

### BloodMNIST

BloodMNIST is accessed through the [`medmnist`](https://medmnist.com/) Python package.

### RNAcompete

The RNA-binding experiments use RNAcompete data. The raw RNAcompete data is not included in this repository and must be obtained separately.

The scripts expect the following local structure:

```text
data/
├── metadata.xlsx
└── norm_data.txt
```

Processed RNA datasets are cached locally in the same directory and are excluded from version control.

## Technologies

- **Python**
- **PyTorch**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **MedMNIST**
- **Optuna**

## Project Context

This project was developed as part of the **Deep Learning** course at **Instituto Superior Técnico (University of Lisbon)**.

The project was completed collaboratively by **Afonso Gouveia, Catarina Gonçalves, and André Amaro**.

Some configuration and RNAcompete data-loading utilities were provided as course infrastructure and are included to support reproducibility.

For the complete methodology, experimental analysis, and discussion, see the [project report](report/project_report.pdf).