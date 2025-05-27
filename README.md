# Solution to the CVPR'2025 FGVC Challenge
This repository provides the solution(code and checkpoint) of the CVPR'2025 FathomNet-FGVC challenge.
[[FathomNet 2025 @ CVPR-FGVC]](https://www.kaggle.com/competitions/fathomnet-2025/overview)


## Installation
(1) PyTorch. See https://pytorch.org/ for instruction. For example,
```
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```
(2) Requirements
```python
pip install -r requirements.txt
```

## Data
The dataset can be downloaded from the official challenge page on [[FathomNet 2025 @ CVPR-FGVC Data]](https://www.kaggle.com/competitions/fathomnet-2025/data).

## Data Preprocessing
```python
python A0.data_preprocess.py --data_path Path/to/dataset_train.json
```
The script generates the following files:
| File | Description |
|------|-------------|
| dist_categories.csv | Taxonomic distance matrix between categories based on the biological tree |
| hierarchical_label.csv | Hierarchical labels (Phylum to Species) for each class |

## Quick Start
## Train
To train the model from scratch, run the following command:
```python
python B1.BuildModel.py --config ./config/experiment-final06.yaml
```
## Checkpoint
Download the pretrained checkpoint from the [Google Drive link](https://drive.google.com/file/d/14cig7fanfNMsC2WFFBvbeuJMwYnogy8g/view?usp=sharing).
## Test
To evaluate the model or run inference on new data, use:
```python
python C1.TestModel.py --config ./config/experiment-final06.yaml
```

