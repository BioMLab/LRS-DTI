# LRS-DTI

## 1. File Directory

```text
git_code/
├── README.md
├── requirements.txt
├── data/
│   ├── li's dataset/
│   └── luo's dataset/
├── LRS-DTI/
│   ├── Dataset.py
│   ├── Model.py
│   ├── Additionaltools.py
│   └── train.py
└── subgraph_extration/
    └── build_subgraphs.py
```

`data/` contains the datasets and fold files. `train.py` is the training script. `build_subgraphs.py` is the subgraph extraction script.

## 2. Subgraph Extraction

Download link: [subgraphs](https://pan.baidu.com/s/19aK4_9kxPDIqDOIF8j7xdw?pwd=fwtb).

Put the archive in `subgraph_extration/` and extract it there. The extracted directory should be `subgraph_extration/outputs/`.

```powershell
python .\subgraph_extration\build_subgraphs.py --dataset_name "luo's dataset" --setting normal --ratio 1:1
```

Arguments: `--dataset_name` (`li's dataset` or `luo's dataset`), `--setting` (`normal`, `drug_cold`, or `protein_cold`), and `--ratio` (`1:1`, `1:3`, or `1:5`). Cold-start settings only support `1:1`.

## 3. Model Training

Prepare `data/` and the matching subgraphs, then run:

```powershell
python .\LRS-DTI\train.py --dataset_name "luo's dataset" --setting normal --ratio 1:1
```

The three arguments are the same. Training runs five-fold evaluation. Required packages are listed in `requirements.txt`.
