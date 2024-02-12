# SeeM

This is the official source code of the SeeM: A Shared Latent Variable Model for Unsupervised Multi-View Anomaly Detection [paper](), published at the 2024 PAKDD conference.

## Install Python Environment

Make sure you have `miniconda` installed.

Create an virtual enviroment and install all packages from the requirements.txt file.

```bash
conda create --name mytorch python=3.8
conda activate mytorch
pip install -r requirements.txt
```

## Preprocess Datasets

- The `heart` dataset is taken from [UCI](https://archive.ics.uci.edu/ml/datasets/heart+disease).
- Run recreate the `heart` dataset, just run `heart.py` in directory `src/datamodules`:
```sh
python heart.py
```

## Run Model
Navigate to the `src` directory and run the shell script file `scripts/benchmark.sh`

## Cite
Update citation once the proceddings published.