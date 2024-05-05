# MAPLE for diatom classification and Out of Distribution Detection


## Dataset

The dataset is arranged such that each class has a directory with the corresponding images placed in them. An example directory structure is shown below.

```bash
├── dataset
│   ├── train_data
│   │   ├── class1
│   │   ├── class2
...
│   │   ├── classN
│   ├── test_data
│   │   ├── class1
│   │   ├── class2
...
│   │   ├── classN

```
Each dataset is followed by a csv file containing the class name and the corresponding classification label. The CSV files for the D25 and D50 in-distribution (ID) and out-of-distribution (OOD) experiments are available in the `data/` folder. The ID files for D25 and D50 are `ude_dataset_25.csv` and `ude_dataset_50.csv` respectively. The OOD files for D25 and D50 are `ude_ood_dataset_25.csv` and `ude_ood_dataset_50.csv` respectively.

Before launching the training, please make sure that the dataset paths and the id paths (csv files) are included in the `config.py`. 


## Training

Python 3.10 was used for the training. The necessary python libraries and their versions to run the code are specified in `requirements.txt`. 
The hyperparameters and arguments needed for training the network are available in `config.py`. Depending on the dataset used, please make sure to change the respective hyperparameters. 
To launch the training, run 
```
python3 train.py
```
The code automatically splits the dataset into train and validation.

## Inference
To launch the inference, run
```
python3 mahalanobis_calculation.py
```
This calculates the Mahalanobis distance and the prediction probability for both the in distribution and out-of-distribution dataset, and computes the in distribution and out-of-distribution metrics.

The model weights for the experiments are available in the repository. The model weights for the 'deterministic' experiments on D25 and D50 are available in `deterministic_25` and `deterministic_50` respectively. The model weights for the MAPLE experiments on D25 and D50 are available in `maple_25` and `maple_50` respectively.

## Containerization
The docker images for the experiments are publicly available. 

<a href="https://hub.docker.com/repository/docker/vaishwarya96/maple_50/general" target="_blank">Docker image for MAPLE experiment on D50</a>

<a href="https://hub.docker.com/repository/docker/vaishwarya96/deterministic_50/general" target="_blank">Docker image for Deterministic experiment on D50</a>

<a href="https://hub.docker.com/repository/docker/vaishwarya96/maple_25/general" target="_blank">Docker image for MAPLE experiment on D25</a>

<a href="https://hub.docker.com/repository/docker/vaishwarya96/deterministic_25/general" target="_blank">Docker image for Deterministic experiment on D25</a>

