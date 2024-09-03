# Spatiotemporal Block Adjacency Matrix (STBAM)

## Dataset

C2D2 dataset [Link](https://pern-my.sharepoint.com/:u:/g/personal/22060007_lums_edu_pk/EV2e3JCDyqxJk6wYlHJrWWoBIiIhNMGf6Dv9NDg_0Gdz_w?e=AqS7DT).

Download and place it in the `main/data/` directory.

## 1. Preprocessing 

To preprocess the data, run the following command:

```
python preprocessing/preprocessing.py --nodes 64 --compactness 20.0
```
Preprocessing involves creating the block adjacency matrix for each time series of images. This will use the original data file to create three separate files for training, testing, and validation within the same data folder. 


## 2. Training 
### GAT Model
To train the original Graph Attention Network (GAT) model on the simple block adjacency matrix without any mending, use the corresponding script:
```
python src/model/gat.py --mode train
```
### STBAM Model
To train the model using the Spatiotemporal Block Adjacency Matrix (STBAM), run the following command:
```
python src/model/stbam.py --mode train
```


## 3. Evaluation 
To evaluate the model, set the --mode parameter to test:
```
python src/model/stbam.py --mode test
```

