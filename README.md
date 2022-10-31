# Introduction
This code is the companion to the "Aleatoric uncertainty of overnight sleep statistics through posterior sampling using conditional normalizing flows" paper. 

## Directory Structure

- 'data_preprocessing': Instructions to download and preprocess the required data.
- 'checkpoints': Folder used to store and load model checkpoints, U-Net and U-Flow checkpoints are available.
- 'testing': Instructions to test and compare model performance.

## Environment
To make use of the environment, first install [Anaconda](https://www.anaconda.com/). In anaconda prompt navigate to the parent directory and run:
```
conda env create -f environment.yml --prefix ./env
```
and then activate the environment:
```
conda activate ./env
```

## Dataloading
Instructions for downloading and preprocessing the data can be found in the folder 'data_preprocessing'. 
To verify if recordings have been downloaded and preprocessed correctly, run:
```
python dataloading.py -data_loc="target_folder"
```
It should then list how many recordings were found by the dataloader.

## Testing
The provided model checkpoints and additional models can be tested in terms of their KL-divergence on overnight sleep statistics, accuracy and kappa, and visual inspection via plots.
Instructions regarding testing can be found in the 'testing' folder.

## Training
To train a new instance of either U-Net or U-Flow, run:
```
python Main_UNet.py  -data_loc="target_folder" -save_name="U-Net-2"
python Main_UFlow.py -data_loc="target_folder" -save_name="U-Flow-2"
```