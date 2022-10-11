# Data Preprocessing
Instructions to download and preprocess the required and optional datasets.

## Downloading the data
### (required) test dataset
The IS-RC dataset is required to run the testing scripts and can be downloaded from here: [link](https://stanfordmedicine.app.box.com/s/r9e92ygq0erf7hn5re6j51aaggf50jly/folder/53209541138).

### (optional) train and validation datasets
These datasets are optional and only needed for training:
- ISRUC subset I and III can be downloaded from here: [link](https://sleeptight.isr.uc.pt/)
- SSC can be downloaded from here on request: [link](https://sleepdata.org/datasets/mnc)
- DOD-O and DOD-H can be downloaded from here: [link1](https://dreem-dod-o.s3.eu-west-3.amazonaws.com/index.html), [link2](https://dreem-dod-h.s3.eu-west-3.amazonaws.com/index.html), [link3](https://github.com/Dreem-Organization/dreem-learning-evaluation/tree/master/scorers/dodo), [link4](https://github.com/Dreem-Organization/dreem-learning-evaluation/tree/master/scorers/dodh)

## Preprocessing the Data
The preprocessing of the datasets is done per data source. 
In each of these commands data_loc="folder" should point towards the raw data and target_loc="folder" should point towards the folder where you want to save all the preprocessed data. 
To preprocess the required test dataset run:
```
python extract_IS_RC.py -data_loc="download_folder" -target_loc="target_folder"
```
To preprocess the optional training and validation sets run:
```
python extract_DOD_H.py -data_loc="download_folder" -target_loc="target_folder"
python extract_DOD_O.py -data_loc="download_folder" -target_loc="target_folder"
python extract_ISRUC.py -data_loc="download_folder" -target_loc="target_folder"
python extract_SSC.py -data_loc="download_folder" -target_loc="target_folder"
```