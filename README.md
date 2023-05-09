# CS598 DL4H Final Project - BiteNet

## Steps to run code

### Prerequisites
1. A Google account that's been authorized to access the MIMIC-III dataset hosted on Google Cloud Storage by PhysioNet
2. A project set up in Google Cloud with billing enabled
3. Python 3.9 and Git installed
4. (Optional) A CUDA-enabled GPU

### Download the repo
```
git clone https://github.com/greenwaltc/CS598-DL4H-Final-Project.git && cd CS598-DL4H-Final-Project
```

### Set up Python environment from a Python 3.9 base
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Download the dataset
In the file `download-dataset.ipynb`, replace `project_id` with your Google Cloud project ID with billing enabled. Then,
run the cells in `download-dataset.ipynb`. You will be prompted to sign in to Google Cloud. After running this Jupyter
Notebook file, the MIMIC-III dataset should exist in the `./data` directory.

### Prepare the dataset
Run all the cells in `dataset-preparation.ipynb`. This will parse, filter, and prepare the data and dump the prepared 
dataset to a local .pkl file `mimic3_dataset.pkl`. This .pkl file is used for model training and evaluation in the
following section.

### Train and evaluate models
Run all the cells in `train-and-eval.ipynb`. This Jupyter Notebook loads the prepared dataset, imports BiteNet and the 
baseline models defined in the `./model` directory, defines the tasks for training and evaluation, trains the models,
evaluates the models, and records metrics that are saved to the `./results` directory.