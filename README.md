# ai201-miniproject-2324s1
Mini-Project for AI 201
Coconut leaf disease classification using Extreme Learning Machines
by Miguel Luis Martinez and Jace Roldan

## Installing the dependencies

Note: This assumes the use of pyenv, but you can use any package manager / python version manager as long as it satisfied our requirements.

1. Install pyenv to better manage your python dependencies on your local machine. The python version used in this project is found in `./python-version`. It is currently at 3.10.7.
2. If using pyenv, you can install pyenv-virtualenv: https://github.com/pyenv/pyenv-virtualenv to manage the virtual environment from your home directory.
3. Run `pyenv install` which will use the version found in the `.python-version` file. Check the python version to confirm. `python --version`.
4. Run `pyenv virtualenv 3.10 venv` to create a virtualenv tied to the 3.10 shim.
5. Run `pip install -r requirements.txt` to install the necessary packages.

## Importing the datasets
###Dataset 1: Coconut Tree Disease Dataset
1. `cd` into the root of the project.
2. Install wget if you haven't already. Run `wget "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/gh56wbsnj5-1.zip" -P "datasets/"`
3. Unzip the compressed folder. Run `unzip -q "datasets/gh56wbsnj5-1.zip" -d "datasets/"` followed by `rm "datasets/gh56wbsnj5-1.zip"` to delete the zip file.
4. Unzip the dataset once more. Run `unzip -q "datasets/Coconut Tree Disease Dataset/Coconut Tree Disease Dataset.zip" -d "datasets/"` followed by `rm "datasets/Coconut Tree Disease Dataset/Coconut Tree Disease Dataset.zip"` to delete the zip file.
Alternatively, you can run `bash preprocessing/extract.sh`.

###Dataset 2: Coconut Diseases and Pest Infestations
1. Go to https://www.kaggle.com/datasets/samitha96/coconutdiseases
2. Sign in to a Kaggle account. If you do not have a Kaggle account, sign up for one.
3. Download the dataset and extract the "archive.zip" zip file.
4. For each folder in the extracted dataset, move the folder of the same name contained inside that folder to the dataset root directory.
## Using the program
1. Run `python classify-coconut.py` to preprocess the dataset, generate feature vectors, and run the testbench for different classes of ELMs.
2. The "results" folder will contain the predictions, confusion matrices, and performance metrics for each classifier and accross all classifiers in the testbench.
