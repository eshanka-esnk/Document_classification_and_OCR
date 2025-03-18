# Document classification and OCR <a href="https://github.com/eshanka-esnk/Document_classification_and_OCR/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/github/github-tile.svg" alt="pytorch" width="15" height="15"/></a>

<a href="#" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/python/python-ar21.svg" alt="pytorch" width="80" height="40"/></a>
<a href="https://pytorch.org/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/pytorch/pytorch-ar21.svg" alt="pytorch" width="80" height="40"/></a>
<a href="https://pytorch.org/" target="_blank" rel="noreferrer"> <img src="https://streamlit.io/images/brand/streamlit-logo-primary-colormark-lighttext.svg" alt="pytorch" width="80" height="40"/></a>

A tool which utilies the power of machine learning to classfy documents and extract details OCR.

- Simple UI
- Document classification
- Tessaract OCR

## Features

- Import a image file(png, jpg, jpeg) or even a PDF file containing multiple documents in one and watch it magically extract details.
- Drag and drop images.
- Save extracted information in a stuctured format.
- Intelligent classification of documents.

## Installation

This requires <a href="#" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/python/python-icon.svg" alt="pytorch" width="20" height="20"/></a>python v3.10+ to run.
Linux environment is recommended.

### Clone repo in to a directory
```sh
git clone https://github.com/eshanka-esnk/Document_classification_and_OCR.git
```

### Create Virtual Env and install all dependencies.
```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
> Note: When installing in machine with gpu make sure to install torch with cuda enabled.

### Install Poppler as a dependancy.
```sh
sudo apt-get install poppler-utils
```
> Note: Use [chocolatey](https://community.chocolatey.org/) to install [poppler](https://community.chocolatey.org/packages/poppler)https://community.chocolatey.org/packages/poppler on windows and add it's path to the .env file as key 'tesseract_path'.

### Downloading model

method 1
> Note: The model file is uploaded into git as parts(50MB each) to satisfy max file limit in github.
- Use a tool([Winrar](https://www.win-rar.com/) recommended) to extract 'best_model_compressed_part_01.zip' and extract with all parts.
- Model file 'best_model.pth' should be extracted.

method2
> Note: Use [guide](https://git-lfs.com/) to install git lfs
- After cloning repo, run
```sh
git lfs pull
```
- Use a tool([Winrar](https://www.win-rar.com/) recommended) to extract 'best_model_compressed.zip'.
- Model file 'best_model.pth' should be extracted.

### Run ui
> Note: Make sure [streamlit](https://streamlit.io/) is installed.
```sh
streamlit run ui.py
```

## Author

Eshanka Ruhunuhewa
<a href="https://www.linkedin.com/in/eshanka-ruhunuhewa/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/linkedin/linkedin-icon.svg" alt="pytorch" width="15" height="15"/></a>
<a href="https://github.com/eshanka-esnk/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/github/github-tile.svg" alt="pytorch" width="15" height="15"/></a>