# Self-Supervised Vision Transformers for Breast Histopathology Image Embeddings in Invasive Ductal Carcinoma Detection

_This project was developed by Faiaz Rahman originally for CS 482: Applied Machine Learning under Dr. David van Dijk at Yale University._

# Setup

We recommend using a virtual environment via Conda. We have provided an environment YAML file to rebuild the same virtual environment used in our experiments. We use Python 3.7, PyTorch 1.11.0, and CUDA 11.3.1.

```
conda env create --file environment.yml
conda activate ssbh-transformers
```

# Data

```
cd data
kaggle datasets download paultimothymooney/breast-histopathology-images
unzip breast-histopathology-images.zip
```

If you have issues with the Kaggle API, create a separate virtual environment (to be used only for data downloading) and try running as follows.
```
cd data
conda env create --name download-data python=3.7
conda activate download-data
pip install kaggle
python -m pip install requirements.txt
kaggle datasets download paultimothymooney/breast-histopathology-images
unzip breast-histopathology-images.zip
conda deactivate
```

Then, reactivate your main `ssbh-transformers` virtual environment to continue with running the experiments.
