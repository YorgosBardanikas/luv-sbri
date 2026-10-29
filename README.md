# Learning under uncertainty and volatility project

`luv-sbri` is a repository of scripts for the preprocessing and analysis of data from the project **"Learning under Uncertainty and Volatility" (LUV)**, collected at the [Stem Cell and Brain Research Institute (SBRI)](https://www.sbri.fr/) in Lyon, France.

# Overview

The LUV project studies how subjects adapt their learning and decision-making policies when the environment is uncertain. 
The research aim was to investigate the encoding of Information Gain and the prefronto-cingulate interactions during trial-and-error learning.

This repository contains the code used to:

- **Preprocess** the raw data parsing session by session.
- **Model** the subjects' behavior with Reinforcement Learning algorithms.
- **Analyze** the preprocessed neural and behavioral data using Machine Learning algorithms (SVMs, CCA, cross-temporal decoding, etc).

# Installation

Clone the repository:

```bash
git clone https://github.com/YorgosBardanikas/luv-sbri.git
cd luv-sbri
```

```bash
pip install -r requirements.txt   # once a requirements file is available
pip install -e .
```

# Repository Structure

```
luv-sbri/
├── scripts/
│   ├── analysis_scripts/
│   ├── preprocessing/
├── requirements.txt
├── setup.py
└── README.md
```

# Data Statement

The data were collected at the Stem Cell and Brain Research Institute (SBRI), Lyon, France and they are not included here. 
