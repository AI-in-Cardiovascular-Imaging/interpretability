# Generate interpretability plots for trained models <!-- omit in toc -->

## Table of contents <!-- omit in toc -->

- [Installation](#installation)
- [Configuration](#configuration)
- [Run](#run)

## Installation

```bash
    python3 -m venv env
    source env/bin/activate
    pip install poetry
    poetry install
```
  
## Configuration

Make sure to configure the models_dir parameter in the **config.yaml** file.\
In the current state, each subdirectory of the models_dir should contain a **model.pickle** file as well as a **train_feature.csv** and a **test_feature.csv** file.

## Run

After the config file is set up properly, you can run the pipeline using:

```bash
python3 main.py
```
