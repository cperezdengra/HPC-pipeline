# AI Pipeline Setup

This guide will walk you through setting up your AI Factory project, including creating a virtual environment, installing essential tools, and initializing DVC.

## 1. Environment Setup

First, let's create and activate a virtual environment to manage our project's dependencies.

Requirements:

-docker & docker compose or Docker Desktop

```bash
python3 -m venv ai-factory-env
source ai-factory-env/bin/activate
pip install --upgrade pip
```

## 2. Install Required Tools

Next, install the necessary libraries for data versioning, machine learning, and automation.

```bash
pip install "dvc[gs]" mlflow transformers datasets scikit-learn ansible
pip install torch torchvision torchaudio
```

## 3. Initialize Project

Now, let's set up your project directory, initialize a Git repository, and configure DVC.

```bash
mkdir ai-factory-pipeline && cd ai-factory-pipeline
git init
dvc init
```
