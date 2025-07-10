from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score
import mlflow

# MLflow experiment config
mlflow.set_experiment("distilbert-demo")

# Load and preprocess
dataset = load_dataset("imdb", split="train[:2%]")
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
dataset = dataset.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length'), batched=True)

# Split and format
train_dataset = dataset.shuffle(seed=42).select(range(200))
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Model setup
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
args = TrainingArguments(output_dir="results", num_train_epochs=1, per_device_train_batch_size=4)
trainer = Trainer(model=model, args=args, train_dataset=train_dataset)

# 🧠 Entrenamiento con tracking de MLflow
with mlflow.start_run():
    mlflow.log_param("model_name", "distilbert-base-uncased")
    mlflow.log_param("epochs", 1)
    mlflow.log_param("batch_size", 4)
    mlflow.log_param("dataset", "imdb[:2%]")
    
    trainer.train()
    
    mlflow.log_artifacts("results/")
