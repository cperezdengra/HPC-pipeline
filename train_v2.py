from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
import numpy as np
import mlflow

# Configuración experimento MLflow
mlflow.set_experiment("distilbert-demo-v2")

# Carga y preprocesado de datos
dataset = load_dataset("imdb", split="train[:4%]").train_test_split(test_size=0.2)
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def tokenize_fn(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length')

dataset = dataset.map(tokenize_fn, batched=True)
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Definición de métrica
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

# Modelo
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Configuración de entrenamiento
args = TrainingArguments(
    output_dir="results_v2",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="no",
    report_to=[]  # evitamos duplicación con MLflow
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics
)

# Entrenamiento + tracking
with mlflow.start_run():
    mlflow.log_param("model", "distilbert-base-uncased")
    mlflow.log_param("epochs", 5)
    mlflow.log_param("batch_size", 4)

    trainer.train()

    eval_metrics = trainer.evaluate()
    mlflow.log_metrics(eval_metrics)
    mlflow.log_artifacts("results/")