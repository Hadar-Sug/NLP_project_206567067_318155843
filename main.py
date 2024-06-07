import pandas as pd
from datasets import Dataset, load_metric
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
import torch
import logging
from sklearn.metrics import accuracy_score
import numpy as np
from earlyStopping import EarlyStoppingCallback
from trackScores import TrackAccuracyCallback
from savePredictions import SavePredictionsCallback

def main():
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    # Load your CSV file into a pandas DataFrame
    df = pd.read_csv('./Data/fineTuning_train_df_review_bot_strat_no_expl.csv')  # Replace 'your_file.csv' with your file path
    test_df = pd.read_csv('./Data/fineTuning_test_df_review_bot_strat_no_expl.csv') 
    # Ensure the labels are in integer format
    df['label'] = df['label'].astype(int)
    test_df['label'] = test_df['label'].astype(int)
    # Transform DataFrame into Hugging Face Dataset
    train_dataset = Dataset.from_pandas(df)
    test_dataset = Dataset.from_pandas(test_df)

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token

    # Function to preprocess data
    def preprocess_function(examples):
        tokens = tokenizer(examples['features'], truncation=True, padding='max_length', max_length=601)
        tokens['labels'] = examples['label']
        return tokens

    # Apply tokenization
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    test_dataset = test_dataset.map(preprocess_function, batched=True)


    metric = load_metric('accuracy')

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)
    
    # Load model (use distilgpt2 configuration)
    model = GPT2ForSequenceClassification.from_pretrained('./models3/checkpoint-26842')
    # model = GPT2ForSequenceClassification.from_pretrained('distilgpt2', num_labels=2)
    model.config.pad_token_id = tokenizer.eos_token_id  # Set pad token ID to eos token ID

    # Move model to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Define training arguments with reduced batch size and gradient accumulation
    training_args = TrainingArguments(
        output_dir='./models3',
        evaluation_strategy='epoch',
        save_strategy='epoch',  # Save the model after every epoch
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=45,
        weight_decay=0.01,
        gradient_accumulation_steps=2,
        logging_dir='./logs',  # Directory for logging
        logging_strategy='epoch',  # Log only after each epoch
        metric_for_best_model="accuracy",  # Specify the metric to monitor
        greater_is_better=True,  # Specify if higher metric value is better
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,  # Use the separate test dataset
        compute_metrics=compute_metrics,
        callbacks=[
        EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0001),  # Stop if metric doesn't improve for 3 evaluations
        TrackAccuracyCallback()  # Track accuracy over epochs
        ]
    )

    # # Train the model
    logger.info("Starting training")
    trainer.train()
    logger.info("Training complete")

    # Evaluate the model
    logger.info("Starting evaluation")
    results = trainer.evaluate()
    # Extract and print accuracy
    accuracy = results.get("eval_accuracy", None)
    if accuracy is not None:
        print(f"Accuracy: {accuracy}")
    
    # # Save the model
    logger.info("Saving the model")
    trainer.save_model('./distilgpt2-binary-classification-no-expl-review-bot-strat-2')
    logger.info("Model saved")

if __name__ == "__main__":
    main()
    