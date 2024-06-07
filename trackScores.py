import os
import json
from transformers import TrainerCallback
import numpy as np

class TrackAccuracyCallback(TrainerCallback):
    def __init__(self):
        self.accuracy_scores = []

    def on_evaluate(self, args, state, control, **kwargs):
        metrics = kwargs['metrics']
        accuracy = metrics.get('eval_accuracy')
            
        if accuracy is not None:
            self.accuracy_scores.append(accuracy)

    def on_train_end(self, args, state, control, **kwargs):
        # Define the file name
        file_name = 'json_scores.json'

        # Load existing accuracy scores if the file exists
        if os.path.exists(file_name):
            with open(file_name, 'r') as f:
                existing_scores = json.load(f)
        else:
            existing_scores = []

        # Append new scores
        existing_scores.extend(self.accuracy_scores)

        # Save updated accuracy scores to JSON file
        with open(file_name, 'w') as f:
            json.dump(existing_scores, f)