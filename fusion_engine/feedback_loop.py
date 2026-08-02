import pandas as pd
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class FeedbackLoop:
    """
    Manages the 'Active Learning' cycle.
    1. Receives feedback from Dashboard (True Positive / False Positive).
    2. Logs the flow data + Correct Label to 'verified_new.csv'.
    3. Triggers Retraining if enough new samples are collected.
    """
    def __init__(self, dataset_dir=None):
        if dataset_dir is None:
            self.dataset_dir = PROJECT_ROOT / "dataset"
        else:
            self.dataset_dir = Path(dataset_dir)
        self.feedback_file = self.dataset_dir / "verified_new.csv"
        self.retrain_threshold = 100 # Retrain after 100 new verified samples

    def log_feedback(self, flow_data, user_label):
        """
        flow_data: dict of flow features (must match loader columns).
        user_label: 0 (Benign) or 1 (Attack).
        """
        # Add timestamp and label
        entry = flow_data.copy()
        entry['timestamp'] = time.time()
        entry['label'] = user_label # Confirmed Label
        
        df = pd.DataFrame([entry])
        
        # Append to CSV
        if not self.feedback_file.exists():
            df.to_csv(self.feedback_file, index=False, mode='w', header=True)
        else:
            df.to_csv(self.feedback_file, index=False, mode='a', header=False)
            
        print(f"Feedback logged. Total verified samples: {self.get_sample_count()}")
        
        # Check for Retrain Trigger
        if self.get_sample_count() >= self.retrain_threshold:
            print(">> RETRAIN THRESHOLD REACHED. Initiating Retrain Pipeline...")
            # In a real app, this would spawn a subprocess or Celery task
            # from models.retrain_pipeline import retrain_model
            # retrain_model()
            pass

    def get_sample_count(self):
        if not self.feedback_file.exists():
            return 0
        try:
            with open(self.feedback_file, 'r') as f:
                return sum(1 for line in f) - 1 # Minus header
        except:
            return 0

if __name__ == "__main__":
    # Test
    loop = FeedbackLoop()
    mock_flow = {'Flow Duration': 500, 'Total Fwd Packets': 10}
    loop.log_feedback(mock_flow, 1) # Confirm Attack
