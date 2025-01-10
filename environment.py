import numpy as np
import torch
import json

class DiagnosisEnvironment:
    def __init__(self, symptom_set, disease_set, diagnosis_rules, dataset_path='toy_dataset.json'):
        self.symptom_set = symptom_set
        self.disease_set = disease_set
        self.diagnosis_rules = diagnosis_rules
        self.dataset = self.load_dataset(dataset_path)
        self.num_cases = len(self.dataset)
        self.current_index = -1  # Initialize to -1; will be set to 0 on reset
    
    def load_dataset(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        return data
    
    def reset(self):
        self.current_index = 0
        self.current_symptoms = self.dataset[self.current_index]['symptoms']
        self.true_disease = self.dataset[self.current_index]['disease']
        self.symptom_history = np.zeros(len(self.symptom_set), dtype=int)
        return self.get_state()
    
    def step(self, predicted_disease):
        # entropy of predicted_disease small
        if predicted_disease == self.true_disease:
            reward = 1.0
            done = True
        else:
            reward = -0.1
            done = False
            # Move to the next case or allow retries
            self.current_index += 1
            if self.current_index >= self.num_cases:
                done = True  # End episode if no more cases
            else:
                self.current_symptoms = self.dataset[self.current_index]['symptoms']
                self.true_disease = self.dataset[self.current_index]['disease']
                # Update symptom history
                self.symptom_history = np.bitwise_or(self.symptom_history, self.current_symptoms)
        next_state = self.get_state()
        info = {'true_disease': self.true_disease}
        return next_state, reward, done, info
    
    def get_state(self):
        return {
            'current_symptoms': torch.tensor(self.current_symptoms, dtype=torch.float32),
            'symptom_history': torch.tensor(self.symptom_history, dtype=torch.float32)
        } 