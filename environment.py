import numpy as np
import torch
import json

class DiagnosisEnvironment:
    def __init__(self, symptom_set, disease_set, diagnosis_rules, dataset_path, max_attempts=5):
        self.symptom_set = symptom_set
        self.disease_set = disease_set
        self.diagnosis_rules = diagnosis_rules
        self.dataset = self.load_dataset(dataset_path)
        self.num_cases = len(self.dataset)
        self.current_index = -1  # Initialize to -1; will be set to 0 on reset
        self.max_attempts = max_attempts
        self.current_attempts = 0
    
    def load_dataset(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        return data
    
    def reset(self):
        self.current_index = 0
        self.current_attempts = 0
        self.current_symptoms = self.dataset[self.current_index]['symptoms']
        self.true_disease = self.dataset[self.current_index]['disease']
        self.symptom_history = np.zeros(len(self.symptom_set), dtype=int)
        return self.get_state()
    
    def step(self, predicted_disease_probs):
        """
        Process one diagnosis attempt with multi-level entropy thresholds
        """
        # Debug print
        print(f"Current case: {self.current_index}, Attempts: {self.current_attempts}, Total cases: {self.num_cases}")
        
        # First check if we've exceeded total cases
        if self.current_index >= self.num_cases:
            print("All cases processed")
            return self.get_state(), 0, True, {
                'entropy': float('inf'),
                'confidence_level': 'episode_complete',
                'attempts': self.current_attempts,
                'max_attempts_reached': False,
                'true_disease': -1  # No more cases
            }
        
        # Check if we've already hit max attempts
        if self.current_attempts >= self.max_attempts:
            print("Max attempts reached, moving to next case")
            reward = -0.2  # Penalty for failing to diagnose within max attempts
            
            # Move to next case
            self.current_index += 1
            if self.current_index >= self.num_cases:
                print("All cases processed")
                done = True
            else:
                print(f"Moving to case {self.current_index}")
                self.current_symptoms = self.dataset[self.current_index]['symptoms']
                self.true_disease = self.dataset[self.current_index]['disease']
                self.symptom_history = [0] * len(self.symptom_set)
                self.current_attempts = 0
                done = False  # Continue with next case
            
            next_state = self.get_state()
            info = {
                'entropy': float('inf'),
                'confidence_level': 'max_attempts_reached',
                'attempts': self.current_attempts,
                'max_attempts_reached': True,
                'true_disease': self.true_disease
            }
            return next_state, reward, done, info
        
        # If we haven't hit max attempts, proceed with normal step
        self.current_attempts += 1
        
        # Compute entropy of prediction
        entropy = torch.distributions.Categorical(probs=predicted_disease_probs).entropy()
        
        # Multi-level entropy thresholds
        VERY_CONFIDENT = 0.4    # ~80% sure of one disease
        UNCERTAIN = 1.2         # Close to uniform distribution
        
        # Determine reward based on confidence
        if entropy <= VERY_CONFIDENT:
            print("Confident prediction made")
            reward = 1.0
            done = True  # Move to next case
        elif entropy >= UNCERTAIN:
            reward = -0.2
            done = False
        else:
            reward = -0.1
            done = False
        
        # Handle case transition if confident prediction made
        if done:
            print("Moving to next case after confident prediction")
            self.current_index += 1
            if self.current_index >= self.num_cases:
                print("All cases processed")
                done = True
            else:
                print(f"Moving to case {self.current_index}")
                self.current_symptoms = self.dataset[self.current_index]['symptoms']
                self.true_disease = self.dataset[self.current_index]['disease']
                self.symptom_history = [0] * len(self.symptom_set)
                self.current_attempts = 0
                done = False  # Continue with next case
        
        next_state = self.get_state()
        info = {
            'entropy': entropy.item(),
            'confidence_level': 'high' if entropy <= VERY_CONFIDENT else 'low' if entropy >= UNCERTAIN else 'medium',
            'attempts': self.current_attempts,
            'max_attempts_reached': self.current_attempts >= self.max_attempts,
            'true_disease': self.true_disease
        }
        
        return next_state, reward, done, info
    
    def get_state(self):
        return {
            'current_symptoms': torch.tensor(self.current_symptoms, dtype=torch.float32),
            'symptom_history': torch.tensor(self.symptom_history, dtype=torch.float32)
        } 