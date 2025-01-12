from abc import ABC, abstractmethod
import json
import os
import torch
import numpy as np

class BaseDataset(ABC):
    @abstractmethod
    def load_data(self):
        pass
    
    @abstractmethod
    def get_input_size(self):
        pass
    
    @abstractmethod
    def get_output_size(self):
        pass

class MIMICDataset(BaseDataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.symptom_set = []
        self.disease_set = []
        self.diagnosis_rules = {}
        self.data = self.load_data()
    
    def load_data(self):
        with open(self.dataset_path, 'r') as f:
            data = json.load(f)
        return data
    
    def get_input_size(self):
        return len(self.symptom_set)
    
    def get_output_size(self):
        return len(self.disease_set)

class ARCDataset(BaseDataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.train_tasks = []
        self.eval_tasks = []
        self.load_data()
    
    def load_data(self):
        # Load training tasks
        train_dir = os.path.join(self.data_dir, 'training')
        self.train_tasks = self._load_tasks(train_dir)
        
        # Load evaluation tasks
        eval_dir = os.path.join(self.data_dir, 'evaluation')
        self.eval_tasks = self._load_tasks(eval_dir)
    
    def _load_tasks(self, directory):
        tasks = []
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                with open(os.path.join(directory, filename), 'r') as f:
                    task = json.load(f)
                    tasks.append(task)
        return tasks
    
    def get_input_size(self):
        # For ARC, input size is the grid size (typically 30x30)
        return 900  # 30x30
    
    def get_output_size(self):
        # For ARC, output size is the same as input
        return 900 