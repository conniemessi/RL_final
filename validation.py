from typing import List, Dict, Set, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

class MedicalGuidelineValidator:
    def __init__(self):
        # Define standard medical guidelines
        self.guidelines = self._initialize_guidelines()
        # Define critical paths that must be followed
        self.critical_paths = self._initialize_critical_paths()
        # Define contraindications
        self.contraindications = self._initialize_contraindications()
        
    def _initialize_guidelines(self) -> Dict[str, Dict]:
        """
        Initialize medical guidelines based on standard protocols
        Returns a nested dictionary of conditions and their recommended diagnostic paths
        """
        return {
            'pneumonia': {
                'required_symptoms': {'fever', 'cough'},
                'supporting_symptoms': {'shortness_of_breath', 'chest_pain'},
                'minimum_required': 2,
                'minimum_supporting': 1,
                'required_tests': ['chest_xray', 'respiratory_rate'],
                'severity_indicators': ['oxygen_saturation < 92%', 'respiratory_rate > 30']
            },
            'myocardial_infarction': {
                'required_symptoms': {'chest_pain'},
                'supporting_symptoms': {'shortness_of_breath', 'nausea', 'sweating'},
                'minimum_required': 1,
                'minimum_supporting': 1,
                'required_tests': ['ecg', 'troponin'],
                'severity_indicators': ['st_elevation', 'troponin > 0.04']
            },
            # Add more conditions as needed
        }
    
    def _initialize_critical_paths(self) -> Dict[str, List[str]]:
        """
        Define critical diagnostic paths that must be followed in specific situations
        """
        return {
            'chest_pain': ['vital_signs', 'ecg', 'troponin'],
            'shortness_of_breath': ['vital_signs', 'oxygen_saturation', 'respiratory_rate'],
            'severe_fever': ['blood_culture', 'complete_blood_count']
        }
    
    def _initialize_contraindications(self) -> Dict[str, Set[str]]:
        """
        Define contraindicated actions or combinations
        """
        return {
            'aspirin': {'active_bleeding', 'bleeding_disorder'},
            'beta_blockers': {'severe_asthma', 'heart_block'},
        }

class ValidationMetrics:
    def __init__(self, validator: MedicalGuidelineValidator):
        self.validator = validator
        self.metrics = {
            'guideline_adherence': [],
            'critical_path_compliance': [],
            'safety_score': [],
            'diagnosis_accuracy': [],
            'path_efficiency': []
        }
    
    def evaluate_path(self, 
                     path: List[str], 
                     true_diagnosis: str, 
                     patient_symptoms: Set[str],
                     taken_actions: List[str]) -> Dict[str, float]:
        """
        Evaluate a diagnostic path against medical guidelines
        """
        # Calculate guideline adherence
        guideline_score = self._calculate_guideline_adherence(
            path, true_diagnosis, patient_symptoms)
        
        # Check critical path compliance
        critical_path_score = self._check_critical_path_compliance(path)
        
        # Evaluate safety
        safety_score = self._evaluate_safety(taken_actions, patient_symptoms)
        
        # Calculate diagnosis accuracy
        diagnosis_accuracy = self._calculate_diagnosis_accuracy(
            path[-1], true_diagnosis)
        
        # Evaluate path efficiency
        path_efficiency = self._calculate_path_efficiency(path)
        
        results = {
            'guideline_adherence': guideline_score,
            'critical_path_compliance': critical_path_score,
            'safety_score': safety_score,
            'diagnosis_accuracy': diagnosis_accuracy,
            'path_efficiency': path_efficiency
        }
        
        # Update metrics history
        for key, value in results.items():
            self.metrics[key].append(value)
            
        return results
    
    def _calculate_guideline_adherence(self, 
                                     path: List[str], 
                                     true_diagnosis: str,
                                     patient_symptoms: Set[str]) -> float:
        """
        Calculate how well the path adheres to medical guidelines
        """
        if true_diagnosis not in self.validator.guidelines:
            return 0.0
            
        guideline = self.validator.guidelines[true_diagnosis]
        required_steps = set(guideline['required_tests'])
        taken_steps = set(path)
        
        # Check required symptoms
        required_symptoms_present = len(
            guideline['required_symptoms'].intersection(patient_symptoms)
        ) >= guideline['minimum_required']
        
        # Check supporting symptoms
        supporting_symptoms_present = len(
            guideline['supporting_symptoms'].intersection(patient_symptoms)
        ) >= guideline['minimum_supporting']
        
        # Calculate adherence score
        required_steps_taken = len(required_steps.intersection(taken_steps))
        adherence_score = (
            (required_steps_taken / len(required_steps)) * 0.6 +
            (required_symptoms_present * 0.2) +
            (supporting_symptoms_present * 0.2)
        )
        
        return adherence_score
    
    def _check_critical_path_compliance(self, path: List[str]) -> float:
        """
        Check if critical diagnostic paths were followed when necessary
        """
        compliance_scores = []
        
        for trigger, required_steps in self.validator.critical_paths.items():
            if trigger in path:
                # Check if all required steps were taken in order
                trigger_index = path.index(trigger)
                required_steps_taken = all(
                    step in path[trigger_index:] for step in required_steps
                )
                compliance_scores.append(float(required_steps_taken))
                
        return np.mean(compliance_scores) if compliance_scores else 1.0
    
    def _evaluate_safety(self, 
                        taken_actions: List[str], 
                        patient_symptoms: Set[str]) -> float:
        """
        Evaluate the safety of the diagnostic path
        """
        safety_violations = 0
        
        for action in taken_actions:
            if action in self.validator.contraindications:
                contraindicated_conditions = self.validator.contraindications[action]
                if contraindicated_conditions.intersection(patient_symptoms):
                    safety_violations += 1
                    
        return 1.0 - (safety_violations / len(taken_actions) if taken_actions else 0)
    
    def _calculate_diagnosis_accuracy(self, 
                                    predicted_diagnosis: str, 
                                    true_diagnosis: str) -> float:
        """
        Calculate the accuracy of the final diagnosis
        """
        return float(predicted_diagnosis == true_diagnosis)
    
    def _calculate_path_efficiency(self, path: List[str]) -> float:
        """
        Calculate the efficiency of the diagnostic path
        """
        # Assume maximum reasonable path length is 10
        MAX_REASONABLE_LENGTH = 10
        return max(0, 1 - (len(path) / MAX_REASONABLE_LENGTH))
    
    def get_summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate summary statistics for all metrics
        """
        summary = {}
        for metric_name, values in self.metrics.items():
            summary[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        return summary 