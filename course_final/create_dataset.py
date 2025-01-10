import pandas as pd
import numpy as np

def create_toy_dataset():
    # Simplified symptoms and diagnoses
    symptoms = [
        'fever', 'cough', 'shortness_of_breath', 'chest_pain', 
        'fatigue', 'headache', 'nausea', 'abdominal_pain'
    ]
    
    diagnoses = [
        'pneumonia', 'covid19', 'myocardial_infarction', 
        'bronchitis', 'gastroenteritis'
    ]
    
    # Create synthetic patient cases
    n_cases = 100
    synthetic_data = []
    
    for _ in range(n_cases):
        if np.random.random() < 0.2:  # Pneumonia case
            symptoms_present = ['fever', 'cough', 'shortness_of_breath']
            if np.random.random() < 0.7:
                symptoms_present.append('chest_pain')
            if np.random.random() < 0.6:
                symptoms_present.append('fatigue')
            diagnosis = 'pneumonia'
            
        elif np.random.random() < 0.4:  # COVID-19 case
            symptoms_present = ['fever', 'cough', 'fatigue']
            if np.random.random() < 0.8:
                symptoms_present.append('shortness_of_breath')
            if np.random.random() < 0.3:
                symptoms_present.append('headache')
            diagnosis = 'covid19'
            
        elif np.random.random() < 0.6:  # Myocardial infarction case
            symptoms_present = ['chest_pain', 'shortness_of_breath']
            if np.random.random() < 0.7:
                symptoms_present.append('fatigue')
            if np.random.random() < 0.4:
                symptoms_present.append('nausea')
            diagnosis = 'myocardial_infarction'
            
        elif np.random.random() < 0.8:  # Bronchitis case
            symptoms_present = ['cough', 'fatigue']
            if np.random.random() < 0.6:
                symptoms_present.append('fever')
            if np.random.random() < 0.5:
                symptoms_present.append('shortness_of_breath')
            diagnosis = 'bronchitis'
            
        else:  # Gastroenteritis case
            symptoms_present = ['nausea', 'abdominal_pain']
            if np.random.random() < 0.6:
                symptoms_present.append('fatigue')
            if np.random.random() < 0.3:
                symptoms_present.append('fever')
            diagnosis = 'gastroenteritis'
        
        synthetic_data.append({
            'symptoms': symptoms_present,
            'diagnosis': diagnosis
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(synthetic_data)
    
    # Save to CSV
    df.to_csv('synthetic_medical_data.csv', index=False)
    print(f"Dataset created with {len(df)} cases and saved to 'synthetic_medical_data.csv'")
    
    return df, symptoms, diagnoses

if __name__ == "__main__":
    # Create the dataset when running this file directly
    df, symptoms, diagnoses = create_toy_dataset()
    
    # Print some basic statistics
    print("\nDataset Statistics:")
    print(f"Number of unique symptoms: {len(symptoms)}")
    print(f"Number of unique diagnoses: {len(diagnoses)}")
    print("\nDiagnosis distribution:")
    print(df['diagnosis'].value_counts()) 