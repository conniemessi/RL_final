import json
import random

# Define the symptom and disease sets
symptom_set = ['fever', 'cough', 'headache', 'fatigue', 'nausea']
disease_set = ['Common Cold', 'Flu', 'Migraine', 'Gastroenteritis']

# Define diagnosis rules where each disease has multiple symptom combination rules
diagnosis_rules = {
    'Common Cold': [
        ['cough'],                    # Rule 1: just cough
        ['cough', 'fever'],           # Rule 2: cough + fever
        ['cough', 'fatigue'],         # Rule 3: cough + fatigue
        ['cough', 'fever', 'fatigue'] # Rule 4: all three symptoms
    ],
    'Flu': [
        ['fever', 'cough', 'fatigue'],     # Rule 1: classic presentation
        ['fever', 'fatigue'],              # Rule 2: without cough
        ['fever', 'cough', 'headache'],    # Rule 3: with headache
        ['fever', 'cough', 'fatigue', 'headache']  # Rule 4: severe case
    ],
    'Migraine': [
        ['headache'],                 # Rule 1: just headache
        ['headache', 'nausea'],       # Rule 2: with nausea
        ['headache', 'fatigue'],      # Rule 3: with fatigue
        ['headache', 'nausea', 'fatigue']  # Rule 4: severe case
    ],
    'Gastroenteritis': [
        ['nausea'],                   # Rule 1: just nausea
        ['nausea', 'fatigue'],        # Rule 2: with fatigue
        ['nausea', 'fever'],          # Rule 3: with fever
        ['nausea', 'fatigue', 'fever']  # Rule 4: severe case
    ]
}

# Mapping diseases to their index for label encoding
disease_to_index = {disease: idx for idx, disease in enumerate(disease_set)}

# Create dataset based on these multiple rules
toy_dataset = []

# Helper function to convert symptom list to binary vector
def symptoms_to_vector(symptoms):
    return [1 if sym in symptoms else 0 for sym in symptom_set]

# Generate cases for each disease and each rule
for disease in disease_set:
    for rule in diagnosis_rules[disease]:
        # Create two cases for each rule
        for _ in range(2):
            toy_dataset.append({
                'symptoms': symptoms_to_vector(rule),
                'disease': disease_to_index[disease]
            })

# Add some complex/overlapping cases
complex_cases = [
    {
        'symptoms': symptoms_to_vector(['fever', 'cough', 'headache', 'fatigue', 'nausea']),
        'disease': disease_to_index['Flu']  # Full symptom set - classified as Flu
    },
    {
        'symptoms': symptoms_to_vector(['fever', 'headache', 'nausea']),
        'disease': disease_to_index['Migraine']  # Could be Flu or Migraine - classified as Migraine
    },
    {
        'symptoms': symptoms_to_vector(['cough', 'fatigue', 'nausea']),
        'disease': disease_to_index['Gastroenteritis']  # Mixed symptoms
    }
]

toy_dataset.extend(complex_cases)

# Shuffle the dataset
random.shuffle(toy_dataset)

# Save to JSON file
with open('toy_dataset.json', 'w') as f:
    json.dump(toy_dataset, f, indent=4)

print("Enhanced toy dataset with multiple rules per disease created and saved to toy_dataset.json") 