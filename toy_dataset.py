import json

# Define the symptom and disease sets
symptom_set = ['fever', 'cough', 'headache', 'fatigue', 'nausea']
disease_set = ['Common Cold', 'Flu', 'Migraine', 'Gastroenteritis']

# Define diagnosis rules
diagnosis_rules = {
    'Common Cold': ['cough', 'fever'],
    'Flu': ['fever', 'cough', 'fatigue'],
    'Migraine': ['headache', 'nausea'],
    'Gastroenteritis': ['nausea', 'fatigue']
}

# rule_set: knowledge base complete rules
# knowedge_region: subset of rules (fixed for an agent), latent variable, need to learn

# Mapping diseases to their index for label encoding
disease_to_index = {disease: idx for idx, disease in enumerate(disease_set)}

# Create the dataset
toy_dataset = [
    # Common Cold Cases
    {
        'symptoms': [1, 1, 0, 0, 0],  # fever, cough
        'disease': disease_to_index['Common Cold']
    },
    {
        'symptoms': [1, 1, 0, 0, 0],
        'disease': disease_to_index['Common Cold']
    },
    
    # Flu Cases
    {
        'symptoms': [1, 1, 0, 1, 0],  # fever, cough, fatigue
        'disease': disease_to_index['Flu']
    },
    {
        'symptoms': [1, 1, 0, 1, 0],
        'disease': disease_to_index['Flu']
    },
    
    # Migraine Cases
    {
        'symptoms': [0, 0, 1, 0, 1],  # headache, nausea
        'disease': disease_to_index['Migraine']
    },
    {
        'symptoms': [0, 0, 1, 0, 1],
        'disease': disease_to_index['Migraine']
    },
    
    # Gastroenteritis Cases
    {
        'symptoms': [0, 0, 0, 1, 1],  # fatigue, nausea
        'disease': disease_to_index['Gastroenteritis']
    },
    {
        'symptoms': [0, 0, 0, 1, 1],
        'disease': disease_to_index['Gastroenteritis']
    },
    
    # Mixed/Overlapping Cases
    {
        'symptoms': [1, 1, 1, 1, 1],  # All symptoms
        'disease': disease_to_index['Flu']  # Prioritize Flu in case of overlap
    },
    {
        'symptoms': [1, 0, 1, 0, 1],  # fever, headache, nausea
        'disease': disease_to_index['Migraine']
    },
    {
        'symptoms': [0, 1, 0, 1, 1],  # cough, fatigue, nausea
        'disease': disease_to_index['Gastroenteritis']
    },
    {
        'symptoms': [1, 1, 1, 0, 0],  # fever, cough, headache
        'disease': disease_to_index['Common Cold']
    }
]

# Save to JSON file
with open('toy_dataset.json', 'w') as f:
    json.dump(toy_dataset, f, indent=4)

print("Toy dataset created and saved to toy_dataset.json") 