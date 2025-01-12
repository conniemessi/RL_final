# Define comprehensive symptom and disease sets based on Training.csv
import json
import random


symptom_set = ['abdominal_pain', 'acidity', 'burning_micturition', 'chills', 'continuous_sneezing', 
               'cough', 'indigestion', 'itching', 'loss_of_appetite', 'nausea', 'nodal_skin_eruptions', 
               'shivering', 'skin_rash', 'stomach_pain', 'ulcers_on_tongue', 'vomiting', 'yellowish_skin']

disease_set = [
    'Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis',
    'Drug Reaction', 'Peptic ulcer disease'
]

# Define comprehensive diagnosis rules
diagnosis_rules = {
    'Fungal infection': [
        ['itching', 'skin_rash'],
        ['itching', 'nodal_skin_eruptions'],
        ['skin_rash', 'nodal_skin_eruptions'],
        ['itching', 'skin_rash', 'nodal_skin_eruptions']
    ],
    'Allergy': [
        ['continuous_sneezing', 'shivering'],
        ['continuous_sneezing', 'chills'],
        ['shivering', 'chills'],
        ['continuous_sneezing', 'shivering', 'chills']
    ],
    'GERD': [
        ['stomach_pain', 'acidity'],
        ['stomach_pain', 'ulcers_on_tongue'],
        ['acidity', 'ulcers_on_tongue', 'vomiting'],
        ['stomach_pain', 'acidity', 'ulcers_on_tongue', 'cough']
    ],
    'Chronic cholestasis': [
        ['itching', 'vomiting'],
        ['vomiting', 'yellowish_skin'],
        ['yellowish_skin', 'nausea', 'loss_of_appetite'],
        ['itching', 'vomiting', 'yellowish_skin', 'nausea', 'loss_of_appetite']
    ],
    'Drug Reaction': [
        ['itching', 'skin_rash'],
        ['itching', 'stomach_pain'],
        ['skin_rash', 'stomach_pain'],
        ['itching', 'skin_rash', 'stomach_pain', 'burning_micturition']
    ],
    'Peptic ulcer disease': [
        ['vomiting', 'indigestion'],
        ['vomiting', 'abdominal_pain'],
        ['stomach_pain', 'loss_of_appetite'],
        ['vomiting', 'indigestion', 'abdominal_pain', 'loss_of_appetite']
    ]
}

# Add more diseases with their rules...
# (You can continue adding more diseases and their rules following the same pattern)
used_symptoms = set()
for disease in diagnosis_rules:
    for rule in diagnosis_rules[disease]:
        used_symptoms.update(rule)

def symptoms_to_vector(symptoms):
    return [1 if sym in symptoms else 0 for sym in symptom_set]

# Create dataset
toy_dataset = []

# Generate cases for each disease and each rule
# for disease in diagnosis_rules:
#     for rule in diagnosis_rules[disease]:
#         # Create multiple cases for each rule
#         for _ in range(3):  # Creating 3 cases per rule
#             toy_dataset.append({
#                 'symptoms': symptoms_to_vector(rule),
#                 'disease': disease_set.index(disease)
#             })

# Add real complex cases from Training.csv for our specific diseases
complex_cases = [
    # Fungal infection cases
    {
        'symptoms': symptoms_to_vector(['itching', 'skin_rash', 'nodal_skin_eruptions']),
        'disease': disease_set.index('Fungal infection')
    },
    # Allergy cases
    {
        'symptoms': symptoms_to_vector(['continuous_sneezing', 'shivering', 'chills', 'watering_from_eyes']),
        'disease': disease_set.index('Allergy')
    },
    # GERD cases
    {
        'symptoms': symptoms_to_vector(['stomach_pain', 'acidity', 'ulcers_on_tongue', 'vomiting', 'cough']),
        'disease': disease_set.index('GERD')
    },
    # Chronic cholestasis cases
    {
        'symptoms': symptoms_to_vector(['itching', 'vomiting', 'yellowish_skin', 'nausea', 'loss_of_appetite', 'abdominal_pain']),
        'disease': disease_set.index('Chronic cholestasis')
    },
    # Drug Reaction cases
    {
        'symptoms': symptoms_to_vector(['itching', 'skin_rash', 'stomach_pain', 'burning_micturition', 'spotting_urination']),
        'disease': disease_set.index('Drug Reaction')
    },
    # Peptic ulcer disease cases
    {
        'symptoms': symptoms_to_vector(['vomiting', 'loss_of_appetite', 'abdominal_pain', 'passage_of_gases', 'internal_itching']),
        'disease': disease_set.index('Peptic ulcer disease')
    }
]

toy_dataset.extend(complex_cases)

# Shuffle the dataset
random.shuffle(toy_dataset)

# Save to JSON file
with open('training_dataset.json', 'w') as f:
    json.dump(toy_dataset, f, indent=4)

print("Enhanced comprehensive dataset created and saved to training_dataset.json")