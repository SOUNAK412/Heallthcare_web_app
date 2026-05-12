import sklearn
import numpy as np
import pandas as pd
import pickle
import re
import difflib
from flask import Flask, request, render_template, jsonify
from collections import defaultdict
from textblob import TextBlob

# flask app
app = Flask(__name__)

# load databasedataset===================================
sym_des = pd.read_csv("dataset/symtoms_df.csv")
precautions = pd.read_csv("dataset/precautions_df.csv")
workout = pd.read_csv("dataset/workout_df.csv")
description = pd.read_csv("dataset/description.csv")
medications = pd.read_csv('dataset/medications.csv')
# prefer the more complete file if present
import os
if medications.shape[0] < 10:
    alt = 'dataset/medications (1).csv'
    if os.path.exists(alt):
        medications = pd.read_csv(alt)
diets = pd.read_csv("dataset/diets.csv")

disease_medicine = pd.DataFrame(columns=['Disease', 'Medicine'])
disease_medicine_csv = "dataset/disease_medicine.csv"
disease_medicine_xlsx = "dataset/disease_medicine.csv.xlsx"
if os.path.exists(disease_medicine_csv):
    try:
        disease_medicine = pd.read_csv(disease_medicine_csv)
    except Exception:
        disease_medicine = pd.DataFrame(columns=['Disease', 'Medicine'])
elif os.path.exists(disease_medicine_xlsx):
    try:
        disease_medicine = pd.read_excel(disease_medicine_xlsx)
    except Exception:
        disease_medicine = pd.DataFrame(columns=['Disease', 'Medicine'])

if 'Disease' not in disease_medicine.columns or 'Medicine' not in disease_medicine.columns:
    disease_medicine = pd.DataFrame(columns=['Disease', 'Medicine'])

# load medicine details file for richer medication output
medicine_details = None
medicine_details_path_csv = "dataset/Medicine_Details.csv"
medicine_details_path_xlsx = "dataset/Medicine_Details.xlsx"
if os.path.exists(medicine_details_path_csv):
    medicine_details = pd.read_csv(medicine_details_path_csv)
elif os.path.exists(medicine_details_path_xlsx):
    medicine_details = pd.read_excel(medicine_details_path_xlsx, sheet_name="Medicine_Details")
else:
    raise FileNotFoundError(f"Medicine details file not found: {medicine_details_path_csv} or {medicine_details_path_xlsx}")

medicine_details.columns = [c.strip() for c in medicine_details.columns]
medicine_details.fillna("", inplace=True)

# Build disease to medicines mapping from Medicine_Details
from collections import defaultdict
disease_to_medicines = defaultdict(list)
medicine_to_review = {}
for _, row in medicine_details.iterrows():
    medicine_name = row['Medicine Name']
    excellent_review = row['Excellent Review %']
    try:
        excellent_review = float(excellent_review)
    except:
        excellent_review = 0
    medicine_to_review[medicine_name] = excellent_review

# After diseases_list is defined, build the mapping
# This will be done later in the code

# Function to get medicine details from Medicine_Details.csv
def get_medicine_detail_from_excel(medicine_name):
    if medicine_details is None:
        return None
    row = medicine_details[medicine_details['Medicine Name'].str.lower() == medicine_name.lower()]
    if row.empty:
        return None
    row = row.iloc[0]
    return {
        'name': row['Medicine Name'],
        'purpose': row['Uses'],
        'dosage': '',  # not in dataset
        'precautions': row['Side_effects'],
        'when_to_consult': 'Consult a healthcare professional.'
    }

# load model===========================================
svc = pickle.load(open('model/svc.pkl','rb'))
label_encoder = None
label_encoder_path = 'model/label_encoder.pkl'
if os.path.exists(label_encoder_path):
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)

# Load training data for a simple rule-based fallback predictor
try:
    training_df = pd.read_csv('dataset/Training (1).csv')
    # The last column is prognosis (disease name); other columns are symptom flags
    if 'prognosis' in training_df.columns:
        symptom_cols = [c for c in training_df.columns if c != 'prognosis']
    else:
        # fallback: assume last column is prognosis
        symptom_cols = training_df.columns[:-1].tolist()
    # build disease -> symptom frequency vector
    disease_symptom_freq = {}
    for disease, group in training_df.groupby(training_df.columns[-1]):
        vals = group[symptom_cols].sum(axis=0).values.astype(float)
        disease_symptom_freq[disease] = vals
    disease_symptom_cols = symptom_cols
except Exception:
    training_df = None
    disease_symptom_freq = {}
    disease_symptom_cols = []

# Basic medicines information (safe, commonly used options).
# Only general guidance is provided — not a substitute for medical advice.
MEDICINES_INFO = {
    'Paracetamol': {
        'purpose': 'Relieves pain and reduces fever',
        'dosage': '500–1000 mg every 4–6 hours as needed (max ~3000 mg/day for adults)',
        'precautions': 'Avoid exceeding recommended dose; check liver disease before use; avoid with other acetaminophen-containing products.',
        'when_to_consult': 'If fever persists >3 days, severe pain, signs of liver problems (jaundice), or overdose.'
    },
    'Ibuprofen': {
        'purpose': 'Nonsteroidal anti-inflammatory for pain, inflammation, and fever',
        'dosage': '200–400 mg every 4–6 hours as needed (typical OTC max 1200 mg/day)',
        'precautions': 'Take with food to reduce stomach upset; avoid if history of peptic ulcer, uncontrolled high blood pressure, or severe kidney disease.',
        'when_to_consult': 'If severe stomach pain, blood in stools, shortness of breath, or symptoms persist.'
    },
    'Cetirizine': {
        'purpose': 'Oral antihistamine for allergic symptoms (sneezing, itching, runny nose)',
        'dosage': '10 mg once daily (adults)',
        'precautions': 'May cause drowsiness in some people; use caution when driving.',
        'when_to_consult': 'If symptoms persist despite treatment or if severe allergic reaction occurs.'
    },
    'Loratadine': {
        'purpose': 'Non-drowsy oral antihistamine for allergy symptoms',
        'dosage': '10 mg once daily (adults)',
        'precautions': 'Generally well tolerated; check interactions with other medications.',
        'when_to_consult': 'If no improvement or symptoms worsen.'
    },
    'Amoxicillin': {
        'purpose': 'Broad-spectrum antibiotic used for some bacterial infections (prescription only)',
        'dosage': 'Typical adult dose varies by infection (commonly 500 mg every 8 hours); follow prescriber instructions',
        'precautions': 'Only use when prescribed by a clinician; report penicillin allergy or rash.',
        'when_to_consult': 'If signs of allergic reaction (rash, swelling, breathing problems), severe diarrhea, or no improvement.'
    },
    'Azithromycin': {
        'purpose': 'Antibiotic used for certain respiratory and other bacterial infections (prescription only)',
        'dosage': 'Follow prescriber instructions; common short-course regimens exist',
        'precautions': 'Prescription-only; report liver disease or heart rhythm issues.',
        'when_to_consult': 'If allergic reaction, severe diarrhea, or symptoms do not improve.'
    },
    'Topical hydrocortisone': {
        'purpose': 'Mild topical steroid for itching and inflammation (skin)',
        'dosage': 'Apply a thin layer to affected area 1–2 times daily as directed',
        'precautions': 'Avoid long-term use on large areas; do not use on infected skin without advice.',
        'when_to_consult': 'If skin worsens, shows signs of infection, or no improvement.'
    }
}
# Map general category names (as used in dataset) to example medicines
CATEGORY_TO_EXAMPLES = {
    'Antiviral drugs': ['Acyclovir'],
    'Pain relievers': ['Paracetamol', 'Ibuprofen'],
    'Antihistamines': ['Cetirizine', 'Loratadine'],
    'Antipyretics': ['Paracetamol', 'Ibuprofen'],
    'Topical antifungal': ['Clotrimazole', 'Ketoconazole'],
    'Antifungal Cream': ['Clotrimazole'],
    'Antibiotics': ['Amoxicillin', 'Azithromycin'],
    'Oral antibiotics': ['Amoxicillin', 'Azithromycin'],
    'Topical antibiotics': ['Amoxicillin'],
    'Antiemetic drugs': ['Ondansetron'],
    'Antidiarrheal drugs': ['Loperamide'],
    'Anticholinergics': ['Ipratropium'],
    'Anticonvulsants': ['Carbamazepine'],
    'Antihypertensive medications': ['Amlodipine', 'Metoprolol'],
    'Diuretics': ['Furosemide'],
    'Beta-blockers': ['Metoprolol'],
    'ACE inhibitors': ['Enalapril'],
    'Calcium channel blockers': ['Amlodipine'],
    'Antimalarial drugs': ['Chloroquine'],
    'Antiviral medications': ['Acyclovir'],
    'Antiretroviral drugs': ['Lamivudine'],
    'Bronchodilators': ['Salbutamol'],
    'Inhaled corticosteroids': ['Budesonide'],
    'Corticosteroids': ['Prednisone'],
    'NSAIDs': ['Ibuprofen'],
    'Proton Pump Inhibitors (PPIs)': ['Omeprazole'],
    'H2 Blockers': ['Ranitidine'],
    'Insulin': ['Insulin'],
    'Metformin': ['Metformin'],
    'Sulfonylureas': ['Glipizide'],
    'DPP-4 inhibitors': ['Sitagliptin'],
    'GLP-1 receptor agonists': ['Liraglutide'],
    'Analgesics': ['Paracetamol', 'Ibuprofen'],
    'Decongestants': ['Pseudoephedrine'],
    'Antacids': ['Ranitidine'],
    'IV fluids': [],
    'Thrombolytic drugs': ['Alteplase', 'Streptokinase'],
    'Blood thinners': ['Warfarin', 'Heparin'],
    'Clot-dissolving medications': ['Alteplase', 'Streptokinase'],
    'Medications for itching': ['Hydrocortisone', 'Calamine lotion'],
    'Topical treatments': ['Hydrocortisone', 'Clotrimazole'],
    'Anticoagulants': ['Warfarin', 'Heparin'],
    'Antifungal drugs': ['Clotrimazole', 'Ketoconazole'],
    'Antihypertensive': ['Amlodipine', 'Metoprolol'],
    'Antiviral': ['Acyclovir'],
    'Antidiarrheal': ['Loperamide'],
    'Antipyretic': ['Paracetamol', 'Ibuprofen'],
    'Antiemetic': ['Ondansetron'],
    'Pain relievers': ['Paracetamol', 'Ibuprofen'],
    'Topical treatments': ['Hydrocortisone', 'Clotrimazole'],
}

SYMPTOM_TO_MEDICINE = defaultdict(list)

GENERIC_MEDICATION_KEYWORDS = {
    'drugs', 'medications', 'treatments', 'therapy', 'analgesics', 'antibiotics',
    'antihistamines', 'antipyretics', 'antidiarrheal', 'antiviral', 'antimalarial',
    'anticoagulants', 'beta-blockers', 'calcium channel blockers', 'ace inhibitors',
    'diuretics', 'antacids', 'antispasmodics', 'corticosteroids', 'topical treatments',
    'oral medications', 'antifungal', 'vaccination', 'antihypertensive', 'pain relievers',
    'clot-dissolving', 'thrombolytic', 'blood thinners'
}

def is_generic_medication_entry(name):
    if not name:
        return False
    text = str(name).lower()
    return any(keyword in text for keyword in GENERIC_MEDICATION_KEYWORDS)

# Normalize medicine names for lookup.
def normalize_medicine_name(name):
    if not isinstance(name, str):
        return ""
    cleaned = name.strip().lower()
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

# Lookup medicine information in the Medicine_Details spreadsheet.
def get_medicine_detail_from_excel(name):
    normalized = normalize_medicine_name(name)
    if not normalized:
        return None

    # Try exact matches on the product name first.
    exact_matches = medicine_details[medicine_details['Medicine Name'].str.lower().str.strip() == normalized]
    if not exact_matches.empty:
        row = exact_matches.iloc[0]
    else:
        # Try matching on composition and uses fields.
        has_name = medicine_details['Medicine Name'].str.lower().str.contains(re.escape(normalized), na=False)
        has_comp = medicine_details['Composition'].str.lower().str.contains(re.escape(normalized), na=False)
        has_uses = medicine_details['Uses'].str.lower().str.contains(re.escape(normalized), na=False)
        matches = medicine_details[has_name | has_comp | has_uses]
        if matches.empty:
            return None
        row = matches.iloc[0]

    return {
        'name': row.get('Medicine Name', name),
        'purpose': row.get('Uses', ''),
        'dosage': row.get('Composition', ''),
        'precautions': row.get('Side_effects', ''),
        'when_to_consult': 'Consult a qualified healthcare provider before prescribing.'
    }

#============================================================
# custome and helping functions
#==========================helper funtions================

def helper(dis):
    # Description (string)
    desc_series = description[description['Disease'] == dis]['Description']
    desc = desc_series.tolist()[0] if len(desc_series) > 0 else ""

    # Precautions (list of strings) - take first matching row and filter out NaNs
    pre_df = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    if not pre_df.empty:
        # take first matching row
        first_row = pre_df.iloc[0].tolist()
        pre = [p for p in first_row if pd.notna(p) and str(p).strip()]
    else:
        pre = []

    # Medications (list)
    med_series = medications[medications['Disease'] == dis]['Medication']
    med_names_raw = [m for m in med_series.tolist() if pd.notna(m) and str(m).strip()]
    # Normalize medication entries: they may be single names or stringified lists
    med_names = []
    
    # First, add specific medicines from Medicine_Details dataset
    disease_lower = dis.lower()
    if disease_lower in disease_to_medicines:
        med_names.extend(disease_to_medicines[disease_lower])
    
    # Then, add general categories from medications.csv if not already present
    import ast
    for item in med_names_raw:
        s = str(item).strip()
        if s.startswith('[') and s.endswith(']'):
            try:
                parsed = ast.literal_eval(s)
                for p in parsed:
                    if p and str(p).strip() and str(p).strip() not in med_names:
                        med_names.append(str(p).strip())
            except Exception:
                if s not in med_names:
                    med_names.append(s)
        elif ',' in s and not s.lower().startswith('http'):
            # comma-separated
            parts = [p.strip() for p in s.split(',') if p.strip()]
            for part in parts:
                if part not in med_names:
                    med_names.append(part)
        else:
            if s not in med_names:
                med_names.append(s)
    # Map medication names to information where available
    med = []
    generic_count = 0
    for name in med_names:
        info = MEDICINES_INFO.get(name)
        if info:
            med.append({
                'name': name,
                'purpose': info.get('purpose',''),
                'dosage': info.get('dosage',''),
                'precautions': info.get('precautions',''),
                'when_to_consult': info.get('when_to_consult','')
            })
            continue

        detail_info = get_medicine_detail_from_excel(name)
        if detail_info:
            med.append(detail_info)
            continue

        # If the dataset lists a general category, expand to example medicines
        examples = CATEGORY_TO_EXAMPLES.get(name, [])
        if examples:
            for ex in examples:
                ex_info = MEDICINES_INFO.get(ex)
                if ex_info:
                    med.append({
                        'name': ex,
                        'purpose': ex_info.get('purpose',''),
                        'dosage': ex_info.get('dosage',''),
                        'precautions': ex_info.get('precautions',''),
                        'when_to_consult': ex_info.get('when_to_consult','')
                    })
                else:
                    detail_info = get_medicine_detail_from_excel(ex)
                    if detail_info:
                        med.append(detail_info)
                    else:
                        med.append({
                            'name': ex,
                            'purpose': '',
                            'dosage': '',
                            'precautions': 'Follow prescriber or pharmacist advice.',
                            'when_to_consult': 'Consult a healthcare professional for specific guidance.'
                        })
            continue

        if is_generic_medication_entry(name):
            generic_count += 1
            continue

        med.append({
            'name': name,
            'purpose': '',
            'dosage': '',
            'precautions': 'Follow prescriber or pharmacist advice.',
            'when_to_consult': 'Consult a healthcare professional for specific guidance.'
        })
    if generic_count > 0 and not med:
        med = []

        detail_info = get_medicine_detail_from_excel(name)
        if detail_info:
            med.append(detail_info)
            

        # If the dataset lists a general category, expand to example medicines
        examples = CATEGORY_TO_EXAMPLES.get(name, [])
        if examples:
            for ex in examples:
                ex_info = MEDICINES_INFO.get(ex)
                if ex_info:
                    med.append({
                        'name': ex,
                        'purpose': ex_info.get('purpose',''),
                        'dosage': ex_info.get('dosage',''),
                        'precautions': ex_info.get('precautions',''),
                        'when_to_consult': ex_info.get('when_to_consult','')
                    })
                else:
                    detail_info = get_medicine_detail_from_excel(ex)
                    if detail_info:
                        med.append(detail_info)
                    else:
                        med.append({
                            'name': ex,
                            'purpose': '',
                            'dosage': '',
                            'precautions': 'Follow prescriber or pharmacist advice.',
                            'when_to_consult': 'Consult a healthcare professional for specific guidance.'
                        })
        else:
            med.append({
                'name': name,
                'purpose': '',
                'dosage': '',
                'precautions': 'Follow prescriber or pharmacist advice.',
                'when_to_consult': 'Consult a healthcare professional for specific guidance.'
            })

    # Diet recommendations (list)
    die_series = diets[diets['Disease'] == dis]['Diet']
    die = [d for d in die_series.tolist() if pd.notna(d) and str(d).strip()]

    # Workouts (list)
    wrkout_series = workout[workout['disease'] == dis]['workout']
    workout_list = [w for w in wrkout_series.tolist() if pd.notna(w) and str(w).strip()]

    return desc, pre, med, die, workout_list

symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

# Build disease to medicines mapping using string matching
#disease_names = list(diseases_list.values())
#for disease in disease_names:
#   disease_lower = disease.lower()
 #   matching_meds = []
  #  for _, row in medicine_details.iterrows():
   #     uses = row['Uses'].lower()
    #    if disease_lower in uses:
     #       med_name = row['Medicine Name']
      #      review = medicine_to_review.get(med_name, 0)
       #     matching_meds.append((med_name, review))
    # Sort by review descending and take top 5
 #   matching_meds.sort(key=lambda x: x[1], reverse=True)
  #  disease_to_medicines[disease_lower] = [m[0] for m in matching_meds[:5]]

# Build proper disease -> medicine mapping
disease_to_medicines = defaultdict(list)

for _, row in disease_medicine.iterrows():
    disease = str(row['Disease']).strip().lower()
    medicine = str(row['Medicine']).strip()

    disease_to_medicines[disease].append(medicine)


















# # Initialize the TextBlob object for spelling correction
def correct_spelling(symptom):
    # Correct the spelling of a single symptom
    blob = TextBlob(symptom)
    return str(blob.correct())

# Try a fuzzy match against known symptoms and synonym phrases.
def find_best_symptom_match(item, cutoff=0.75):
    item = str(item).strip().lower()
    if not item:
        return None

    best_match = None
    best_score = 0.0
    for key, synonyms in symptom_mapping.items():
        score = difflib.SequenceMatcher(None, item, key).ratio()
        if score > best_score:
            best_score = score
            best_match = key
        for syn in synonyms:
            score = difflib.SequenceMatcher(None, item, syn).ratio()
            if score > best_score:
                best_score = score
                best_match = key

    return best_match if best_score >= cutoff else None

# Normalize input symptom text to keys used by the model and mappings.
def normalize_symptom(symptom):
    item = str(symptom).strip().lower()
    item = item.replace('-', ' ').replace('_', ' ').replace('.', ' ').strip()
    item = ' '.join(item.split())
    if not item:
        return None
    if item in symptoms_dict:
        return item
    for key, synonyms in symptom_mapping.items():
        if item == key or item in synonyms:
            return key
    fuzzy = find_best_symptom_match(item)
    if fuzzy:
        return fuzzy
    return item

# Try to resolve symptom by normalization, mapping, or spelling correction.
def resolve_symptom(symptom):
    normalized = normalize_symptom(symptom)
    if not normalized:
        return None
    if normalized in symptoms_dict:
        return normalized
    for key, synonyms in symptom_mapping.items():
        if normalized == key or normalized in synonyms:
            return key
    corrected = normalize_symptom(correct_spelling(normalized))
    if corrected and corrected in symptoms_dict:
        return corrected
    for key, synonyms in symptom_mapping.items():
        if corrected == key or corrected in synonyms:
            return key
    return None

symptom_mapping = defaultdict(lambda: "unknown", {
    "itching": ["itching", "itchy", "itchy skin", "skin itching"],
    "skin_rash": ["skin rash", "rash", "dermatitis", "rashes", "erythema"],
    "nodal_skin_eruptions": ["nodal skin eruptions", "skin eruptions", "bumps"],
    "continuous_sneezing": ["continuous sneezing", "sneezing", "allergic sneezing"],
    "shivering": ["shivering", "trembling", "shaky"],
    "chills": ["chills", "cold sensation", "cold", "feeling cold"],
    "joint_pain": ["joint pain", "arthralgia", "aching joints", "joint ache", "pain in joints"],
    "stomach_pain": ["stomach pain", "abdominal pain", "belly ache", "stomach ache", "tummy pain"],
    "acidity": ["acidity", "heartburn", "acid reflux"],
    "ulcers_on_tongue": ["ulcers on tongue", "tongue ulcers", "mouth sores"],
    "muscle_wasting": ["muscle wasting", "muscle loss", "muscle atrophy"],
    "vomiting": ["vomiting", "emesis", "throwing up", "throwing up"],
    "burning_micturition": ["burning micturition", "burning urination", "painful urination"],
    "spotting_urination": ["spotting urination", "blood in urine", "hematuria"],
    "fatigue": ["fatigue", "tiredness", "exhaustion", "weariness", "low energy"],
    "weight_gain": ["weight gain", "increased weight"],
    "anxiety": ["anxiety", "nervousness", "worry"],
    "cold_hands_and_feets": ["cold hands and feet", "cold extremities"],
    "mood_swings": ["mood swings", "emotional changes"],
    "weight_loss": ["weight loss", "decreased weight"],
    "restlessness": ["restlessness", "agitation"],
    "lethargy": ["lethargy", "sluggishness"],
    "patches_in_throat": ["patches in throat", "throat patches", "throat lesions"],
    "irregular_sugar_level": ["irregular sugar level", "unstable glucose", "blood sugar fluctuations"],
    "cough": ["cough", "coughing", "dry cough", "wet cough", "coughing a lot"],
    "high_fever": ["high fever", "elevated temperature", "temperature"],
    "sunken_eyes": ["sunken eyes", "hollow eyes"],
    "breathlessness": ["breathlessness", "shortness of breath", "dyspnea", "difficulty breathing", "can\'t breathe"],
    "sweating": ["sweating", "perspiration"],
    "dehydration": ["dehydration", "fluid loss"],
    "indigestion": ["indigestion", "upset stomach"],
    "headache": ["headache", "head pain", "migraine"],
    "yellowish_skin": ["yellowish skin", "jaundice"],
    "dark_urine": ["dark urine"],
    "nausea": ["nausea", "queasiness"],
    "loss_of_appetite": ["loss of appetite", "no appetite", "anorexia"],
    "pain_behind_the_eyes": ["pain behind the eyes", "eye pain"],
    "back_pain": ["back pain", "lower back pain"],
    "constipation": ["constipation", "difficulty passing stool"],
    "abdominal_pain": ["abdominal pain", "belly pain", "stomach ache"],
    "diarrhoea": ["diarrhoea", "loose stools"],
    "mild_fever": ["mild fever", "fever", "low-grade fever"],
    "yellow_urine": ["yellow urine"],
    "yellowing_of_eyes": ["yellowing of eyes", "scleral icterus"],
    "acute_liver_failure": ["acute liver failure", "hepatic failure"],
    "fluid_overload": ["fluid overload", "edema"],
    "swelling_of_stomach": ["swelling of stomach", "abdominal bloating"],
    "swelled_lymph_nodes": ["swelled lymph nodes", "enlarged lymph nodes"],
    "malaise": ["malaise", "general discomfort"],
    "blurred_and_distorted_vision": ["blurred and distorted vision", "blurry vision"],
    "phlegm": ["phlegm", "mucus"],
    "throat_irritation": ["throat irritation", "sore throat", "throat pain"],
    "redness_of_eyes": ["redness of eyes", "bloodshot eyes", "red eyes"],
    "sinus_pressure": ["sinus pressure", "sinus congestion"],
    "runny_nose": ["runny nose", "rhinorrhea", "nasal drip"],
    "congestion": ["congestion", "nasal blockage", "stuffy nose", "blocked nose"],
    "chest_pain": ["chest pain", "angina"],
    "weakness_in_limbs": ["weakness in limbs", "limb weakness"],
    "fast_heart_rate": ["fast heart rate", "tachycardia"],
    "pain_during_bowel_movements": ["pain during bowel movements", "painful defecation"],
    "pain_in_anal_region": ["pain in anal region", "anal pain"],
    "bloody_stool": ["bloody stool", "rectal bleeding"],
    "irritation_in_anus": ["irritation in anus", "anal itching"],
    "neck_pain": ["neck pain", "cervical pain"],
    "dizziness": ["dizziness", "lightheadedness", "dizzy", "vertigo", "light headed"],
    "cramps": ["cramps", "muscle cramps", "spasms"],
    "bruising": ["bruising", "hematoma"],
    "obesity": ["obesity", "overweight"],
    "swollen_legs": ["swollen legs", "leg edema"],
    "swollen_blood_vessels": ["swollen blood vessels", "varicose veins"],
    "puffy_face_and_eyes": ["puffy face and eyes", "facial swelling"],
    "enlarged_thyroid": ["enlarged thyroid", "goiter"],
    "brittle_nails": ["brittle nails", "weak nails"],
    "swollen_extremeties": ["swollen extremities", "swollen arms and legs"],
    "excessive_hunger": ["excessive hunger", "polyphagia"],
    "extra_marital_contacts": ["extra marital contacts", "multiple sexual partners"],
    "drying_and_tingling_lips": ["drying and tingling lips", "lip dryness"],
    "slurred_speech": ["slurred speech", "dysarthria"],
    "knee_pain": ["knee pain", "pain in the knees"],
    "hip_joint_pain": ["hip joint pain", "hip pain"],
    "muscle_weakness": ["muscle weakness", "muscle fatigue"],
    "stiff_neck": ["stiff neck", "neck stiffness"],
    "swelling_joints": ["swelling joints", "joint swelling"],
    "movement_stiffness": ["movement stiffness", "rigidity"],
    "spinning_movements": ["spinning movements", "vertigo"],
    "loss_of_balance": ["loss of balance", "balance problems"],
    "unsteadiness": ["unsteadiness", "lack of balance"],
    "weakness_of_one_body_side": ["weakness of one body side", "hemiparesis"],
    "loss_of_smell": ["loss of smell", "anosmia"],
    "bladder_discomfort": ["bladder discomfort", "bladder pain"],
    "foul_smell_of_urine": ["foul smell of urine", "smelly urine"],
    "continuous_feel_of_urine": ["continuous feel of urine", "urgency to urinate"],
    "passage_of_gases": ["passage of gases", "flatulence"],
    "internal_itching": ["internal itching"],
    "toxic_look_(typhos)": ["toxic look (typhos)", "septic appearance"],
    "depression": ["depression", "low mood"],
    "irritability": ["irritability", "easily annoyed"],
    "muscle_pain": ["muscle pain", "myalgia", "body ache", "muscle ache", "body pain"],
    "altered_sensorium": ["altered sensorium", "confusion"],
    "red_spots_over_body": ["red spots over body", "rash with red spots"],
    "belly_pain": ["belly pain", "abdominal pain", "tummy pain"],
    "abnormal_menstruation": ["abnormal menstruation", "irregular periods"],
    "dischromic_patches": ["dischromic patches", "skin discoloration"],
    "watering_from_eyes": ["watering from eyes", "teary eyes"],
    "increased_appetite": ["increased appetite", "hyperphagia"],
    "polyuria": ["polyuria", "excessive urination"],
    "family_history": ["family history", "genetic predisposition"],
    "mucoid_sputum": ["mucoid sputum", "mucus in sputum"],
    "rusty_sputum": ["rusty sputum", "blood-tinged sputum"],
    "lack_of_concentration": ["lack of concentration", "difficulty focusing"],
    "visual_disturbances": ["visual disturbances", "vision problems"],
    "receiving_blood_transfusion": ["receiving blood transfusion"],
    "receiving_unsterile_injections": ["receiving unster"]
    # Add more symptoms and their synonyms here
})

# Model Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    
    for item in patient_symptoms:
        if not item:
            continue
        resolved = resolve_symptom(item)
        if resolved is None:
            continue
        index = symptoms_dict.get(resolved, -1)
        if index != -1:
            input_vector[index] = 1
    
    # First try the loaded model
    try:
        if hasattr(svc, 'feature_names_in_'):
            X_pred = pd.DataFrame([input_vector], columns=svc.feature_names_in_)
        else:
            X_pred = np.asarray([input_vector])
        predicted_value = svc.predict(X_pred)[0]
    except Exception:
        predicted_value = None

    predicted_disease = None
    if predicted_value is not None:
        if label_encoder is not None:
            try:
                predicted_disease = label_encoder.inverse_transform([predicted_value])[0]
            except Exception:
                predicted_disease = None
        if predicted_disease is None:
            if isinstance(predicted_value, (int, np.integer)):
                predicted_disease = diseases_list.get(int(predicted_value))
            else:
                predicted_disease = str(predicted_value)

    # If model failed or produced a low-confidence/default result, fall back to a simple rule-based predictor.
    use_fallback = False
    if predicted_disease is None or predicted_disease == "" or predicted_disease == "Unknown Disease":
        use_fallback = True
    elif hasattr(svc, '__class__') and svc.__class__.__name__ == 'DummyModel':
        use_fallback = True

    if use_fallback and disease_symptom_freq:
        # Map our symptoms_dict order to training symptom columns where possible
        # Build an input vector aligned with disease_symptom_cols
        try:
            train_vec = np.zeros(len(disease_symptom_cols))
            for i, col in enumerate(disease_symptom_cols):
                # column names in training use the same keys as symptoms_dict (mostly)
                if col in symptoms_dict:
                    idx = symptoms_dict[col]
                    if idx < len(input_vector):
                        train_vec[i] = input_vector[idx]
            # compute score per disease
            best_disease = None
            best_score = -1
            for disease, freq_vec in disease_symptom_freq.items():
                score = np.dot(freq_vec, train_vec)
                if score > best_score:
                    best_score = score
                    best_disease = disease
            if best_disease:
                return best_disease
        except Exception:
            pass

    # Final fallback: use model prediction if available
    if predicted_disease is not None:
        return predicted_disease

    return "Unknown Disease"


def get_medication_suggestions_for_symptoms(symptoms):
    suggestions = []
    seen = set()
    for item in symptoms:
        resolved = resolve_symptom(item)
        if not resolved:
            continue
        examples = SYMPTOM_TO_MEDICINE.get(resolved, [])
        for name in examples:
            if name in seen:
                continue
            seen.add(name)
            info = MEDICINES_INFO.get(name)
            if not info:
                info = get_medicine_detail_from_excel(name)
            if not info:
                info = {
                    'name': name,
                    'purpose': '',
                    'dosage': '',
                    'precautions': 'Follow prescriber or pharmacist advice.',
                    'when_to_consult': 'Consult a healthcare professional for specific guidance.'
                }
            suggestions.append({
                'name': info.get('name', name),
                'purpose': info.get('purpose', ''),
                'dosage': info.get('dosage', ''),
                'precautions': info.get('precautions', ''),
                'when_to_consult': info.get('when_to_consult', '')
            })
    return suggestions

# creating routes========================================


@app.route("/")
def index():
    return render_template("index.html")

# Define a route for the home page
@app.route('/predict', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        name = request.form.get('name')
        age = request.form.get('age')
        location = request.form.get('location')
        symptoms = request.form.get('symptoms')
        # mysysms = request.form.get('mysysms')
        # print(mysysms)
        print(symptoms)
        if symptoms =="Symptoms":
            message = "Please either write symptoms or you have written misspelled symptoms"
            return render_template('index.html', message=message)
        
        if not symptoms:
            message = "Please enter symptoms."
            return render_template('index.html', message=message) 
        
        else:

            # Split the user's input into a list of symptoms (assuming they are comma-separated)
            user_symptoms = [s.strip() for s in symptoms.split(',')]
            # Remove any extra characters, if any
            user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]
            symptom_based_meds = get_medication_suggestions_for_symptoms(user_symptoms)
            predicted_disease = get_predicted_value(user_symptoms)
            dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)

            # If the model fails to find a disease-level medication list, use symptom-level fallback suggestions.
            if not medications:
                medications = symptom_based_meds

            # precautions, medications, diets and workout are lists (may be empty)
            my_precautions = precautions  # already a list

            return render_template(
                'index.html',
                name=name,
                age=age,
                location=location,
                symptoms=symptoms,
                predicted_disease=predicted_disease,
                dis_des=dis_des,
                my_precautions=my_precautions,
                medications=medications,
                my_diet=rec_diet,
                workout=workout
            )

    return render_template('index.html')



# about view funtion and path
@app.route('/about')
def about():
    return render_template("about.html")
# contact view funtion and path
@app.route('/contact')
def contact():
    return render_template("contact.html")

# developer view funtion and path
@app.route('/developer')
def developer():
    return render_template("developer.html")

# about view funtion and path
@app.route('/blog')
def blog():
    return render_template("blog.html")

# about view funtion and path
@app.route('/upload',  methods=['GET', 'POST'])
def upload():
    return render_template("upload.html")

if __name__ == '__main__':

    app.run(debug=True)