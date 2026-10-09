from flask import Flask, render_template, request, jsonify
import pickle, json, os
import numpy as np

app = Flask(__name__)

# ── Load model & data ─────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)

with open(os.path.join(BASE, 'model/rf_model.pkl'), 'rb')      as f: model       = pickle.load(f)
with open(os.path.join(BASE, 'model/symptoms_list.pkl'), 'rb') as f: SYMPTOMS    = pickle.load(f)
with open(os.path.join(BASE, 'model/classes.pkl'), 'rb')       as f: CLASSES     = pickle.load(f)

with open(os.path.join(BASE, 'data/description.json'))  as f: DESCRIPTION = json.load(f)
with open(os.path.join(BASE, 'data/precaution.json'))   as f: PRECAUTION  = json.load(f)
with open(os.path.join(BASE, 'data/severity.json'))     as f: SEVERITY    = json.load(f)

# ── Diet & workout mappings ───────────────────────────────────────────────────
DIET = {
    "Fungal infection":       ["Avoid sugar/refined carbs","Probiotic foods (curd, yogurt)","Garlic and ginger","Stay hydrated","Antifungal spices (turmeric)"],
    "Allergy":                ["Anti-inflammatory foods","Vitamin C rich fruits","Omega-3 fatty acids","Probiotics","Avoid known allergens"],
    "GERD":                   ["Oatmeal & whole grains","Ginger tea","Alkaline foods","Green vegetables","Avoid tomatoes & citrus"],
    "Chronic cholestasis":    ["Low-fat diet","High-fiber foods","Vitamin supplements","Fresh fruits","Avoid fried foods"],
    "Drug Reaction":          ["Bland easy-to-digest foods","Stay hydrated","Fresh fruits & vegetables","Avoid alcohol","Light soups"],
    "Peptic ulcer diseae":    ["Cabbage juice","Probiotic foods","Honey & licorice tea","Soft cooked foods","Avoid spicy/acidic foods"],
    "AIDS":                   ["High protein diet","Fresh fruits & vegetables","Avoid raw/undercooked food","Calorie-dense foods","Stay hydrated"],
    "Diabetes ":              ["Low glycemic index foods","Whole grains","Leafy green vegetables","Avoid refined sugar","Legumes & lentils"],
    "Gastroenteritis":        ["BRAT diet (Banana, Rice, Applesauce, Toast)","Clear broths","Coconut water","Avoid dairy & spicy food","Rice water"],
    "Bronchial Asthma":       ["Anti-inflammatory foods","Vitamin D rich foods","Magnesium-rich foods","Avoid sulfites","Ginger tea"],
    "Hypertension ":          ["DASH diet","Potassium-rich foods (banana, spinach)","Reduce salt (<5g/day)","Avoid processed foods","Dark chocolate (moderate)"],
    "Migraine":               ["Magnesium-rich foods","Omega-3 fatty acids","Stay well hydrated","Avoid caffeine, alcohol, aged cheese","Ginger tea"],
    "Cervical spondylosis":   ["Anti-inflammatory foods","Calcium-rich foods","Vitamin D foods","Omega-3 fatty acids","Turmeric milk"],
    "Paralysis (brain hemorrhage)": ["Low sodium diet","Anti-inflammatory foods","High antioxidant foods","Avoid alcohol","Soft easily swallowable foods"],
    "Jaundice":               ["High carbohydrate low fat diet","Sugarcane juice","Lemon water","Papaya","Avoid oily & spicy food"],
    "Malaria":                ["High fluid intake","Light easily digestible food","Warm soups","Fresh fruit juices","Avoid heavy meals"],
    "Chicken pox":            ["Soft & bland foods","Cold foods to soothe mouth sores","Plenty of fluids","Avoid salty & spicy foods","Ice cream/cold yogurt"],
    "Dengue":                 ["Papaya leaf juice (boosts platelets)","Coconut water","Orange juice","Pomegranate juice","High fluid intake"],
    "Typhoid":                ["Soft easily digestible foods","High calorie liquid diet","Boiled vegetables","Ripe bananas","Avoid raw fruits"],
    "hepatitis A":            ["High carbohydrate low fat diet","Fruits & vegetables","Plenty of fluids","Avoid alcohol","Sugarcane juice"],
    "Hepatitis B":            ["Low fat diet","High fiber foods","Avoid alcohol","Antioxidant-rich foods","Boiled vegetables"],
    "Hepatitis C":            ["Low sodium diet","High fiber diet","Avoid alcohol & fatty foods","Fruits & vegetables","Green tea"],
    "Hepatitis D":            ["Low fat high protein diet","Avoid alcohol","Stay hydrated","Antioxidant foods","Fresh fruits"],
    "Hepatitis E":            ["Light diet","Plenty of fluids","Avoid alcohol completely","Fresh fruits","Boiled foods"],
    "Alcoholic hepatitis":    ["High protein diet","Vitamin supplements (B1, B12)","Avoid alcohol strictly","Small frequent nutritious meals","Soft cooked foods"],
    "Tuberculosis":           ["High protein & calorie diet","Vitamin-rich foods","Avoid alcohol","Stay well hydrated","Egg & lean meat"],
    "Common Cold":            ["Warm soups & broths","Ginger-honey tea","Citrus fruits (Vitamin C)","Garlic","Turmeric milk"],
    "Pneumonia":              ["High protein diet","Plenty of fluids","Warm soups","Vitamin C rich foods","Avoid cold beverages"],
    "Dimorphic hemmorhoids(piles)": ["High fiber diet (whole grains, fruits)","Plenty of water","Avoid spicy food","Prunes & psyllium husk","Papaya"],
    "Heart attack":           ["Post-recovery: Low sodium low saturated fat diet","Mediterranean diet","Omega-3 rich fish","Avoid processed foods","Fresh fruits & vegetables"],
    "Varicose veins":         ["High fiber diet","Anti-inflammatory foods","Vitamin C & E rich foods","Reduce sodium intake","Stay hydrated"],
    "Hypothyroidism":         ["Iodine-rich foods (seafood, dairy)","Selenium-rich foods (Brazil nuts)","Avoid excess soy","Adequate protein intake","Eggs & lean meat"],
    "Hyperthyroidism":        ["Calcium & Vitamin D foods","Anti-inflammatory foods","Avoid iodine-rich foods","Cruciferous vegetables (broccoli, cabbage)","Non-caffeinated drinks"],
    "Hypoglycemia":           ["Regular meals & snacks","Complex carbohydrates","Protein with each meal","Avoid alcohol on empty stomach","Whole grain bread"],
    "Osteoarthristis":        ["Anti-inflammatory Mediterranean diet","Omega-3 rich foods","Vitamin D & calcium","Avoid processed & sugary foods","Cherries & berries"],
    "Arthritis":              ["Anti-inflammatory foods","Turmeric & ginger","Omega-3 fatty acids","Vitamin D rich foods","Green tea"],
    "(vertigo) Paroymsal  Positional Vertigo": ["Low sodium diet","Adequate hydration","Avoid caffeine & alcohol","Small frequent meals","Ginger tea"],
    "Acne":                   ["Low glycemic index diet","Zinc-rich foods","Omega-3 fatty acids","Avoid dairy & high-sugar foods","Green vegetables"],
    "Urinary tract infection":["Drink 8+ glasses of water daily","Cranberry juice","Probiotic foods","Avoid caffeine & alcohol","Fresh vegetables"],
    "Psoriasis":              ["Anti-inflammatory diet","Omega-3 rich foods","Vitamin D foods","Avoid alcohol & processed foods","Turmeric"],
    "Impetigo":               ["Nutritious balanced diet","Vitamin C for immune support","Stay hydrated","Protein-rich foods for healing","Fresh fruits"],
}

WORKOUT = {
    "Fungal infection":       ["Keep skin dry & clean","Wear loose cotton clothes","Light yoga","Avoid gym/pool until healed","Short walks"],
    "Allergy":                ["Indoor exercises","Yoga & breathing exercises","Avoid outdoor activity during high pollen","Swimming (indoor pool)","Tai Chi"],
    "GERD":                   ["Light walks after meals","Elevate head while sleeping","Avoid tight clothing","Gentle yoga","No exercise immediately after eating"],
    "Chronic cholestasis":    ["Light exercise","Avoid strenuous activity","Gentle walks","Yoga","Regular monitoring"],
    "Drug Reaction":          ["Rest","Avoid sun exposure","Monitor symptoms closely","Light stretching","Short walks when stable"],
    "Peptic ulcer diseae":    ["Stress management exercises","Regular gentle walks","Adequate sleep routine","Meditation","Light yoga"],
    "AIDS":                   ["Moderate exercise","Yoga & meditation","Adequate rest","Light resistance training","Walking"],
    "Diabetes ":              ["30 min brisk walk daily","Cycling","Resistance training","Yoga for stress management","Swimming"],
    "Gastroenteritis":        ["Complete rest","Stay hydrated","Avoid strenuous activity","Short gentle walks when recovering","Light stretching"],
    "Bronchial Asthma":       ["Swimming (best for asthma)","Yoga & breathing exercises","Light cardio in clean air","Avoid outdoor exercise in pollution","Pursed-lip breathing"],
    "Hypertension ":          ["Brisk walking 30 min/day","Swimming","Tai Chi","Deep breathing & meditation","Cycling"],
    "Migraine":               ["Yoga & meditation","Regular sleep routine","Avoid strenuous exercise during attack","Gentle stretching","Short walks"],
    "Cervical spondylosis":   ["Neck stretches","Physiotherapy exercises","Swimming","Avoid prolonged screen time","Gentle yoga"],
    "Paralysis (brain hemorrhage)": ["Physiotherapy after stabilization","Occupational therapy","Speech therapy if needed","Passive range-of-motion","Breathing exercises"],
    "Jaundice":               ["Complete bed rest","Light walks only","Avoid strenuous activities","Gentle stretching","Deep breathing"],
    "Malaria":                ["Complete bed rest during fever","Light activity after recovery","Stay indoors during peak mosquito hours","Short walks post-recovery","Gentle stretching"],
    "Chicken pox":            ["Complete rest & isolation","Oatmeal baths for itching relief","Avoid school/work until blisters crust","Light stretching at home","Deep breathing"],
    "Dengue":                 ["Complete bed rest","Monitor platelet count daily","Avoid aspirin & ibuprofen","Short walks after recovery","Light stretching"],
    "Typhoid":                ["Complete bed rest during fever","Light activity after recovery","Maintain hygiene strictly","Short walks post-recovery","Gentle stretching"],
    "hepatitis A":            ["Rest during acute phase","Light walks during recovery","Avoid strenuous exercise","Gentle yoga post-recovery","Deep breathing"],
    "Hepatitis B":            ["Moderate exercise","Avoid contact sports","Regular medical checkups","Light walking","Yoga"],
    "Hepatitis C":            ["Moderate exercise","Yoga","Avoid excessive fatigue","Light resistance training","Walking"],
    "Hepatitis D":            ["Rest during acute phase","Light exercise during recovery","Gentle walks","Yoga","Deep breathing"],
    "Hepatitis E":            ["Complete bed rest","Avoid strenuous exercise until recovery","Light walks when feeling better","Gentle stretching","Deep breathing"],
    "Alcoholic hepatitis":    ["Complete rest","Nutritional rehabilitation","Counseling & support groups","Light walks after stabilization","Gentle stretching"],
    "Tuberculosis":           ["Rest during intensive phase","Light exercise during continuation phase","Deep breathing exercises","Short walks","Gentle yoga"],
    "Common Cold":            ["Rest at home","Steam inhalation","Gargling with warm salt water","Light stretching","Short indoor walks"],
    "Pneumonia":              ["Complete rest","Deep breathing exercises","Avoid strenuous activity","Sit upright to breathe easier","Short walks after recovery"],
    "Dimorphic hemmorhoids(piles)": ["Sitz baths 2-3x daily","Avoid prolonged sitting","Light walking","Avoid heavy lifting","Kegel exercises"],
    "Heart attack":           ["Cardiac rehabilitation program","Gradual return to activity under supervision","Light walking","Supervised cycling","Breathing exercises"],
    "Varicose veins":         ["Walking & cycling","Leg elevation exercises","Swimming","Avoid high-impact exercises","Compression stocking walks"],
    "Hypothyroidism":         ["Regular aerobic exercise","Yoga","Strength training","Adequate sleep","Swimming"],
    "Hyperthyroidism":        ["Low-impact exercise","Yoga & relaxation","Avoid overexertion","Adequate rest","Tai Chi"],
    "Hypoglycemia":           ["Regular meal timing around exercise","Monitor blood sugar before & after exercise","Carry snacks during activity","Light walking","Gentle yoga"],
    "Osteoarthristis":        ["Swimming & water aerobics","Cycling","Tai Chi","Physiotherapy exercises","Range-of-motion exercises"],
    "Arthritis":              ["Range-of-motion exercises","Strength training","Swimming","Hot/cold therapy","Tai Chi"],
    "(vertigo) Paroymsal  Positional Vertigo": ["Epley maneuver exercises","Balance training","Avoid sudden head movements","Vestibular rehabilitation","Short gentle walks"],
    "Acne":                   ["Regular face washing","Aerobic exercise (reduces stress)","Adequate sleep","Stress management yoga","Outdoor walks"],
    "Urinary tract infection":["Stay well hydrated","Maintain hygiene","Light walking","Avoid tight clothing during exercise","Gentle yoga"],
    "Psoriasis":              ["Gentle exercise","Stress management (yoga, meditation)","Moderate sunbathing","Moisturizing routine","Swimming"],
    "Impetigo":               ["Keep affected area clean","Avoid school until non-contagious","Light home exercises","Change bedding frequently","Personal hygiene routine"],
}

MEDICINES = {
    "Fungal infection":       [{"name":"Clotrimazole cream","dose":"Apply twice daily","side":"Mild skin irritation"},{"name":"Fluconazole 150mg","dose":"Single dose orally","side":"Nausea, headache"}],
    "Allergy":                [{"name":"Cetirizine 10mg","dose":"Once daily","side":"Drowsiness, dry mouth"},{"name":"Levocetirizine 5mg","dose":"Once daily at night","side":"Fatigue"}],
    "GERD":                   [{"name":"Omeprazole 20mg","dose":"Once daily before meals","side":"Headache, diarrhea"},{"name":"Pantoprazole 40mg","dose":"Once daily","side":"Nausea, abdominal pain"}],
    "Chronic cholestasis":    [{"name":"Ursodeoxycholic acid","dose":"8-10mg/kg/day","side":"Diarrhea"},{"name":"Cholestyramine 4g","dose":"Twice daily","side":"Constipation, bloating"}],
    "Drug Reaction":          [{"name":"Antihistamines","dose":"As prescribed","side":"Drowsiness"},{"name":"Corticosteroids","dose":"As prescribed","side":"Weight gain, mood changes"}],
    "Peptic ulcer diseae":    [{"name":"Omeprazole 40mg","dose":"Twice daily","side":"Headache, diarrhea"},{"name":"Amoxicillin 1g","dose":"Twice daily if H.pylori","side":"Nausea, diarrhea"}],
    "AIDS":                   [{"name":"Antiretroviral therapy (ART)","dose":"As prescribed daily","side":"Nausea, fatigue"},{"name":"Prophylactic antibiotics","dose":"As prescribed","side":"Various"}],
    "Diabetes ":              [{"name":"Metformin 500mg","dose":"Twice daily with meals","side":"Nausea, stomach upset"},{"name":"Glipizide 5mg","dose":"Once daily before breakfast","side":"Hypoglycemia risk"}],
    "Gastroenteritis":        [{"name":"ORS (Oral Rehydration Salts)","dose":"After every loose stool","side":"None"},{"name":"Metronidazole 400mg","dose":"Thrice daily if bacterial","side":"Nausea, metallic taste"}],
    "Bronchial Asthma":       [{"name":"Salbutamol inhaler","dose":"2 puffs as needed","side":"Tremors, rapid heartbeat"},{"name":"Budesonide inhaler","dose":"2 puffs twice daily","side":"Oral thrush"}],
    "Hypertension ":          [{"name":"Amlodipine 5mg","dose":"Once daily","side":"Ankle swelling, flushing"},{"name":"Losartan 50mg","dose":"Once daily","side":"Dizziness, fatigue"}],
    "Migraine":               [{"name":"Sumatriptan 50mg","dose":"At onset of migraine","side":"Tingling, dizziness"},{"name":"Ibuprofen 400mg","dose":"Every 6-8 hrs during attack","side":"Gastric irritation"}],
    "Cervical spondylosis":   [{"name":"Ibuprofen 400mg","dose":"Thrice daily after meals","side":"Gastric irritation"},{"name":"Muscle relaxants","dose":"As prescribed","side":"Drowsiness"}],
    "Paralysis (brain hemorrhage)":[{"name":"Emergency medical care required","dose":"Call 108 immediately","side":"N/A"}],
    "Jaundice":               [{"name":"Silymarin (Liv 52)","dose":"As prescribed","side":"Mild GI upset"},{"name":"Vitamins B & C supplements","dose":"Daily","side":"Generally safe"}],
    "Malaria":                [{"name":"Artemether-Lumefantrine","dose":"As prescribed by doctor","side":"Dizziness, nausea"},{"name":"Chloroquine 500mg","dose":"As prescribed","side":"Nausea, headache"}],
    "Chicken pox":            [{"name":"Acyclovir 800mg","dose":"5 times/day for 7 days","side":"Nausea, headache"},{"name":"Calamine lotion","dose":"Apply on rash as needed","side":"Minimal"}],
    "Dengue":                 [{"name":"Paracetamol 500mg","dose":"Every 6 hrs for fever","side":"Liver issues if overdosed"},{"name":"ORS for hydration","dose":"Regularly throughout day","side":"None"}],
    "Typhoid":                [{"name":"Azithromycin 500mg","dose":"Once daily for 5-7 days","side":"Nausea, diarrhea"},{"name":"Ciprofloxacin 500mg","dose":"Twice daily for 7-10 days","side":"Dizziness, nausea"}],
    "hepatitis A":            [{"name":"Supportive treatment only","dose":"Rest and hydration","side":"N/A"},{"name":"Vitamin supplements","dose":"As prescribed","side":"Generally safe"}],
    "Hepatitis B":            [{"name":"Tenofovir 300mg","dose":"Once daily","side":"Kidney issues with long use"},{"name":"Entecavir 0.5mg","dose":"Once daily","side":"Headache, fatigue"}],
    "Hepatitis C":            [{"name":"Sofosbuvir + Daclatasvir","dose":"As prescribed (12 weeks)","side":"Fatigue, headache"},{"name":"Ribavirin","dose":"As prescribed","side":"Anemia, fatigue"}],
    "Hepatitis D":            [{"name":"Pegylated Interferon alfa","dose":"As prescribed","side":"Flu-like symptoms, depression"}],
    "Hepatitis E":            [{"name":"Supportive treatment only","dose":"Rest and hydration","side":"N/A"},{"name":"Ribavirin (severe cases)","dose":"As prescribed","side":"Anemia"}],
    "Alcoholic hepatitis":    [{"name":"Prednisolone 40mg","dose":"Once daily for 28 days","side":"Weight gain, mood changes"},{"name":"Pentoxifylline 400mg","dose":"Three times daily","side":"Nausea, dizziness"}],
    "Tuberculosis":           [{"name":"HRZE regimen (Isoniazid + Rifampicin + Pyrazinamide + Ethambutol)","dose":"Daily for 6 months as prescribed","side":"Liver toxicity, vision changes"},{"name":"Pyridoxine (Vit B6) 10mg","dose":"Daily with INH","side":"None"}],
    "Common Cold":            [{"name":"Paracetamol 500mg","dose":"Every 6-8 hrs for fever","side":"Generally safe"},{"name":"Cetirizine 10mg","dose":"Once daily","side":"Mild drowsiness"}],
    "Pneumonia":              [{"name":"Amoxicillin 500mg","dose":"Thrice daily for 7 days","side":"Nausea, diarrhea"},{"name":"Azithromycin 500mg","dose":"Once daily for 5 days","side":"Stomach upset"}],
    "Dimorphic hemmorhoids(piles)":[{"name":"Lactulose syrup","dose":"15-30ml twice daily","side":"Bloating, diarrhea"},{"name":"Lignocaine ointment","dose":"Apply topically as needed","side":"Local irritation"}],
    "Heart attack":           [{"name":"🚨 CALL 108 IMMEDIATELY","dose":"Do not delay — emergency","side":"N/A"},{"name":"Aspirin 325mg","dose":"Chew immediately while waiting for ambulance","side":"Stomach upset"}],
    "Varicose veins":         [{"name":"Diosmin 450mg + Hesperidin 50mg","dose":"Once daily","side":"GI discomfort"},{"name":"Compression stockings","dose":"Wear daily","side":"Skin irritation if worn improperly"}],
    "Hypothyroidism":         [{"name":"Levothyroxine (T4)","dose":"As prescribed, empty stomach","side":"Palpitations if overdosed"},{"name":"Selenium supplements 200mcg","dose":"Daily","side":"Nausea if overdosed"}],
    "Hyperthyroidism":        [{"name":"Carbimazole 10mg","dose":"Thrice daily initially","side":"Agranulocytosis (rare)"},{"name":"Propranolol 20mg","dose":"Twice daily for symptoms","side":"Fatigue, cold extremities"}],
    "Hypoglycemia":           [{"name":"Glucose tablets/gel","dose":"15g at onset of symptoms","side":"None"},{"name":"Glucagon injection (severe)","dose":"1mg IM emergency","side":"Nausea, vomiting"}],
    "Osteoarthristis":        [{"name":"Paracetamol 1g","dose":"Up to 4 times daily","side":"Liver issues if overdosed"},{"name":"Diclofenac gel","dose":"Apply to joint 3-4x daily","side":"Local skin irritation"}],
    "Arthritis":              [{"name":"Ibuprofen 400mg","dose":"Thrice daily after meals","side":"Stomach upset"},{"name":"Methotrexate","dose":"As prescribed weekly","side":"Nausea, liver issues"}],
    "(vertigo) Paroymsal  Positional Vertigo":[{"name":"Betahistine 16mg","dose":"Thrice daily","side":"Nausea, headache"},{"name":"Cinnarizine 25mg","dose":"Thrice daily","side":"Drowsiness, weight gain"}],
    "Acne":                   [{"name":"Benzoyl peroxide 2.5% gel","dose":"Apply once daily","side":"Dryness, redness"},{"name":"Clindamycin 1% gel","dose":"Apply twice daily","side":"Mild irritation"}],
    "Urinary tract infection":[{"name":"Nitrofurantoin 100mg","dose":"Twice daily for 5-7 days","side":"Nausea, headache"},{"name":"Trimethoprim 200mg","dose":"Twice daily for 3-7 days","side":"Rash, nausea"}],
    "Psoriasis":              [{"name":"Betamethasone cream","dose":"Apply once daily on affected area","side":"Skin thinning with long use"},{"name":"Salicylic acid shampoo","dose":"Use as directed","side":"Mild scalp irritation"}],
    "Impetigo":               [{"name":"Mupirocin 2% ointment","dose":"Apply 3x daily for 5-10 days","side":"Mild burning/stinging"},{"name":"Flucloxacillin 250mg","dose":"4x daily for 7 days","side":"Nausea, diarrhea"}],
}

HOSPITALS = [
    {"name":"Aruna Hospital","type":"General Hospital","rating":"⭐ 4.2","address":"Dr Radhakrishnan Rd, Tumkur","phone":"08162276408","open":"Open 24 hours"},
    {"name":"Kasturba Hospital","type":"Private Hospital","rating":"⭐ 4.8","address":"SS Puram, Tumkur","phone":"08164021011","open":"Open 24 hours"},
    {"name":"THS Super Speciality Hospital","type":"Super Speciality","rating":"⭐ 3.0","address":"Bangalore–Honnavar Hwy","phone":"09900900080","open":"Open 24 hours"},
    {"name":"District Hospital Tumakuru","type":"Government Hospital","rating":"⭐ 2.9","address":"B4Q2+X2R, Tumkur","phone":"8000108104","open":"Closes 1pm"},
    {"name":"Bharathi Hospital","type":"Multi Speciality","rating":"⭐ 4.1","address":"NH-4, Tumkur","phone":"08162234567","open":"Open 24 hours"},
    {"name":"B Siddharamanna Hospital","type":"General","rating":"⭐ 3.8","address":"Main Road, Tumkur","phone":"08162215000","open":"Mon–Sat 8am–8pm"},
]

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    symptoms_display = [s.replace('_', ' ').title() for s in SYMPTOMS]
    return render_template('index.html', symptoms=symptoms_display, symptoms_raw=SYMPTOMS)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    selected = data.get('symptoms', [])

    if not selected:
        return jsonify({'error': 'No symptoms provided'}), 400

    # Build input vector
    input_vec = np.zeros(len(SYMPTOMS))
    severity_score = 0
    matched = []

    for sym in selected:
        sym_clean = sym.strip().lower().replace(' ', '_')
        if sym_clean in SYMPTOMS:
            idx = SYMPTOMS.index(sym_clean)
            input_vec[idx] = 1
            matched.append(sym_clean)
            severity_score += SEVERITY.get(sym_clean, 0)

    if not matched:
        return jsonify({'error': 'No valid symptoms matched'}), 400

    # Predict
    pred = model.predict([input_vec])[0]
    proba = model.predict_proba([input_vec])[0]
    confidence = round(float(max(proba)) * 100, 1)

    # Top 3 predictions
    top3_idx = np.argsort(proba)[::-1][:3]
    top3 = [{"disease": CLASSES[i], "confidence": round(float(proba[i])*100,1)} for i in top3_idx if proba[i] > 0]

    result = {
        "disease":      pred,
        "confidence":   confidence,
        "top3":         top3,
        "description":  DESCRIPTION.get(pred, "Description not available."),
        "precautions":  PRECAUTION.get(pred, []),
        "medicines":    MEDICINES.get(pred, [{"name":"Consult a doctor","dose":"Professional advice needed","side":"N/A"}]),
        "diet":         DIET.get(pred, ["Eat balanced nutritious meals","Stay hydrated","Avoid junk food"]),
        "workout":      WORKOUT.get(pred, ["Light walking","Rest adequately","Gentle stretching"]),
        "severity":     int(severity_score),
        "matched_symptoms": [s.replace('_', ' ') for s in matched],
    }
    return jsonify(result)

@app.route('/hospitals')
def hospitals():
    return jsonify(HOSPITALS)

@app.route('/symptoms')
def get_symptoms():
    symptoms_display = [s.replace('_', ' ') for s in SYMPTOMS]
    return jsonify(symptoms_display)

if __name__ == '__main__':
    app.run(debug=False, port=5000)
