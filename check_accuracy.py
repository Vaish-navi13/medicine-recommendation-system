import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load raw dataset
df = pd.read_csv('data/dataset.csv')

# Clean symptom values: strip, lowercase, replace spaces with underscore
def clean(x):
    if isinstance(x, str):
        return x.strip().lower().replace(' ', '_')
    return x

for col in df.columns:
    if col.startswith('Symptom_'):
        df[col] = df[col].map(clean)

# Get symptom columns
symptom_cols = [c for c in df.columns if c.startswith('Symptom_')]

# Build symptom list
all_symptoms = set()
for col in symptom_cols:
    all_symptoms.update(df[col].dropna().unique())

# Build binary matrix
rows = []
for _, row in df.iterrows():
    vec = {s: 0 for s in all_symptoms}
    for col in symptom_cols:
        val = row[col]
        if isinstance(val, str) and val in vec:
            vec[val] = 1
    vec['prognosis'] = row['Disease']
    rows.append(vec)

data = pd.DataFrame(rows)

# Load the exact symptom list your model was trained on
with open('model/symptoms_list.pkl', 'rb') as f:
    MODEL_SYMPTOMS = pickle.load(f)

# Align columns to match the model
X = pd.DataFrame(0, index=data.index, columns=MODEL_SYMPTOMS)
for col in MODEL_SYMPTOMS:
    if col in data.columns:
        X[col] = data[col]

y = data['prognosis']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Load model
with open('model/rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Score
preds = model.predict(X_test)
print("ACCURACY:", round(accuracy_score(y_test, preds) * 100, 2), "%")