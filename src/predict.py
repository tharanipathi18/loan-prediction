import os
import pickle
import pandas as pd
import gdown

MODEL_PATH = "model/random_forest_model.pkl"
COLUMNS_PATH = "model/model_columns.pkl"

# download model if missing
if not os.path.exists(MODEL_PATH):
    os.makedirs("model", exist_ok=True)
    url = "https://drive.google.com/uc?id=1yUHf5QEQAUK-VsVrMFg_koiO0JFlTVWp"
    gdown.download(url, MODEL_PATH, fuzzy=True)

model = pickle.load(open(MODEL_PATH, "rb"))

# load training columns safely
if os.path.exists(COLUMNS_PATH):
    model_columns = pickle.load(open(COLUMNS_PATH, "rb"))
else:
    model_columns = list(model.feature_names_in_)

def predict_loan(data):

    df = pd.DataFrame([data])

    # align features
    df = df.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return prediction, probability