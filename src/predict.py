import os
import pickle
import pandas as pd
import gdown

MODEL_PATH = "model/random_forest_model.pkl"

if not os.path.exists(MODEL_PATH):
    os.makedirs("model", exist_ok=True)
    url = "https://drive.google.com/uc?id=1yUHf5QEQAUK-VsVrMFg_koiO0JFlTVWp"
    gdown.download(url, MODEL_PATH, fuzzy=True)

model = pickle.load(open(MODEL_PATH, "rb"))

def predict_loan(data):

    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return prediction, probability