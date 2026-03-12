import pickle
import pandas as pd
import os

# get project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# model paths
model_path = os.path.join(BASE_DIR, "model", "random_forest_model.pkl")
columns_path = os.path.join(BASE_DIR, "model", "model_columns.pkl")

# load model and columns
model = pickle.load(open(model_path, "rb"))
model_columns = pickle.load(open(columns_path, "rb"))

def predict_loan(data):

    # convert input to dataframe
    df = pd.DataFrame([data])

    # encode categorical variables if present
    df = pd.get_dummies(df)

    # align with training columns
    df = df.reindex(columns=model_columns, fill_value=0)

    # prediction
    prediction = model.predict(df)[0]

    # probability
    probability = model.predict_proba(df)[0][1]

    return prediction, probability