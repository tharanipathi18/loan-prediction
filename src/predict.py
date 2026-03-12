import os
import pickle
import pandas as pd
import gdown

MODEL_PATH = "model/random_forest_model.pkl"

# download model if not present
if not os.path.exists(MODEL_PATH):
    os.makedirs("model", exist_ok=True)

    url = "https://drive.google.com/file/d/1yUHf5QEQAUK-VsVrMFg_koiO0JFlTVWp/view?usp=drive_link"
    gdown.download(url, MODEL_PATH, quiet=False)

# load model
model = pickle.load(open(MODEL_PATH, "rb"))

# load columns
columns_path = "model/model_columns.pkl"
model_columns = pickle.load(open(columns_path, "rb"))