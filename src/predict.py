import os
import pickle
import gdown

MODEL_PATH = "model/random_forest_model.pkl"

if not os.path.exists(MODEL_PATH):
    os.makedirs("model", exist_ok=True)

    url = "https://drive.google.com/file/d/1yUHf5QEQAUK-VsVrMFg_koiO0JFlTVWp/view?usp=drive_link"
    gdown.download(url, MODEL_PATH, fuzzy=True)

model = pickle.load(open(MODEL_PATH, "rb"))