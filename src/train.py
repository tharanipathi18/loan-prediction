from sklearn.ensemble import RandomForestClassifier
import pickle

def train_model(X_train, y_train):

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=4,
        random_state=42
    )

    model.fit(X_train, y_train)

    pickle.dump(model, open("model/random_forest_model.pkl", "wb"))

    return model