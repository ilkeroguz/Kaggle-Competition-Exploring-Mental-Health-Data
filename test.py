import pandas as pd
import pickle
from utils import fill_missing_value, encode_categorical_columns, scale_features

test_set = pd.read_csv("data/test.csv")

object_columns = ["Profession", "Dietary Habits", "Degree", "Gender", "City", "Working Professional or Student", "Sleep Duration", "Have you ever had suicidal thoughts ?", "Family History of Mental Illness"]
float_columns = ["Academic Pressure", "Work Pressure", "CGPA", "Study Satisfaction", "Job Satisfaction", "Financial Stress"]

test_set = fill_missing_value(test_set, object_columns, float_columns)
test_set = encode_categorical_columns(test_set, object_columns)

X_test = test_set.drop(columns= "Name")
test_ids = test_set["id"]

with open("model.pkl", "rb") as file:
    model, scaler = pickle.load(file)

X_test, _ = scale_features(X_test, scaler)

y_test_pred = model.predict(X_test)

output = pd.DataFrame({"id": test_ids, "cevap": y_test_pred})
output.to_csv("tahmin_sonuclari.csv", index=False)