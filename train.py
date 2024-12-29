import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import fill_missing_value, encode_categorical_columns, scale_features
import pickle

train_set = pd.read_csv("data/train.csv")
test_set = pd.read_csv("data/test.csv")

#tekrar ediyor
object_columns = ["Profession", "Dietary Habits", "Degree", "Gender", "City", "Working Professional or Student", "Sleep Duration", "Have you ever had suicidal thoughts ?", "Family History of Mental Illness"]
float_columns = ["Academic Pressure", "Work Pressure", "CGPA", "Study Satisfaction", "Job Satisfaction", "Financial Stress"]

train_set = fill_missing_value(train_set, object_columns, float_columns)
train_set = encode_categorical_columns(train_set, object_columns)
print(train_set.isnull().sum())
    
X = train_set.drop(columns= ["Depression", "Name"])
y = train_set["Depression"]

X, scaler = scale_features(X)

X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                    random_state= 43,
                                                    test_size= 0.2)

#model
model = LogisticRegression(max_iter= 1000, random_state= 43)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_val, model.predict(X_val))
print(accuracy)

'''
with open("model.pkl", "wb") as file:
    pickle.dump((model, scaler), file)
'''