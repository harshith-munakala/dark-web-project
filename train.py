import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

print("Starting training...")

# 🔥 Load very small data (FAST)
dataset = pd.read_csv("Dataset/DUTA.csv", encoding="iso-8859-1")
dataset = dataset.head(500)   # VERY FAST

X = dataset['Item_Description'].astype(str)
y = dataset['Category']

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(X)

# Train model
model = RandomForestClassifier(n_estimators=20)
model.fit(X, y)

# Create folder
os.makedirs("model", exist_ok=True)

# Save files
pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))
pickle.dump(le, open("model/label_encoder.pkl", "wb"))

print("✅ Model created successfully!")