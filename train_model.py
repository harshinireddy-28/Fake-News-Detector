import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
df_true = pd.read_csv("True.xlsx.csv")
df_fake = pd.read_csv("Fake.xlsx.csv")

# Add labels
df_true["label"] = "REAL"
df_fake["label"] = "FAKE"

# Merge datasets
df = pd.concat([df_true, df_fake])
print(df["label"].value_counts())
# Features and labels
X = df["text"]
y = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)

# Model training
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)
from sklearn.metrics import accuracy_score
# Transform test data
X_test_tfidf = vectorizer.transform(X_test)

# Predict
y_pred = model.predict(X_test_tfidf)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:",round(accuracy * 100, 2), "%" )

# Save model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model and vectorizer saved successfully!")
