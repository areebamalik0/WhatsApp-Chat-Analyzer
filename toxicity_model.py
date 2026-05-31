#libraries
import pandas as pd
import re
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import accuracy_score

#dataset
toxic_df = pd.read_csv("C:/Users/SystemUsername/Desktop/WhatsApp Chat Analyzer/csv files/train.csv")

# text cleaning
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if pd.isna(text) or text.strip() == "":
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", '', text)
    text = re.sub(r"[^a-z\s]", '', text)
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

toxic_df["clean_text"] = toxic_df["comment_text"].apply(clean_text)
toxic_df = toxic_df[toxic_df["clean_text"].str.len() > 0]
toxic_cols = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
toxic_df["toxic_label"] = toxic_df[toxic_cols].max(axis=1)

X = toxic_df["clean_text"]
y = toxic_df["toxic_label"]

# tf-idf vectorization
vectorizer = TfidfVectorizer(max_features=2000)
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

toxicity_model = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
toxicity_model.fit(X_train, y_train)

# model accuracy
#pred = toxicity_model.predict(X_test)
#print("Toxicity Model Accuracy:", accuracy_score(y_test, pred))

#dump
joblib.dump(toxicity_model, "C:/Users/SystemUsername/Desktop/WhatsApp Chat Analyzer/toxicity_model.pkl")
joblib.dump(vectorizer, "C:/Users/SystemUsername/Desktop/WhatsApp Chat Analyzer/toxicity_vectorizer.pkl")
print("Toxicity model trained and saved successfully")
