import streamlit as st
import pickle
import nltk
import os
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK resources if needed
try:
    stopwords.words('english')
except:
    nltk.download('stopwords')
    nltk.download('punkt')

# Preprocessing function
def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalnum()]
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    tokens = [PorterStemmer().stem(t) for t in tokens]
    return " ".join(tokens)

# Page configuration
st.set_page_config(page_title="Fake News Detector 📰", layout="centered")
st.title("Fake News Detection App")
st.write("Enter a news article below and get a prediction:")

# Load pretrained model & vectorizer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
    vectorizer = pickle.load(open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb"))
    st.success("✅ Model & Vectorizer loaded")
except Exception as e:
    st.error(f"Error loading model/vectorizer: {e}")
    st.stop()

# User input
user_input = st.text_area("Paste news article here:", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.error("⚠️ Please enter text.")
    else:
        cleaned = transform_text(user_input)
        vect = vectorizer.transform([cleaned])
        pred = model.predict(vect)[0]
        proba = model.predict_proba(vect)[0][pred]
        label = "Fake News ❌" if pred == 1 else "Real News ✅"
        st.markdown(f"### **{label}**")
        st.markdown(f"**Confidence:** {proba*100:.2f}%")
        st.balloons()
