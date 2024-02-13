import streamlit as st
import pickle
import re
import numpy as np

# Load the exported model using pickle
model_load_path = "./model/logisticreg_model.pkl"

with open(model_load_path, 'rb') as file:
    loaded_model = pickle.load(file)

# Load the TF-IDF vectorizer used during training
vectorizer_path = "./model/tfidf_vectorizer.pkl"
with open(vectorizer_path, 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# Function to predict fake news
def predict_fake_news(text, vectorizer, model):
    # Preprocess the input textpreprocess   
    processed_text = preprocess_text(text)
    
    # Use the provided TF-IDF vectorizer to transform the text to a numeric format
    text_vectorized = vectorizer.transform([processed_text])

    # Make predictions using the loaded model
    prediction = model.predict(text_vectorized)[0]
    return "It's a Fake News" if prediction == 0 else "It's a Real News"

# Streamlit app
st.title("Fake News Prediction App")

st.markdown("##### Hey i am Kartik (13 years old) and i have built this ML Project which is Fake News Prediction in which i have used 2 ML Models Logistic Regression and Random Forest")
st.markdown("##### \n-The dataset used in this project is: [Fake News Detection Dataset](https://www.kaggle.com/datasets/jainpooja/fake-news-detection)")
st.markdown("##### \n- The merged dataset contains a collection of news articles labeled as either 'fake' or 'true,' sourced from two separate files. Each article includes a title, text, subject category, date of publication, and a class label indicating its authenticity. Spanning from August to December 2017, the dataset covers diverse topics including domestic and international politics, with mentions of prominent figures like Donald Trump and Pope Francis. This dataset provides valuable insights into the landscape of news reporting during that time period, offering potential avenues for analyzing the spread of misinformation and the dissemination of factual information.")
st.markdown("##### \n-My Linkedin Profile: [Kartik Singh - LinkedIn](https://www.linkedin.com/in/kartik-singh-codolearn/)")
st.markdown("##### \n-My Kaggle Profile: [Kartikexe - Kaggle](https://www.kaggle.com/kartikexe)")


# Input text box
user_input = st.text_area("Enter news text:")

# Predict button
if st.button("Predict"):
    if user_input:
        prediction = predict_fake_news(user_input, tfidf_vectorizer, loaded_model)
        st.success(f"The news is: {prediction}")
    else:
        st.warning("Please enter some news text.")
