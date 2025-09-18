# main.py

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

# Load word index and reverse it
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load trained model
model = tf.keras.models.load_model("simple_rnn_imdb.h5")

# Helper: decode review (optional)
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Helper: preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Streamlit UI
st.title("🎬 IMDB Movie Review Sentiment Classifier")
st.write("Enter a movie review below to predict its sentiment.")

user_input = st.text_area("Your Review", height=150)

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        processed = preprocess_text(user_input)
        prediction = model.predict(processed)[0][0]
        sentiment = "Positive 😊" if prediction > 0.5 else "Negative 😞"
        st.success(f"Sentiment: **{sentiment}**")
        st.write(f"Confidence Score: `{prediction:.4f}`")
        st.progress(float(prediction))

else:
    st.info("Awaiting input...")

# Optional footer
st.markdown("---")
st.markdown("Made with ❤️ by ANUM — aspiring AI Engineer")
