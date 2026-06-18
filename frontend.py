
# frontend.py
import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"  # FastAPI backend URL

st.set_page_config(
    page_title="Next Word Predictor by Nadish",
    page_icon="🤖",
    layout="centered"
)

st.markdown("<h1 style='text-align:center; color:#4B0082;'>Next Word Predictor 🤖</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Enter a sentence and get AI to predict the next word!</p>", unsafe_allow_html=True)
st.markdown("---")

user_input = st.text_area("Enter your text here:", placeholder="Type a sentence...", height=100)

if st.button("Predict Next Word"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        # Call FastAPI endpoint
        try:
            response = requests.post(API_URL, json={"text": user_input})
            predicted_word = response.json().get("next_word", "")
            st.success(f"Predicted next word: **{predicted_word}**")
        except Exception as e:
            st.error(f"Error calling API: {e}")

st.markdown("<div style='text-align:center; margin-top:50px;'><p style='color:gray;'>Assignment Demo using FastAPI + Streamlit</p></div>", unsafe_allow_html=True)
