import streamlit as st
from inference import predict

st.title("Food Image Authenticity Detector")

uploaded = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])

if uploaded:
    st.image(uploaded)
    with open("temp.jpg", "wb") as f:
        f.write(uploaded.getbuffer())

    results = predict("temp.jpg")
    st.subheader("Prediction")
    st.json(results)