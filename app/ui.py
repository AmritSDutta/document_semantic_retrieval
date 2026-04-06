import streamlit as st
import requests

st.title("FastAPI + Streamlit Demo")

if st.button("Send hello"):
    res = requests.get("http://localhost:8000/api/hello")
    st.write("Response:", res.json()["message"])
