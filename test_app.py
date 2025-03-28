import streamlit as st

st.title("Fish Species Classifier - Test")
st.write("This is a minimal test app to verify deployment.")

st.markdown("## Features")
st.write("- Upload fish images")
st.write("- Get species predictions")
st.write("- View information about fish species")

if st.button("Click me"):
    st.success("Everything is working!")