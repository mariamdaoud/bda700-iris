import requests
import streamlit as st

st.set_page_config(page_title="Iris Predictor", layout="centered")
st.title("🌸 Iris Species Predictor")
st.caption("Streamlit front end -> FastAPI inference endpoint (http://localhost:8000)")

sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 0.2)

if st.button("Predict species"):
    payload = {
        "sepal_length": sepal_length, "sepal_width": sepal_width,
        "petal_length": petal_length, "petal_width": petal_width,
    }
    try:
        result = requests.post("http://localhost:8000/predict", json=payload, timeout=5).json()
        st.success(f"Predicted species: **{result['predicted_class']}**")
        st.bar_chart(result["probabilities"])
    except requests.exceptions.ConnectionError:
        st.error("Could not reach the FastAPI backend at localhost:8000 — is it running? "
                  "(Run the uvicorn cell in the notebook first.)")
