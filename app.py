import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de la página
st.set_page_config(page_title="🫀 Predicción de Infarto", layout="centered")
st.title("🫀 Predicción de Riesgo de Infarto")
st.markdown("Esta aplicación permite evaluar el riesgo de infarto a partir de variables clínicas. Ideal para actividades colaborativas y validación ética en el aula.")

# Cargar modelo
with open("model4.pkl", "rb") as file:
    model = pickle.load(file)

# Controles de entrada
st.sidebar.header("📋 Ingrese los datos del paciente")
age = st.sidebar.slider("Edad", 18, 100, 50)
cholesterol = st.sidebar.number_input("Colesterol (mg/dL)", min_value=100, max_value=400, value=200)
blood_pressure = st.sidebar.number_input("Presión arterial (mmHg)", min_value=80, max_value=200, value=120)
smoker = st.sidebar.selectbox("¿Fuma actualmente?", ["Sí", "No"])
diabetic = st.sidebar.selectbox("¿Es diabético?", ["Sí", "No"])

# Codificación simbólica
smoker_bin = 1 if smoker == "Sí" else 0
diabetic_bin = 1 if diabetic == "Sí" else 0

# Botón de predicción
if st.button("🔍 Predecir riesgo"):
    input_data = pd.DataFrame({
        "age": [age],
        "cholesterol": [cholesterol],
        "blood_pressure": [blood_pressure],
        "smoker": [smoker_bin],
        "diabetic": [diabetic_bin]
    })

    prediction = model.predict_proba(input_data)[0][1]
    st.success(f"🧠 Riesgo estimado de infarto: {round(prediction * 100, 2)}%")

    # Visualización simbólica
    fig, ax = plt.subplots()
    sns.barplot(x=input_data.columns, y=input_data.values[0], palette="Reds")
    ax.set_title("🔎 Perfil clínico del paciente")
    st.pyplot(fig)

    # Ficha simbólica
    st.markdown("### 🧾 Ficha de validación ética")
    st.markdown(f"- Edad: {age} años")
    st.markdown(f"- Colesterol: {cholesterol} mg/dL")
    st.markdown(f"- Presión arterial: {blood_pressure} mmHg")
    st.markdown(f"- Fuma: {smoker}")
    st.markdown(f"- Diabético: {diabetic}")
    st.markdown(f"**Riesgo estimado:** {round(prediction * 100, 2)}%")

