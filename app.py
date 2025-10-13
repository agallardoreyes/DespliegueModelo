import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de la página
st.set_page_config(page_title="🫀 Predicción de Infarto", layout="centered")
st.title("🫀 Predicción de Riesgo de Infarto")
st.markdown("Esta aplicación permite evaluar el riesgo de infarto a partir de variables codificadas. Ideal para actividades colaborativas y validación ética en el aula.")

# Cargar modelo
with open("model4.pkl", "rb") as file:
    model = pickle.load(file)

# Controles de entrada
st.sidebar.header("📋 Ingrese los datos codificados del paciente")

flag_hipertension = st.sidebar.selectbox("¿Tiene hipertensión?", [0, 1])
flag_problem_cardiaco = st.sidebar.selectbox("¿Tiene problemas cardíacos?", [0, 1])
edad_encoded = st.sidebar.slider("Edad codificada", 0, 10, 4)
gluocosa_encoded = st.sidebar.slider("Glucosa codificada", 0, 5, 2)
imc_encoded = st.sidebar.slider("IMC codificado", 0, 5, 2)
estado_encoded = st.sidebar.slider("Estado civil codificado", 0, 3, 1)
tipo_trabajo_encoded = st.sidebar.slider("Tipo de trabajo codificado", 0, 5, 4)

# Botón de predicción
if st.button("🔍 Predecir riesgo"):
    input_data = pd.DataFrame({
        'Flag_hipertension': [flag_hipertension],
        'Flag_problem_cardiaco': [flag_problem_cardiaco],
        'Edad_Encoded': [edad_encoded],
        'Gluocosa_Encoded': [gluocosa_encoded],
        'IMC_Encoded': [imc_encoded],
        'Estado_Encoded': [estado_encoded],
        'TipoTrabajo_Encoded': [tipo_trabajo_encoded]
    })

    prediction = model.predict_proba(input_data)[0][1]
    st.success(f"🧠 Riesgo estimado de infarto: {round(prediction * 100, 2)}%")

    # Visualización simbólica mejorada
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=input_data.columns, y=input_data.values[0], palette="Reds", ax=ax)
    ax.set_title("🔎 Perfil codificado del paciente")
    ax.set_ylabel("Valor codificado")
    ax.set_xlabel("Variable")
    ax.tick_params(axis='x', rotation=30)
    st.pyplot(fig)

    # Ficha simbólica
    st.markdown("### 🧾 Ficha de validación ética")
    for col in input_data.columns:
        st.markdown(f"- **{col}**: {input_data[col].values[0]}")
    st.markdown(f"**🧠 Riesgo estimado:** {round(prediction * 100, 2)}%")
