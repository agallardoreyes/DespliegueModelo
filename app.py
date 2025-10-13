import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de la página
st.set_page_config(page_title="🫀 Predicción de Infarto", layout="centered")
st.title("🫀 Predicción de Riesgo de Infarto")
st.markdown("Esta aplicación permite evaluar el riesgo de infarto a partir de variables clínicas codificadas. Ideal para actividades colaborativas y validación ética en el aula.")

# Cargar modelo
with open("model4.pkl", "rb") as file:
    model = pickle.load(file)

# Diccionarios de codificación
edad_dict = {
    "Primera Infancia": 1,
    "Infancia": 2,
    "Adolescencia temprana": 3,
    "Juventud": 4,
    "Adultez": 5,
    "Vejez": 6
}

glucosa_dict = {
    "Normal": 1,
    "Prediabetes": 2,
    "Diabetes": 3
}

imc_dict = {
    "Bajo Peso": 1,
    "Peso Saludable": 2,
    "Sobrepeso": 3,
    "Obesidad": 4
}

estado_dict = {
    "Sí": 1,
    "No": 2
}

trabajo_dict = {
    "Emprendedor": 1,
    "Empresa privada": 2,
    "En gobierno": 3,
    "Nunca trabajó": 4,
    "Cuidar niños": 4
}

# Controles de entrada
st.sidebar.header("📋 Ingrese los datos del paciente")

hipertension = st.sidebar.selectbox("¿Tiene hipertensión?", ["No", "Sí"])
problema_cardiaco = st.sidebar.selectbox("¿Tiene problemas cardíacos?", ["No", "Sí"])
edad_cat = st.sidebar.selectbox("Edad", list(edad_dict.keys()))
glucosa_cat = st.sidebar.selectbox("Glucosa", list(glucosa_dict.keys()))
imc_cat = st.sidebar.selectbox("IMC", list(imc_dict.keys()))
estado_cat = st.sidebar.selectbox("Estado civil", list(estado_dict.keys()))
trabajo_cat = st.sidebar.selectbox("Tipo de trabajo", list(trabajo_dict.keys()))

# Botón de predicción
if st.button("🔍 Predecir riesgo"):
    input_data = pd.DataFrame({
        'Flag_hipertension': [1 if hipertension == "Sí" else 0],
        'Flag_problem_cardiaco': [1 if problema_cardiaco == "Sí" else 0],
        'Edad_Encoded': [edad_dict[edad_cat]],
        'Gluocosa_Encoded': [glucosa_dict[glucosa_cat]],
        'IMC_Encoded': [imc_dict[imc_cat]],
        'Estado_Encoded': [estado_dict[estado_cat]],
        'TipoTrabajo_Encoded': [trabajo_dict[trabajo_cat]]
    })

    prediction = model.predict_proba(input_data)[0][1]
    st.success(f"🧠 Riesgo estimado de infarto: {round(prediction * 100, 2)}%")

    # Visualización simbólica mejorada
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=input_data.columns, y=input_data.values[0], palette="Reds", ax=ax)
    ax.set_title("🔎 Perfil clínico del paciente")
    ax.set_ylabel("Valor codificado")
    ax.set_xlabel("Variable")
    ax.set_xticklabels(
        ['Hipertensión', 'Problema cardíaco', 'Edad', 'Glucosa', 'IMC', 'Estado civil', 'Tipo de trabajo'],
        rotation=90
    )
    st.pyplot(fig)

    # Ficha simbólica
    st.markdown("### 🧾 Ficha de validación ética")
    etiquetas = ['Hipertensión', 'Problema cardíaco', 'Edad', 'Glucosa', 'IMC', 'Estado civil', 'Tipo de trabajo']
    valores = [hipertension, problema_cardiaco, edad_cat, glucosa_cat, imc_cat, estado_cat, trabajo_cat]
    for etiqueta, valor in zip(etiquetas, valores):
        st.markdown(f"- **{etiqueta}**: {valor}")
    st.markdown(f"**🧠 Riesgo estimado:** {round(prediction * 100, 2)}%")

