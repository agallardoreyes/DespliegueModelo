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

# Controles de entrada
st.sidebar.header("📋 Ingrese los datos del paciente")

hipertension = st.sidebar.selectbox("¿Tiene hipertensión?", [0, 1])
problema_cardiaco = st.sidebar.selectbox("¿Tiene problemas cardíacos?", [0, 1])
edad = st.sidebar.slider("Edad", 0, 10, 4)
glucosa = st.sidebar.slider("Glucosa", 0, 5, 2)
imc = st.sidebar.slider("IMC", 0, 5, 2)
estado = st.sidebar.slider("Estado civil", 0, 3, 1)
tipo_trabajo = st.sidebar.slider("Tipo de trabajo", 0, 5, 4)

# Botón de predicción
if st.button("🔍 Predecir riesgo"):
    input_data = pd.DataFrame({
        'Flag_hipertension': [hipertension],
        'Flag_problem_cardiaco': [problema_cardiaco],
        'Edad_Encoded': [edad],
        'Gluocosa_Encoded': [glucosa],
        'IMC_Encoded': [imc],
        'Estado_Encoded': [estado],
        'TipoTrabajo_Encoded': [tipo_trabajo]
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
    for i, col in enumerate(input_data.columns):
        st.markdown(f"- **{etiquetas[i]}**: {input_data[col].values[0]}")
    st.markdown(f"**🧠 Riesgo estimado:** {round(prediction * 100, 2)}%")
