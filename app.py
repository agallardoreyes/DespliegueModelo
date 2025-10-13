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
try:
    with open("model4.pkl", "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Error: No se encontró el archivo del modelo 'model4.pkl'. Asegúrese de que esté en la misma carpeta.")
    st.stop()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

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
    # 1. Preparación de los datos de entrada
    input_data = pd.DataFrame({
        'Flag_hipertension': [1 if hipertension == "Sí" else 0],
        'Flag_problem_cardiaco': [1 if problema_cardiaco == "Sí" else 0],
        'Edad_Encoded': [edad_dict[edad_cat]],
        'Gluocosa_Encoded': [glucosa_dict[glucosa_cat]],
        'IMC_Encoded': [imc_dict[imc_cat]],
        'Estado_Encoded': [estado_dict[estado_cat]],
        'TipoTrabajo_Encoded': [trabajo_dict[trabajo_cat]]
    })

    # 2. Predicción
    try:
        prediction_proba = model.predict_proba(input_data)[0]
        prediction = prediction_proba[1] 
    except Exception as e:
        st.error(f"Error al realizar la predicción. Revise si las columnas del DataFrame de entrada coinciden con las del modelo: {e}")
        st.stop()


    # 3. Cálculo de Nivel de Riesgo y colores
    risk_percentage = round(prediction * 100, 2)
    
    if risk_percentage < 30:
        risk_level = "Bajo"
        color = "green"
    elif risk_percentage < 60:
        risk_level = "Moderado"
        color = "orange"
    else:
        risk_level = "Alto"
        color = "red"
        
    # 4. Visualización de Resultados (¡ORDEN MODIFICADO!)
    
    # A. Nivel de Riesgo y barra de progreso
    st.markdown(
        f"""
        <div style='background-color: {color}; color: white; padding: 10px; border-radius: 5px; text-align: center; margin-bottom: 10px;'>
            **Nivel de Riesgo: {risk_level}**
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # B. Riesgo estimado de infarto (st.success)
    st.success(f"🧠 Riesgo estimado de infarto: **{risk_percentage}%**")
    
    # Barra de progreso
    st.progress(prediction)

    # 5. Ficha de validación (Texto simplificado)
    st.markdown("### 🧾 Ficha de validación")
    
    etiquetas = ['Hipertensión', 'Problema cardíaco', 'Edad', 'Glucosa', 'IMC', 'Estado civil', 'Tipo de trabajo']
    valores = [hipertension, problema_cardiaco, edad_cat, glucosa_cat, imc_cat, estado_cat, trabajo_cat]
    
    # Crear un DataFrame para mostrar el perfil
    perfil_df = pd.DataFrame({
        'Variable': etiquetas,
        'Valor Seleccionado': valores
    })
    
    st.table(perfil_df.set_index('Variable'))
    
    st.markdown(f"**🧠 Riesgo estimado (final):** **{risk_percentage}%**")
