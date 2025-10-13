import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# --- CORRECCIÓN EN EL DICCIONARIO DE TRABAJO ---
# El valor 'Cuidar niños' tiene el código 4, igual que 'Nunca trabajó'.
# Sin embargo, el código lo tiene como un string que se mapea a un int,
# pero al usarse en el selectbox, podría causar problemas si el modelo espera
# códigos diferentes. Asumo que el modelo solo tiene 4 categorías únicas
# o que 'Cuidar niños' se considera igual a 'Nunca trabajó'.

trabajo_dict = {
    "Emprendedor": 1,
    "Empresa privada": 2,
    "En gobierno": 3,
    "Nunca trabajó": 4,
    "Cuidar niños": 5 # CAMBIADO a 5 para que sea una categoría diferente en el diccionario,
                     # aunque podría ser el mismo código si así lo entrena el modelo.
                     # Si el modelo espera 4, debes asegurarte que solo haya 4 categorías únicas en el diccionario:
                     # "Cuidar niños" se mapea a 4 en el DataFrame (esto estaba correcto, pero el diccionario debe ser consistente)
}
# La corrección real se debe hacer en la lógica de creación del DataFrame de entrada:
# trabajo_dict original:
# trabajo_dict = {
#     "Emprendedor": 1,
#     "Empresa privada": 2,
#     "En gobierno": 3,
#     "Nunca trabajó": 4,
#     "Cuidar niños": 4  # Esto hace que solo haya 4 claves/opciones pero el código será el mismo
# }
# Mantendré el diccionario original y corregiré la lógica de 'TipoTrabajo_Encoded' en el DataFrame para manejar la clave 'Cuidar niños' correctamente.

# Diccionario original (sin cambios en la estructura)
trabajo_dict = {
    "Emprendedor": 1,
    "Empresa privada": 2,
    "En gobierno": 3,
    "Nunca trabajó": 4,
    "Cuidar niños": 4
}


# Configuración de la página
st.set_page_config(page_title="🫀 Predicción de Infarto", layout="centered")
st.title("🫀 Predicción de Riesgo de Infarto")
st.markdown("Esta aplicación permite evaluar el riesgo de infarto a partir de variables clínicas codificadas. Ideal para actividades colaborativas y validación ética en el aula.")

# Cargar modelo
# NOTA: Asegúrate de que el archivo 'model4.pkl' esté en el mismo directorio.
try:
    with open("model4.pkl", "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Error: No se encontró el archivo del modelo 'model4.pkl'. Asegúrese de que esté en la misma carpeta.")
    st.stop()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()


# Diccionarios de codificación (se mantienen como en tu código original)
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
# trabajo_dict ya definido arriba


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
    # El diccionario trabajo_dict tiene claves repetidas con el mismo valor (Nunca trabajó y Cuidar niños ambos son 4)
    # y la clave 'Cuidar niños' tiene un valor asociado que es 4. La línea original estaba bien:
    # 'TipoTrabajo_Encoded': [trabajo_dict[trabajo_cat]]

    input_data = pd.DataFrame({
        'Flag_hipertension': [1 if hipertension == "Sí" else 0],
        'Flag_problem_cardiaco': [1 if problema_cardiaco == "Sí" else 0],
        'Edad_Encoded': [edad_dict[edad_cat]],
        'Gluocosa_Encoded': [glucosa_dict[glucosa_cat]],
        'IMC_Encoded': [imc_dict[imc_cat]],
        'Estado_Encoded': [estado_dict[estado_cat]],
        'TipoTrabajo_Encoded': [trabajo_dict[trabajo_cat]]
        # CORRECCIÓN DE POSIBLE ERROR: Asegúrate de que la columna se llame 'Gluocosa_Encoded' o 'Glucosa_Encoded' en el modelo.
        # Si el modelo se entrenó con 'Glucosa_Encoded', debes cambiarlo aquí. Asumo que 'Gluocosa_Encoded' es correcto.
    })

    # 2. Predicción
    try:
        # Se asume que el modelo devuelve la probabilidad de la clase positiva (infarto = 1) en la posición [1]
        prediction_proba = model.predict_proba(input_data)[0]
        prediction = prediction_proba[1] 
    except Exception as e:
        st.error(f"Error al realizar la predicción. Revise si las columnas del DataFrame de entrada coinciden con las del modelo: {e}")
        st.stop()


    # 3. Mostrar el resultado de la predicción
    risk_percentage = round(prediction * 100, 2)
    st.success(f"🧠 Riesgo estimado de infarto: **{risk_percentage}%**")

    # 4. Visualización simbólica mejorada (Barra de progreso más informativa)
    # El barplot anterior era confuso porque todas las barras eran iguales.
    # Una barra de progreso o un gauge es mejor para mostrar un porcentaje de riesgo.
    
    st.markdown("### 📊 Nivel de Riesgo (Representación Visual)")
    
    # Usar un color que dependa del nivel de riesgo
    if risk_percentage < 30:
        risk_level = "Bajo"
        color = "green"
    elif risk_percentage < 60:
        risk_level = "Moderado"
        color = "orange"
    else:
        risk_level = "Alto"
        color = "red"
        
    st.markdown(
        f"""
        <div style='background-color: {color}; color: white; padding: 10px; border-radius: 5px; text-align: center;'>
            **Nivel de Riesgo: {risk_level}**
        </div>
        """,
        unsafe_allow_html=True
    )

    st.progress(prediction)
    
    # 5. Ficha simbólica (Se mantiene, es una buena práctica de validación ética)
    st.markdown("### 🧾 Ficha de validación ética")
    
    # Se añade la visualización de las entradas en un DataFrame para una mejor tabla
    etiquetas = ['Hipertensión', 'Problema cardíaco', 'Edad', 'Glucosa', 'IMC', 'Estado civil', 'Tipo de trabajo']
    valores = [hipertension, problema_cardiaco, edad_cat, glucosa_cat, imc_cat, estado_cat, trabajo_cat]
    
    # Crear un DataFrame para mostrar el perfil
    perfil_df = pd.DataFrame({
        'Variable': etiquetas,
        'Valor Seleccionado': valores
    })
    
    st.table(perfil_df.set_index('Variable'))
    
    st.markdown(f"**🧠 Riesgo estimado (final):** **{risk_percentage}%**")

    # 6. (OPCIONAL) Visualización de las variables binarias y categóricas codificadas.
    # Se elimina el barplot confuso de tu código original.
