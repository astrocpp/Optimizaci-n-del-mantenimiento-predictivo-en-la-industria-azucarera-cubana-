import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(
    page_title="Mantenimiento Predictivo - Industria Azucarera",
    page_icon="🏭",
    layout="wide"
)


@st.cache_resource
def cargar_modelo():
    """Carga el modelo entrenado y el scaler."""
    try:
        modelo = joblib.load('modelo_mantenimiento.pkl')
        scaler = joblib.load('scaler.pkl')
        features = joblib.load('features.pkl')
        return modelo, scaler, features
    except FileNotFoundError:
        st.error("❌ No se encontró el modelo entrenado. Ejecute primero 'main.py' para entrenar el modelo.")
        return None, None, None


st.title("🏭 Sistema de Mantenimiento Predictivo")
st.subheader("Industria Azucarera Cubana")
st.markdown("---")


modelo, scaler, features = cargar_modelo()

if modelo is None:
    st.stop()


st.sidebar.header("📊 Parámetros del Equipo")
st.sidebar.markdown("Ingrese los valores actuales del equipo a evaluar:")


st.sidebar.subheader("Sensores y Condiciones Operativas")
temperatura_motor = st.sidebar.slider("Temperatura del Motor (°C)", 50.0, 130.0, 85.0, 0.5)
vibracion = st.sidebar.slider("Vibración (mm/s)", 0.0, 15.0, 5.5, 0.1)
horas_operacion = st.sidebar.slider("Horas de Operación Acumuladas", 0, 15000, 3000, 100)
presion_hidraulica = st.sidebar.slider("Presión Hidráulica (PSI)", 50.0, 250.0, 150.0, 5.0)
rpm_promedio = st.sidebar.slider("RPM Promedio", 500, 3000, 1500, 50)


st.sidebar.subheader("Contexto Operativo (Escasez)")
disponibilidad_repuestos = st.sidebar.selectbox(
    "Disponibilidad de Repuestos",
    [1, 0],
    format_func=lambda x: "✅ Disponible" if x == 1 else "❌ Escasez"
)
calidad_combustible = st.sidebar.selectbox(
    "Calidad del Combustible",
    [3, 2, 1],
    format_func=lambda x: {3: "🟢 Alta", 2: "🟡 Media", 1: "🔴 Baja"}[x]
)
dias_mantenimiento = st.sidebar.slider("Días desde Último Mantenimiento", 0, 365, 90, 1)
edad_equipo = st.sidebar.slider("Edad del Equipo (años)", 1, 50, 15, 1)

# Botón de predicción
st.sidebar.markdown("---")
predecir = st.sidebar.button("🔮 Predecir Estado del Equipo", use_container_width=True)

# Panel principal
if predecir:
    # Crear DataFrame con los datos de entrada
    datos_entrada = pd.DataFrame({
        'temperatura_motor': [temperatura_motor],
        'vibracion': [vibracion],
        'horas_operacion': [horas_operacion],
        'presion_hidraulica': [presion_hidraulica],
        'rpm_promedio': [rpm_promedio],
        'disponibilidad_repuestos': [disponibilidad_repuestos],
        'calidad_combustible': [calidad_combustible],
        'dias_mantenimiento': [dias_mantenimiento],
        'edad_equipo': [edad_equipo]
    })
    
    # Escalar datos
    datos_escalados = scaler.transform(datos_entrada)
    
    # Hacer predicción
    prediccion = modelo.predict(datos_escalados)[0]
    probabilidad = modelo.predict_proba(datos_escalados)[0]
    
    # Mostrar resultados
    st.header("📈 Resultado de la Predicción")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if prediccion == 1:
            st.error("⚠️ ALTO RIESGO DE FALLA")
            st.markdown(f"**Probabilidad de Falla:** {probabilidad[1]*100:.1f}%")
            st.markdown("---")
            st.subheader("🔧 Recomendaciones")
            st.warning("""
            - **Mantenimiento urgente requerido**
            - Verificar temperatura del motor
            - Revisar sistema de vibración
            - Programar mantenimiento preventivo inmediato
            - Evaluar disponibilidad de repuestos críticos
            """)
        else:
            st.success("✅ EQUIPO OPERATIVO")
            st.markdown(f"**Probabilidad de Falla:** {probabilidad[1]*100:.1f}%")
            st.markdown("---")
            st.subheader("📋 Recomendaciones")
            st.info("""
            - Continuar operación normal
            - Mantener monitoreo periódico
            - Programar próximo mantenimiento según cronograma
            """)
    
    with col2:
        # Gráfico de probabilidades
        fig, ax = plt.subplots(figsize=(8, 4))
        categorias = ['Sin Falla', 'Con Falla']
        probabilidades = [probabilidad[0], probabilidad[1]]
        colores = ['#2ecc71', '#e74c3c']
        
        bars = ax.bar(categorias, probabilidades, color=colores, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Probabilidad')
        ax.set_title('Distribución de Probabilidades')
        ax.set_ylim(0, 1)
        
        # Añadir valores en las barras
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height*100:.1f}%',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Mostrar datos de entrada
    st.markdown("---")
    st.subheader("📊 Datos de Entrada Analizados")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Temperatura Motor", f"{temperatura_motor}°C")
        st.metric("Vibración", f"{vibracion} mm/s")
        st.metric("Horas Operación", f"{horas_operacion} h")
    
    with col2:
        st.metric("Presión Hidráulica", f"{presion_hidraulica} PSI")
        st.metric("RPM Promedio", f"{rpm_promedio}")
        st.metric("Días sin Mantenimiento", f"{dias_mantenimiento}")
    
    with col3:
        estado_rep = "Disponible" if disponibilidad_repuestos == 1 else "Escasez"
        estado_comb = {3: "Alta", 2: "Media", 1: "Baja"}[calidad_combustible]
        st.metric("Repuestos", estado_rep)
        st.metric("Combustible", estado_comb)
        st.metric("Edad Equipo", f"{edad_equipo} años")

else:
    # Página inicial
    st.header("🎯 Objetivo del Sistema")
    st.markdown("""
    Este sistema utiliza **Aprendizaje Automático Supervisado** (Random Forest) para predecir 
    la probabilidad de fallas en equipos de la industria azucarera cubana, considerando las 
    condiciones específicas de escasez de repuestos y combustibles.
    
    ### 📋 Características del Modelo
    - **Tipo:** Clasificación Binaria (Falla / Sin Falla)
    - **Algoritmo:** Random Forest Classifier
    - **Características:** 9 variables (sensores + contexto operativo)
    - **Entrenamiento:** 2000 muestras sintéticas basadas en condiciones reales
    
    ### 🔍 Instrucciones
    1. Ajuste los parámetros en la barra lateral según las condiciones actuales del equipo
    2. Presione el botón **"Predecir Estado del Equipo"**
    3. Revise el resultado y las recomendaciones generadas
    """)
    
    # Mostrar información del modelo
    st.markdown("---")
    st.subheader("ℹ️ Información del Modelo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Variables de Entrada:**
        - Temperatura del motor (°C)
        - Vibración (mm/s)
        - Horas de operación acumuladas
        - Presión hidráulica (PSI)
        - RPM promedio
        """)
    
    with col2:
        st.markdown("""
        **Variables Contextuales:**
        - Disponibilidad de repuestos
        - Calidad del combustible
        - Días desde último mantenimiento
        - Edad del equipo (años)
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
    <p>Proyecto Académico - Aprendizaje Automático y Minería de Datos</p>
    <p>2do Año de Ingeniería Informática | 2026</p>
</div>
""", unsafe_allow_html=True)
