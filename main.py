import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

def generar_dataset_azucarero(n_muestras=2000):
    
    temperatura_motor = np.random.normal(85, 15, n_muestras)  
    vibracion = np.random.normal(5.5, 2.0, n_muestras)  
    horas_operacion = np.random.randint(100, 10000, n_muestras) 
    presion_hidraulica = np.random.normal(150, 30, n_muestras)  
    rpm_promedio = np.random.randint(800, 2500, n_muestras)
    
    disponibilidad_repuestos = np.random.choice([0, 1], n_muestras, p=[0.4, 0.6])  
    calidad_combustible = np.random.choice([1, 2, 3], n_muestras, p=[0.3, 0.5, 0.2])  
    dias_mantenimiento = np.random.randint(1, 365, n_muestras)  
    edad_equipo = np.random.randint(5, 40, n_muestras)  
    
    probabilidad_falla = (
        0.3 * (temperatura_motor > 95) +
        0.25 * (vibracion > 7.0) +
        0.15 * (horas_operacion > 5000) +
        0.15 * (1 - disponibilidad_repuestos) +
        0.10 * (calidad_combustible == 1) +
        0.05 * (dias_mantenimiento > 180) +
        0.05 * (edad_equipo > 25)
    )
    
    probabilidad_falla = np.clip(probabilidad_falla + np.random.normal(0, 0.1, n_muestras), 0, 1)
    falla = (probabilidad_falla > 0.5).astype(int)
    
    df = pd.DataFrame({
        'temperatura_motor': temperatura_motor,
        'vibracion': vibracion,
        'horas_operacion': horas_operacion,
        'presion_hidraulica': presion_hidraulica,
        'rpm_promedio': rpm_promedio,
        'disponibilidad_repuestos': disponibilidad_repuestos,
        'calidad_combustible': calidad_combustible,
        'dias_mantenimiento': dias_mantenimiento,
        'edad_equipo': edad_equipo,
        'falla': falla
    })
    
    return df

def entrenar_modelo():
    """
    Entrena un modelo de Random Forest para predecir fallas en equipos azucareros.
    """
    print("=" * 60)
    print("ENTRENAMIENTO DE MODELO DE MANTENIMIENTO PREDICTIVO")
    print("Industria Azucarera Cubana")
    print("=" * 60)
    
    print("\n[1/5] Generando dataset sintético...")
    df = generar_dataset_azucarero(2000)
    print(f"Dataset generado: {df.shape[0]} muestras, {df.shape[1]} características")
    print(f"Distribución de clases: {df['falla'].value_counts().to_dict()}")
    
    df.to_csv('dataset_azucarero.csv', index=False)
    print("✓ Dataset guardado en 'dataset_azucarero.csv'")
    
    print("\n[2/5] Preparando datos para entrenamiento...")
    X = df.drop('falla', axis=1)
    y = df['falla']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Conjunto de entrenamiento: {X_train.shape[0]} muestras")
    print(f"Conjunto de prueba: {X_test.shape[0]} muestras")
    
    print("\n[3/5] Escalando características...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n[4/5] Entrenando modelo Random Forest...")
    modelo = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    modelo.fit(X_train_scaled, y_train)
    print("✓ Modelo entrenado exitosamente")
    
    print("\n[5/5] Evaluando modelo...")
    y_pred = modelo.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nMétricas de Evaluación:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred, target_names=['Sin Falla', 'Con Falla']))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Sin Falla', 'Con Falla'],
                yticklabels=['Sin Falla', 'Con Falla'])
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión')
    plt.tight_layout()
    plt.savefig('matriz_confusion.png', dpi=150)
    print("✓ Matriz de confusión guardada en 'matriz_confusion.png'")
    
    # Importancia de características
    importancias = modelo.feature_importances_
    importancia_df = pd.DataFrame({
        'Característica': X.columns,
        'Importancia': importancias
    }).sort_values('Importancia', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importancia_df, x='Importancia', y='Característica', palette='viridis')
    plt.title('Importancia de Características en la Predicción de Fallas')
    plt.xlabel('Importancia')
    plt.ylabel('Característica')
    plt.tight_layout()
    plt.savefig('importancia_caracteristicas.png', dpi=150)
    print("✓ Gráfico de importancia guardado en 'importancia_caracteristicas.png'")
    
    # Guardar modelo y scaler
    joblib.dump(modelo, 'modelo_mantenimiento.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(list(X.columns), 'features.pkl')
    print("\n✓ Modelo, scaler y características guardados")
    
    print("\n" + "=" * 60)
    print("ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print("=" * 60)
    
    return modelo, scaler, accuracy

if __name__ == "__main__":
    modelo, scaler, accuracy = entrenar_modelo()