# Predicción de Precios mediante Regresión Lineal desde Cero

## Descripción del Proyecto

Este proyecto aborda la implementación completa **desde cero** de un modelo de **Regresión Lineal Múltiple con Regularización** utilizando exclusivamente **NumPy** para el cálculo vectorial. 

El objetivo principal es aplicar mis conocimientos aprendidos en el curso de Aprendizaje Supervisado de Andrew Ng y validar el rendimiento de la solución propia comparándola directamente con la librería estándar de la industria, **Scikit-Learn** (`SGDRegressor`).

---

## Aspectos Clave de la Implementación

* **Preparación de Datos:** Procesamiento y escalado Z-Score evitando la fuga de datos al ajustar la media y desviación estándar únicamente con el conjunto de entrenamiento (`X_train`).
* **Función de Coste Vectorizada:** Implementación del Error Cuadrático Medio incorporando el término de penalización de regularización:
  $$J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2 + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2$$
* **Descenso de Gradiente Regularizado:** Cálculo vectorial de derivadas parciales aplicando descenso de gradiente sobre los parámetros $w$ para prevenir sobreajuste (*overfitting*).
* **Validación Cruzada:** Comparación término a término entre los parámetros aprendidos ($w$ y $b$) desde cero y los optimizados por `Scikit-Learn`.

---

## Resultados y Validación

La implementación matemática propia demostró una convergencia prácticamente exacta frente a `Scikit-Learn`:

| Parámetro / Métrica | Mi Modelo desde Cero | Scikit-Learn (`SGDRegressor`) | Diferencia Relative |
| :--- | :---: | :---: | :---: |
| **Sesgo ($b$)** | `~60,845.11` | `~60,898.44` | **< 0.08%** |
| **Peso $w_1$** | `-7,652.88` | `-8,352.98` | Misma magnitud y signo |
| **Predicción Muestra ($X_{test}[0]$)** | `36,343.25 €` | `36,363.93 €` | **~20 € (0.05%)** |

> **Nota sobre el Error:** El modelo alcanzó un MAE de **~16,000 €** (Error Relativo de ~29%). Este resultado representa el límite teórico de capacidad de un modelo lineal simple para capturar las relaciones complejas de este dataset sin caer en sobreajuste.

---

## Estructura del Repositorio

```text
├── laptop_data.csv                # Datos usados para el proyecto, descargados desde Kaggle
├── PredictorPrecioPortatil.ipynb  # Notebook con la implementación del proyecto
├── README.md                      # Presentación del proyecto
└── requirements.txt               # Dependencias
