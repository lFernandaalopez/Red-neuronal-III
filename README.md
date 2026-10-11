# Redes Neuronales III – Optimización, regularización y evaluación (caso de salud)

Trabajo de la **Especialización en Inteligencia Artificial** – Módulo de Deep Learning (semanas 5, 6 y 7).

El proyecto aplica técnicas de optimización, regularización y evaluación de redes neuronales con **TensorFlow + Keras** (y un anexo en **PyTorch**) a un problema del sector salud: la **clasificación de tumores de mama como benignos o malignos** a partir de características de núcleos celulares.

> **Alcance:** uso exclusivamente académico. Los modelos no están validados clínicamente y no deben usarse para tomar decisiones médicas.

## Caso de uso y datos

- **Dataset:** Breast Cancer Wisconsin (Diagnostic) – 569 muestras, 30 características numéricas, sin valores faltantes (357 benignos y 212 malignos). Se carga directamente desde `scikit-learn`, sin descargas externas.
- **Clase positiva:** maligno (1). Clase negativa: benigno (0).
- **Distribución estratificada (semilla 42):**

| Conjunto | Proporción | Muestras | Malignos |
|---|---|---|---|
| Train | 60 % | 341 | 127 |
| Dev | 20 % | 114 | 43 |
| Test | 20 % | 114 | 42 |

- **Preprocesamiento:** estandarización (media 0, desviación 1) ajustada solo con el conjunto de entrenamiento, para evitar fuga de datos.

## Notebooks (Google Colab)

| # | Práctica | Contenido | Colab |
|---|---|---|---|
| 1 | Optimización (semana 5) | Feature Scaling, mini-batch, Momentum, RMSProp, Adam, Learning Rate Decay, Batch Normalization | [Abrir en Colab](https://colab.research.google.com/drive/1hbcE3z4zZ1avVJt9U6YZt9QERoXFPm_f?usp=sharing) |
| 2 | Regularización (semana 6) | Bias/variance, L1, L2, Dropout, Data Augmentation (ruido gaussiano), Early Stopping | [Abrir en Colab](https://colab.research.google.com/drive/1ZnUQogqkt003SksJy7jxv-w3Z3ZHZJnZ?usp=sharing) |
| 3 | Métricas de evaluación (semana 7) | Accuracy, Precision, Recall, F1, Sensibilidad, Especificidad, matriz de confusión, curva ROC, ajuste del umbral | [Abrir en Colab](https://colab.research.google.com/drive/1H8_PsWshQCwtrj1LKBtGBCsajnRCBEiN?usp=sharing) |
| 4 | Anexo PyTorch | Modelo final equivalente en PyTorch con Early Stopping manual | [Abrir en Colab](https://colab.research.google.com/drive/1VK-v0y-JZpOM-dqGnuaSMkDR9zgJB7di?usp=sharing) |

Cada notebook es independiente: carga los datos, define el modelo y ejecuta su práctica. No requieren GPU.

## Cómo ejecutarlos

1. Abre el enlace de Colab del notebook que quieras revisar.
2. Ve a **Entorno de ejecución → Ejecutar todo**.
3. No hace falta instalar nada: TensorFlow, PyTorch, scikit-learn, pandas y matplotlib ya vienen en Colab.

Para ejecutarlos en local:

```bash
pip install -r requirements.txt
jupyter notebook
```

## Metodología

- Cada configuración se entrena con **tres semillas** y se reporta el promedio, porque el conjunto de datos es pequeño y una sola ejecución depende mucho de la inicialización.
- Las comparaciones de hiperparámetros se hacen sobre el conjunto **dev**; el conjunto **test** se usa una sola vez para evaluar el modelo final.
- Arquitectura base: MLP de 2 capas ocultas (64 y 32 neuronas, ReLU) con salida sigmoide y pérdida de entropía cruzada binaria. Para estudiar el sobreajuste se usa una red mayor (256-128-64).

## Resultados principales

Los resultados detallados, tablas, figuras y el análisis completo están en el informe en PDF entregado en la plataforma institucional. Resumen:

- El **Feature Scaling** fue la mejora más importante (accuracy en dev de 92,4 % a 96,8 %).
- Con datos estandarizados, los optimizadores quedaron en un rango estrecho (95,9 % a 97,4 % en dev), dentro del ruido que permite un conjunto de 114 muestras.
- El modelo sin regularizar mostró **alta variance**; la combinación **L2 + Dropout + Early Stopping** dio la mejor exactitud en dev (97,7 %) con menor variabilidad entre semillas.
- El modelo final obtuvo en test accuracy de 97,4 %, precision de 100 %, sensibilidad de 92,9 % y AUC de 0,989. Ajustar el umbral de decisión aumenta la sensibilidad a costa de la especificidad.

> Las cifras pueden variar ligeramente al volver a ejecutar los notebooks, por la inicialización aleatoria y las diferencias entre librerías.

## Estructura del repositorio

```text
redes-neuronales-iii/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_optimizacion.ipynb
│   ├── 02_regularizacion.ipynb
│   ├── 03_metricas_y_evaluacion.ipynb
│   └── 04_pytorch_anexo.ipynb
└── imagenes/            (capturas de resultados)
```

## Tecnologías

Python · TensorFlow / Keras · PyTorch · scikit-learn · pandas · NumPy · Matplotlib · Google Colab

## Referencias

- Wolberg, W. H., Street, W. N., & Mangasarian, O. L. (1995). *Breast Cancer Wisconsin (Diagnostic)* [Conjunto de datos]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
- Deep Learning with TensorFlow and Keras (3rd ed.). (2023). Packt Publishing.
- Deep Learning with PyTorch. (2020). Manning Publications.

## Autor

**LAURA LOPEZ** – Especialización en Inteligencia Artificial, Universidad de Cundinamarca.
