# Pre-entrega III: Clasificador supervisado con TF-IDF

Este repositorio contiene un pipeline de clasificación supervisada para el dataset **AG News**, usando vectorización estadística **TF-IDF** y un modelo baseline de **Regresión Logística**.

## Objetivo

Transformar texto crudo de noticias en vectores numéricos con `TfidfVectorizer`, entrenar un clasificador supervisado y evaluar su rendimiento sobre el split de prueba provisto por la cátedra.

## Estructura del repositorio

```text
preentrega_3_tfidf/
├── data/
│   ├── ag_news_train.csv
│   └── ag_news_test.csv
├── models/
│   └── tfidf_logistic_regression.joblib
├── outputs/
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   └── metrics.json
├── src/
│   └── train_tfidf_classifier.py
├── README.md
└── requirements.txt
```

## Dataset

Se utiliza el corpus **AG News** provisto en el módulo anterior:

- `ag_news_train.csv`: split de entrenamiento, con 8.000 documentos.
- `ag_news_test.csv`: split de evaluación, con 2.000 documentos.
- Columnas requeridas: `text` y `label`.
- Clases: `World`, `Sports`, `Business`, `Sci_Tech`.

El split de test no se utiliza para ajustar el vectorizador ni el modelo. Se reserva exclusivamente para la evaluación final.

## Preprocesamiento

La función `preprocess_text(text)` aplica:

- conversión a minúsculas;
- eliminación de etiquetas HTML;
- eliminación de URLs y correos;
- remoción de caracteres no alfabéticos;
- normalización de espacios.

El vectorizador también aplica `stop_words="english"` para reducir ruido lingüístico y evitar que palabras muy frecuentes aporten dimensiones poco informativas.

## Vectorización TF-IDF

Se implementa `TfidfVectorizer` con los siguientes parámetros principales:

- `max_features=12000`: limita el vocabulario y ayuda a controlar memoria y overfitting.
- `ngram_range=(1, 2)`: incluye unigramas y bigramas para capturar palabras individuales y expresiones cortas.
- `min_df=2`: descarta términos que aparecen una sola vez.
- `sublinear_tf=True`: suaviza el peso de términos extremadamente repetidos.
- `stop_words="english"`: elimina stop-words frecuentes del inglés.

Para evitar **Data Leakage**, el pipeline ejecuta `fit` únicamente con `ag_news_train.csv`. Luego evalúa con `ag_news_test.csv`.

## Modelo elegido

Se eligió **Regresión Logística** como baseline porque:

- funciona muy bien con matrices TF-IDF dispersas;
- es rápida de entrenar;
- entrega resultados sólidos en clasificación multiclase;
- es más interpretable y estable que modelos más complejos para una primera línea base.

Este modelo permite validar rápidamente si el preprocesamiento y la vectorización producen señal suficiente antes de pasar a modelos profundos.

## Cómo ejecutar

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar el pipeline:

```bash
python src/train_tfidf_classifier.py
```

Opcionalmente se pueden modificar parámetros:

```bash
python src/train_tfidf_classifier.py --max-features 12000 --ngram-max 2 --c-value 2.0
```

## Resultados obtenidos

Con la configuración base se obtuvo:

- Accuracy en test: **0.8955**.
- F1-score macro: **0.8955**.
- Tamaño del vocabulario TF-IDF: **12.000 términos**.

Resumen por clase:

| Clase | Precision | Recall | F1-score |
|---|---:|---:|---:|
| World | 0.90 | 0.89 | 0.89 |
| Sports | 0.96 | 0.96 | 0.96 |
| Business | 0.85 | 0.86 | 0.85 |
| Sci_Tech | 0.87 | 0.87 | 0.87 |

El script genera:

- `outputs/classification_report.txt`: precision, recall y F1-score por clase.
- `outputs/confusion_matrix.png`: matriz de confusión multiclase.
- `outputs/metrics.json`: métricas en formato estructurado.
- `models/tfidf_logistic_regression.joblib`: pipeline entrenado.

## Análisis preliminar

El baseline alcanza un desempeño alto para una primera aproximación clásica con TF-IDF. La clase más fácil de identificar es `Sports`, con F1-score de 0.96, probablemente porque posee vocabulario deportivo más específico.

Las clases más difíciles son `Business` y `Sci_Tech`, con F1-score de 0.85 y 0.87 respectivamente. Esto es esperable porque muchas noticias tecnológicas mencionan empresas, mercados, productos y resultados financieros, generando solapamiento semántico con noticias de negocios. También puede haber cruces entre `World` y `Business` cuando una noticia económica está vinculada con decisiones políticas o eventos internacionales.

## Reproducibilidad

El script fija `random_state=42` en el clasificador. Además, los splits ya vienen separados por la cátedra, por lo que el experimento puede repetirse sin cambiar la partición de datos.
