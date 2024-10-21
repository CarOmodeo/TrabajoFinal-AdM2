# TP Final Aprendizaje de Máquina II - Tasa de abandono (Churn)

## Integrantes

- Carolina Perez Omodeo
- Fausto Juárez Yélamos
- Emiliano Uriel Martino
- Juan Pablo Hagata
- Santiago Francisco Belen.

## Desarrollo

Este repositorio contiene el trabajo práctico final de la materia Aprendizaje de Máquina II de la Especialización en Inteligencia Artificial (CEIA) de la Facultad de Ingeniería de la Universidad de Buenos Aires (FIUBA). El objetivo es implementar un modelo productivo para predecir clientes con mayor probabilidad de darse de baja (churn) del servicio proporcionado por la empresa.
El dataset utilziado es el de [Customer Churn Dataset (Kaggle)](https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset).

Nuestra implementación incluye:

- En **Metaflow**, tres flujos distintos que orquestan las tareas de extracción/procesamiento, entrenamiento de un árbol de decisión optimizado con Optuna y predicción por lotes de datos. Estos flujos son:

  - **Data Engineering (data_engineering.py)**: Para la extracción y procesamiento inicial de los datos. Este flujo se encarga de cargar los datos crudos desde S3, limpiarlos, realizar transformaciones necesarias y guardar los datos procesados nuevamente en S3.
  - **Model Training (model_training.py)**: Para el entrenamiento del modelo de machine learning. Utiliza Optuna para la búsqueda de hiperparámetros, entrena un modelo de clasificación y guarda el modelo entrenado en S3.
  - **Batch Processing (batch_processing.py)**: Para realizar predicciones por lotes utilizando el modelo entrenado. Carga los datos de prueba desde S3, realiza las predicciones y guarda los resultados en S3.

- Un servicio **MinIO** desplegado en Docker que proporciona una alternativa compatible con S3 para gestionar el almacenamiento de objetos. El servicio permite manejar buckets y objetos de manera similar a Amazon S3, brindando una solución local y flexible para almacenar y recuperar datos. La configuración del contenedor y los volúmenes persistentes se encuentran definidos en el archivo **docker-compose.yml**, facilitando la implementación

- Una Jupyter notebook para visualizar las predicciónes realizadas por el modelo.

## Instrucciones para levantar el proyecto

1. Instalar [Docker](https://www.docker.com/

2. Clonar el repositorio y moverse al directorio raíz del proyecto.

3. En la carpeta raíz del proyecto, ejecutar el siguiente comando para levantar el multi-contenedor:

```bash
docker-compose up
```

## Ver el estado de los contenedores

Para ver el estado de los contenedores, ejecutar el siguiente comando:

```bash
docker-compose ps
```

## Detener los contenedores

Para detener los contenedores, ejecutar el siguiente comando:

```bash
docker-compose down
```

## Instrucciones para probar el funcionamiento del proyecto utilizando WSL (Windows Subsystem for Linux)

Una vez levantado el multi-contenedor, es necesario utilizar WSL (Windows subsystem for linux) dado que Metaflow tiene alguntas limitaciones de compatibilidiad con windows.

1.  En la carpeta raíz del proyecto, ejecutar el siguiente comando para ingresar a WSL

```bash
wsl
```

2. Crear un entorno virtual en Python

```bash
python3 -m venv Churn
```

3. Activar entorno virtual

```bash
source Churn/bin/activate
```

4. Instalar dependencias:

```bash
pip install -r requirements.txt
```

5. Ejecutar script de forma secuencial.

```bash
python3 metaflow/data_engineering.py run
```

```bash
python3 metaflow/model_training.py run
```

```bash
python3 metaflow/bach_processing.py run
```

## Visualización de las predicciónes

Para visualizar las predicciónes ejecutar la notebook predictions_out.ipynb.
