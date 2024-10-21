import os
from metaflow import FlowSpec, step, S3

# Configuración de las credenciales de acceso a AWS S3 (minio)
os.environ['AWS_ACCESS_KEY_ID'] = "minio"
os.environ['AWS_SECRET_ACCESS_KEY'] = "minio123"
os.environ['AWS_ENDPOINT_URL_S3'] = "http://localhost:9000"


class BatchProcessingModel(FlowSpec):

    @step
    def start(self):
        """
        Step para iniciar el flujo. Imprime un mensaje de inicio y avanza.
        """
        print("Starting Batch Prediction")
        self.next(self.load_data, self.load_model)

    @step
    def load_data(self):
        """
        Paso para cargar los datos de entrada de S3
        """
        import pandas as pd

        # Se utiliza el objeto S3 para acceder a los datos desde el bucket en S3.
        s3 = S3(s3root="s3://batch/")
        data_obj = s3.get("data/X_test.csv")
        self.X_batch = pd.read_csv(data_obj.path)
        self.next(self.batch_processing)

    @step
    def load_model(self):
        """
        Paso para cargar el modelo de DecisionTreeClassifier previamente entrenado desde S3.
        """
        import joblib
        from metaflow import S3

        # Se utiliza el objeto S3 para acceder al modelo desde el bucket en S3.
        s3 = S3(s3root="s3://batch/")
        model_param = s3.get("artifact/model.pkl")

        # Cargamos el modelo desde el archivo .pkl
        with open(model_param.path, 'rb') as model_file:
            loaded_model = joblib.load(model_file)

        self.model = loaded_model
        self.next(self.batch_processing)

    @step
    def batch_processing(self, previous_tasks):
        """
        Paso para realizar el procesamiento por lotes y obtener predicciones.
        """
        import numpy as np
        import hashlib

        print("Obtaining predictions")

        model = None
        data = None

        # Se recorren las tareas previas para obtener los datos y el modelo.
        for task in previous_tasks:
            if hasattr(task, 'X_batch'):
                data = task.X_batch
            if hasattr(task, 'model'):
                model = task.model

        # Se obtienen las predicciones utilizando el modelo.
        out = model.predict(data)

        # Se define un diccionario de mapeo
        label_map = {0: "no churned", 1: "churned"}

        # Y obtenemos la salida del modelo en modo de string. Esto podríamos haberlo implementado directamente en
        # la lógica del modelo
        labels = np.array([label_map[idx] for idx in out])

        # Convertimos labels en un array 2D para poder concatenarlo
        labels_reshaped = labels.reshape(-1, 1)

        # Concatenamos X_batch (datos) con labels (etiquetas) horizontalmente
        self.all_data = np.hstack([data, labels_reshaped])

        self.next(self.save_data)

    @step
    def save_data(self):
        """
        Paso para guardar los resultados como un archivo CSV en S3.
        """
        import pandas as pd
        import redis
        from io import StringIO

        # Convertimos all_data a un DataFrame
        df = pd.DataFrame(self.all_data, columns=['Age', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction', 'Gender_Female', 'Contract Length_Annual', 'Contract Length_Monthly', 'Prediction'])

        # Guardamos el DataFrame en un buffer como CSV
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)

        # Usamos S3 para subir el archivo CSV al bucket
        s3 = S3(s3root="s3://batch/")
        s3.put("predictions/predictions.csv", csv_buffer.getvalue())

        print("Data saved to S3 as predictions.csv")

        # Almacenar en Redis
        r = redis.StrictRedis(host='localhost', port=6379, db=0)

        # Guardar cada fila en Redis
        for idx, row in df.iterrows():
            r.set(f"prediction:{idx}", row.to_json())

        print("Data saved to S3 and Redis")
        self.next(self.end)

    @step
    def end(self):
        """
        Paso final del flujo. Imprime un mensaje de finalización.
        """
        print("Finished")


if __name__ == "__main__":
    BatchProcessingModel()