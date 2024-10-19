import os
from metaflow import FlowSpec, step, S3

# Configuración de las credenciales de acceso a AWS S3 (minio)
os.environ['AWS_ACCESS_KEY_ID'] = "minio"
os.environ['AWS_SECRET_ACCESS_KEY'] = "minio123"
os.environ['AWS_ENDPOINT_URL_S3'] = "http://localhost:9000"


class DataProcessingModel(FlowSpec):

    @step
    def start(self):
        """
        Step para iniciar el flujo. Imprime un mensaje de inicio y avanza.
        """
        print("Starting Data Engineering")
        self.next(self.load_data)

    @step
    def load_data(self):
        """
        Paso para cargar los datos de entrada de S3
        """
        import pandas as pd

        # Se utiliza el objeto S3 para acceder a los datos desde el bucket en S3.
        s3 = S3(s3root="s3://batch/")
        data_obj_train = s3.get("raw/training.csv")
        data_obj_test = s3.get("raw/testing.csv")
        self.customer_train = pd.read_csv(data_obj_train.path)
        self.customer_test = pd.read_csv(data_obj_test.path)
        self.next(self.data_processing)

    @step
    def data_processing(self):
        """
        Paso para realizar el procesamiento de los datos.
        """
        import pandas as pd
        from sklearn.preprocessing import StandardScaler # Reescalamiento de variables

        # ---------------- Remuestreo aleatorio ---------------------
        customer_all = pd.concat([self.customer_train, self.customer_test], axis=0)

        train_sample = customer_all['CustomerID'].sample(n=int(customer_all.shape[0]*0.8), random_state=42)
        customer_train = customer_all[customer_all['CustomerID'].isin(train_sample)]
        customer_test = customer_all[~customer_all['CustomerID'].isin(train_sample)]

        # ---------------- Eliminar nulos ---------------------
        customer_train = customer_train.dropna()

        # ---------------- Transformacion a Dummy  -------------
        # Las variables categóricas es recomendable pasarlas a variables *dummy*, ya que esto permite que los modelos de ML procesen la información categórica de manera efectiva y sin inducir un orden o relación no deseada entre las categorías.

        categorical_features = ['Gender','Subscription Type','Contract Length']

        customer_train = pd.get_dummies(data=customer_train,
                                        columns=categorical_features)

        customer_test = pd.get_dummies(data=customer_test,
                                        columns=categorical_features)

        # ------------ Variables a utilizar ------------------
        customer_train = customer_train.drop(['CustomerID','Tenure','Usage Frequency','Subscription Type_Basic','Subscription Type_Premium','Subscription Type_Standard'], axis=1)
        customer_test = customer_test.drop(['CustomerID','Tenure','Usage Frequency','Subscription Type_Basic','Subscription Type_Premium','Subscription Type_Standard'], axis=1)
        

        # ------------ División de datos ------------------
        # Separamos las variables objetivos de las predictoras y eliminamos una de las dummy (en cada categoría que tenga), para evitar correlación.
        self.X_train = customer_train.loc[:, ['Age', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction', 'Gender_Female', 'Contract Length_Annual', 'Contract Length_Monthly']].values
        self.y_train = customer_train.loc[:, 'Churn'].values

        # ------------ Set de testeo        ------------------
        self.X_test = customer_test.loc[:, ['Age', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction', 'Gender_Female', 'Contract Length_Annual', 'Contract Length_Monthly']].values
        self.y_test = customer_test.loc[:, 'Churn'].values

        # ------------ Escalado  ------------------
        # sc_X = StandardScaler()

        # self.X_train = sc_X.fit_transform(X_train)
        # self.X_test = sc_X.transform(X_test)

        self.next(self.save_data)


    @step
    def save_data(self):
        """
        Paso para guardar los datos procesados en formato CSV en S3.
        """
        import pandas as pd
        from io import StringIO
        
        # Crear DataFrames a partir de los arrays
        df_X_train = pd.DataFrame(self.X_train, columns=['Age', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction', 'Gender_Female', 'Contract Length_Annual', 'Contract Length_Monthly'])
        df_y_train = pd.DataFrame(self.y_train, columns=['Churn'])
        df_X_test = pd.DataFrame(self.X_test, columns=['Age', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction', 'Gender_Female', 'Contract Length_Annual', 'Contract Length_Monthly'])
        df_y_test = pd.DataFrame(self.y_test, columns=['Churn'])

        # Guardar los DataFrames en CSV en S3 usando el objeto S3 de Metaflow
        with S3(s3root="s3://batch/") as s3:
            # Guardar X_train
            csv_buffer_X_train = StringIO()
            df_X_train.to_csv(csv_buffer_X_train, index=False)
            s3.put("data/X_train.csv", csv_buffer_X_train.getvalue())

            # Guardar y_train
            csv_buffer_y_train = StringIO()
            df_y_train.to_csv(csv_buffer_y_train, index=False)
            s3.put("data/y_train.csv", csv_buffer_y_train.getvalue())

            # Guardar X_test
            csv_buffer_X_test = StringIO()
            df_X_test.to_csv(csv_buffer_X_test, index=False)
            s3.put("data/X_test.csv", csv_buffer_X_test.getvalue())

            # Guardar y_test
            csv_buffer_y_test = StringIO()
            df_y_test.to_csv(csv_buffer_y_test, index=False)
            s3.put("data/y_test.csv", csv_buffer_y_test.getvalue())

        print("Datos guardados en S3.")

        self.next(self.end)

    @step
    def end(self):
        """
        Paso final del flujo. Imprime un mensaje de finalización.
        """
        print("Finished")


if __name__ == "__main__":
    DataProcessingModel()