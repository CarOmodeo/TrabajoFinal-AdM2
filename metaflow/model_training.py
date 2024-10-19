import os
from metaflow import FlowSpec, step, S3

# Configuración de las credenciales de acceso a AWS S3 (minio)
os.environ['AWS_ACCESS_KEY_ID'] = "minio"
os.environ['AWS_SECRET_ACCESS_KEY'] = "minio123"
os.environ['AWS_ENDPOINT_URL_S3'] = "http://localhost:9000"


class TrainModel(FlowSpec):

    @step
    def start(self):
        """
        Step para iniciar el flujo. Imprime un mensaje de inicio y avanza.
        """
        print("Starting Model Training")
        self.next(self.load_data)

    @step
    def load_data(self):
        """
        Paso para cargar los datos de entrada de S3
        """
        import pandas as pd

        # Se utiliza el objeto S3 para acceder a los datos desde el bucket en S3.
        s3 = S3(s3root="s3://batch/")
        data_obj_train = s3.get("data/X_train.csv")
        data_obj_y = s3.get("data/y_train.csv")
        self.X_train = pd.read_csv(data_obj_train.path)
        self.y_train = pd.read_csv(data_obj_y.path)
        self.next(self.train_model)

    @step
    def train_model(self):
        """
        Paso para realizar entrenamiento del modelo.
        """
        from sklearn.model_selection import cross_val_score
        from sklearn.tree import DecisionTreeClassifier

        import optuna

        # ------------ Optuna ------------------

        def objective(trial):
            max_depth = trial.suggest_int("max_depth", 1, 30)
            criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)

            classifier = DecisionTreeClassifier(criterion=criterion, splitter='best',
                                                max_depth=max_depth, min_samples_split=min_samples_split,
                                                min_samples_leaf=min_samples_leaf, random_state=42)

            # Realizamos la validación cruzada
            score = cross_val_score(classifier, self.X_train, self.y_train, cv=5, scoring='f1', n_jobs=-1)

            return score.mean()

        def champion_callback(study, frozen_trial):
            """
            Mostramos menos información, sino es demasiado verboso
            """
            winner = study.user_attrs.get("winner", None)
            if study.best_value and winner != study.best_value:
                study.set_user_attr("winner", study.best_value)
                if winner:
                    improvement_percent = (abs(winner - study.best_value) / study.best_value) * 100
                    print(
                        f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
                        f"{improvement_percent: .4f}% improvement"
                    )
                else:
                    print(f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}")

        # ----------------------------------------------------------------

        # Se crea un estudio de Optuna

        optuna.logging.set_verbosity(optuna.logging.ERROR)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=10, callbacks=[champion_callback])

        # ------------ Modelo Arbol optimizado ------------------
        self.tree_classification = DecisionTreeClassifier(**study.best_params, random_state=42)

        # Y ese entrena el modelo
        self.tree_classification.fit(self.X_train, self.y_train)

        self.next(self.save_model)

    @step
    def save_model(self):
        """
        Paso para guardar el modelo entrenado como un archivo .pkl en S3.
        """
        import joblib
        from io import BytesIO

        # Convertimos el modelo a bytes usando un buffer
        model_buffer = BytesIO()
        joblib.dump(self.tree_classification, model_buffer)
        model_buffer.seek(0)  # Volvemos al inicio del buffer

        # Subimos el archivo al bucket S3
        s3 = S3(s3root="s3://batch/")
        s3.put("artifact/model.pkl", model_buffer)

        print("Modelo guardado en S3 como tree_classification_model.pkl")
        self.next(self.end)

    @step
    def end(self):
        """
        Paso final del flujo. Imprime un mensaje de finalización.
        """
        print("Finished")


if __name__ == "__main__":
    TrainModel()