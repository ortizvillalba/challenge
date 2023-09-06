import os
import pandas as pd

from typing import Tuple, Union, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report

class DelayModel:

     # Definir FEATURES_COLS como una lista de nombres de columnas esperadas
    FEATURES_COLS = [
        'OPERA_Latin American Wings', 'MES_7', 'MES_10', 'OPERA_Grupo LATAM', 'MES_12', 'TIPOVUELO_I',  # Agrega aquí todas las columnas
        # ...
    ]

    def __init__(
        self
    ):
        self._model = None # Model should be saved in this attribute.

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        # Asegúrate de que features tenga las columnas esperadas
        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix='OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'),
            pd.get_dummies(data['MES'], prefix='MES')],
            axis=1
        )

        # Verifica si las columnas son iguales a self.FEATURES_COLS
        assert set(features.columns) == set(self.FEATURES_COLS)
         # Intenta convertir la columna 'Fecha-I' en datetime y maneja errores
        try:
            data['Fecha-I'] = pd.to_datetime(data['Fecha-I'], errors='coerce')  # 'coerce' para manejar errores
        except pd.errors.ParserError:
            # Maneja las filas con valores no válidos en 'Fecha-I' aquí
            pass

        # Filtra las filas donde 'Fecha-I' no se pudo convertir a datetime
        data = data[~data['Fecha-I'].isnull()]

        data['min_diff'] = data.apply(self.get_min_diff, axis=1)
        data['period_day'] = data.apply(self.get_period_day, axis=1)  # Modificación aquí

        data['high_season'] = data.apply(self.is_high_season, axis=1)  # Aplica la función a la fila completa

        threshold_in_minutes = 15
        data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)

        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix='OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'),
            pd.get_dummies(data['MES'], prefix='MES')],
            axis=1
        )
        target = data['delay']

        # Después de generar todas las columnas necesarias en 'features'
        # Asegúrate de que todas las columnas estén presentes y en el mismo orden
        required_columns = self.FEATURES_COLS
        missing_columns = [col for col in required_columns if col not in features.columns]

        if missing_columns:
            # Si faltan columnas, agrégalas al DataFrame 'features'
            for col in missing_columns:
                features[col] = 0  # Puedes inicializar las columnas faltantes con un valor predeterminado


        return features, target

    

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """

         # Inicializa el modelo (en este caso, un modelo de Regresión Logística)
        self._model = LogisticRegression()
        
        # Entrena el modelo con los datos proporcionados
        

        # training_data = shuffle(data[['OPERA', 'MES', 'TIPOVUELO', 'SIGLADES', 'DIANOM']], random_state = 111)
        training_data = shuffle(pd.concat([features, target], axis=1), random_state=111)


        x_train, x_test, y_train, y_test = train_test_split(features, target, test_size = 0.33, random_state = 42)

        y_train.value_counts('%')*100

        y_test.value_counts('%')*100

        self._model.fit(x_train, y_train)
        

        return x_train, x_test, y_train, y_test

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """

        # Asegúrate de que el modelo esté entrenado antes de hacer predicciones
        if self._model is None:
            raise ValueError("El modelo aún no ha sido entrenado. Llame primero al método 'fit' o asegúrese de que el modelo esté cargado.")

        # Realiza las predicciones utilizando el modelo
        predictions = self._model.predict(features)
                
        return predictions


    def get_period_day(self, row):
        fecha_i = row['Fecha-I']
        
        morning_min = pd.Timestamp("05:00").time()
        morning_max = pd.Timestamp("11:59").time()
        afternoon_min = pd.Timestamp("12:00").time()
        afternoon_max = pd.Timestamp("18:59").time()
        evening_min = pd.Timestamp("19:00").time()
        evening_max = pd.Timestamp("23:59").time()
        night_min = pd.Timestamp("00:00").time()
        night_max = pd.Timestamp("04:59").time()

        if (morning_min <= fecha_i.time() <= morning_max):
            return 'mañana'
        elif (afternoon_min <= fecha_i.time() <= afternoon_max):
            return 'tarde'
        elif (
            (evening_min <= fecha_i.time() <= evening_max) or
            (night_min <= fecha_i.time() <= night_max)
        ):
            return 'noche'

        
    def is_high_season(self, row):
        fecha = row['Fecha-I']
        # Verifica si 'Fecha-I' es un objeto datetime antes de acceder al año
        if isinstance(fecha, pd.Timestamp):
            fecha_año = fecha.year
            range1_min = datetime.strptime('15-Dec', '%d-%b').replace(year=fecha_año)
            range1_max = datetime.strptime('31-Dec', '%d-%b').replace(year=fecha_año)
            range2_min = datetime.strptime('1-Jan', '%d-%b').replace(year=fecha_año)
            range2_max = datetime.strptime('3-Mar', '%d-%b').replace(year=fecha_año)
            range3_min = datetime.strptime('15-Jul', '%d-%b').replace(year=fecha_año)
            range3_max = datetime.strptime('31-Jul', '%d-%b').replace(year=fecha_año)
            range4_min = datetime.strptime('11-Sep', '%d-%b').replace(year=fecha_año)
            range4_max = datetime.strptime('30-Sep', '%d-%b').replace(year=fecha_año)

            if ((fecha >= range1_min and fecha <= range1_max) or
                    (fecha >= range2_min and fecha <= range2_max) or
                    (fecha >= range3_min and fecha <= range3_max) or
                    (fecha >= range4_min and fecha <= range4_max)):
                return 1
        return 0


        
    def get_min_diff(self, row):
        fecha_o = datetime.strptime(row['Fecha-O'], '%Y-%m-%d %H:%M:%S')
        fecha_i = row['Fecha-I']  
        min_diff = ((fecha_o - fecha_i).total_seconds()) / 60
        return min_diff



    def get_rate_from_column(self, data, column): 
        delays = {}
        for _, row in data.iterrows():
            if row['delay'] == 1:
                if row[column] not in delays:
                    delays[row[column]] = 1
                else:
                    delays[row[column]] += 1
        total = data[column].value_counts().to_dict()

        rates = {}
        for name, total in total.items():
            if name in delays:
                rates[name] = round(total / delays[name], 2)
            else:
                rates[name] = 0

        return pd.DataFrame.from_dict(data=rates, orient='index', columns=['Tasa (%)'])



# Utilizar la clase DelayModel:
# if __name__ == "__main__":
#     # Carga los datos y realiza el preprocesamiento si es necesario
#     print("leemos el dato...")

#     # Obtiene la ruta del directorio actual del script
#     script_dir = os.path.dirname(__file__)

#     # Construye la ruta completa al archivo "data.csv" dentro del directorio "data"
#     data_file = os.path.join(script_dir, '../data', 'data.csv')

#     # Luego carga el archivo
#     data = pd.read_csv(data_file)
#     #data = pd.read_csv('data.csv')

#     target_column = 'OPERA'  # Reemplaza 'target' con el nombre real de la columna de destino
#     # Dentro de la función __main__ donde usas la clase DelayModel

#     model = DelayModel()

#     # Preprocesa los datos y obtén las características y el objetivo
#     features, target = model.preprocess(data, target_column)
#     print("procesamos los dato...")

#     # Entrena el modelo con los datos preprocesados
#     x_train2, x_test2, y_train2, y_test2 = model.fit(features, target)

#     # Realiza predicciones en nuevos datos (features_new)
#     # features_new = pd.DataFrame(...)  # Reemplaza ... con los datos de entrada
#     predictions = model.predict(features)





