import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.model_selection import validation_curve
import numpy as np
from mlflow.tracking import MlflowClient
import mlflow


def check_df(df, file_path):
    with open(file_path, 'w') as file:
        file.write(str(df.describe(include='all') + '\n\n'))

        print("Dados faltantes\n")
        file.write(str(df.isnull().sum()))

        print("Recomendações")
        categorical_columns = df.select_dtypes(include=['object', 'categorical']).columns
        for col in categorical_columns:
            unique = df[col].unique()
            encoding_type = "One-Hot Encoding" if len(unique) <= 10 else "Label Encoding"
            file.write("Coluna: {col} - Codificação Recomendada: {encoding_type}"+"\n")
            file.write("Valores Únicos: {unique}"+"\n")