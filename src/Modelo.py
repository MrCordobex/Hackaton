import json
import joblib
import sys
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def cargar_modelo(ruta_modelo: str):
    rf_comentario = joblib.load("../models/modelo_random_forest.pkl")
    vectorizer_comentario = joblib.load(f"../models/vectorizer_tfidf.pkl")


    return rf_comentario, vectorizer_comentario


def predecir(comentario: str, modelo: str, modelo_vector: str):

    vec = modelo_vector.transform([comentario.lower()])
    prediccion_clavero= modelo.predict(vec)[0]


    return {prediccion_clavero}


mi_modelo, mi_vectorizer=cargar_modelo("../models/modelo_random_forest.pkl")
resultado = predecir("Este producto es excelente y lo recomiendo a todos", mi_modelo, mi_vectorizer)
print(resultado)
