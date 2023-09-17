from typing import Union
from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
modelo_dt = joblib.load(r'ml-base-pipeline\notebooks\atividades\modelo_knn.pk1')

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.get("/dados/{dados_enviados}")
def read_dados(interest: int, parent_salary: float, house_area: float, average_grades: float):
    vetor = [[interest, parent_salary, house_area, average_grades]]
    df = pd.DataFrame(vetor, columns=['interest', 'parent_salary', 'house_area', 'average_grades'])
    pred = modelo_dt.predict(vetor)
    
    if pred == 0:
        msg = " SINTO MUITO, MAS VOCÊ NÃO TEM MUITAS CHANCES DE ENTRAR NUMA FACULDADE"
    else:
        msg = " PARABÉNS !!! VOCÊ TEM GRANDES CHANCES DE ENTRAR NUMA FACULDADE"
    
    return {"RESPOSTA ": msg}