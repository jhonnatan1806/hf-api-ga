from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import random
from typing import List

class InputData(BaseModel):
    data: List[float]  # Lista de características numéricas (flotantes)
    data2: List[float]  # Lista de características numéricas (flotantes)
    
app = FastAPI()

# ------------- algoritmo genetico -------------
# Función para generar una población inicial aleatoria
def generar_poblacion(num_individuos, num_ciudades):
    poblacion = []
    for _ in range(num_individuos):
        individuo = list(range(num_ciudades))
        random.shuffle(individuo)
        poblacion.append(individuo)
    return poblacion

# Función para evaluar la aptitud de un individuo (distancia total del recorrido)
def calcular_aptitud(individuo, distancias, coordenadas):
    distancia_total = 0
    coordenadas_iguales = all(coord == coordenadas[0] for coord in coordenadas)

    if not coordenadas_iguales:
        for i in range(len(individuo) - 1):
            ciudad_actual = individuo[i]
            siguiente_ciudad = individuo[i + 1]
            distancia_total += distancias[ciudad_actual][siguiente_ciudad]

        distancia_total += distancias[individuo[-1]][individuo[0]]

    return distancia_total

# Función para seleccionar individuos para la reproducción (torneo binario)
def seleccion_torneo(poblacion, distancias, coordenadas):
    seleccionados = []
    for _ in range(len(poblacion)):
        torneo = random.sample(poblacion, 2)
        aptitud_torneo = [
            calcular_aptitud(individuo, distancias, coordenadas) for individuo in torneo
        ]
        seleccionado = torneo[aptitud_torneo.index(min(aptitud_torneo))]
        seleccionados.append(seleccionado)
    return seleccionados

# Función para realizar el cruce de dos padres para producir un hijo
def cruzar(padre1, padre2):
    punto_cruce = random.randint(0, len(padre1) - 1)
    hijo = padre1[:punto_cruce] + [
        gen for gen in padre2 if gen not in padre1[:punto_cruce]
    ]
    return hijo


# Función para aplicar mutaciones en la población
def mutar(individuo, probabilidad_mutacion):
    if random.random() < probabilidad_mutacion:
        indices = random.sample(range(len(individuo)), 2)
        individuo[indices[0]], individuo[indices[1]] = (
            individuo[indices[1]],
            individuo[indices[0]],
        )
    return individuo

# Función para generar distancias aleatorias entre ciudades y sus coordenadas bidimensionales
def generar_distancias(num_ciudades):
    distancias = [[0] * num_ciudades for _ in range(num_ciudades)]
    coordenadas = [
        (random.uniform(0, 100), random.uniform(0, 100)) for _ in range(num_ciudades)
    ]

    for i in range(num_ciudades):
        for j in range(i + 1, num_ciudades):
            distancias[i][j] = distancias[j][i] = (
                sum((x - y) ** 2 for x, y in zip(coordenadas[i], coordenadas[j])) ** 0.5
            )

    return distancias, coordenadas
# Función para generar distancias aleatorias entre ciudades y sus coordenadas bidimensionales
def generar_distanaias_coordenadas(listapuntos):
    num_ciudades = int(max(listapuntos.shape)/2)
    distancias = [[0] * (num_ciudades) for _ in range(num_ciudades)]
    coordenadas = [
        (listapuntos[i], listapuntos[i+1]) for i in range(0,2*num_ciudades-1,2)
    ]
    for i in range(num_ciudades):
        for j in range(i + 1, num_ciudades):
            distancias[i][j] = distancias[j][i] = (
                sum((x - y) ** 2 for x, y in zip(coordenadas[i], coordenadas[j])) ** 0.5
            )
    return distancias, coordenadas
    
def algoritmo_genetico(num_generaciones,num_ciudades,num_individuos,probabilidad_mutacion,distancias,coordenadas):
    poblacion = generar_poblacion(num_individuos, num_ciudades)    
    for generacion in range(num_generaciones):
        poblacion = sorted(
            poblacion, key=lambda x: calcular_aptitud(x, distancias, coordenadas)
        )
        mejor_individuo = poblacion[0]
        mejor_distancia = calcular_aptitud(mejor_individuo, distancias, coordenadas)        
        seleccionados = seleccion_torneo(poblacion, distancias, coordenadas)
        nueva_poblacion = []
        for i in range(0, len(seleccionados), 2):
            padre1, padre2 = seleccionados[i], seleccionados[i + 1]
            hijo1 = cruzar(padre1, padre2)
            hijo2 = cruzar(padre2, padre1)
            hijo1 = mutar(hijo1, probabilidad_mutacion)
            hijo2 = mutar(hijo2, probabilidad_mutacion)
            nueva_poblacion.extend([hijo1, hijo2])
        poblacion = nueva_poblacion        
    mejor_solucion = poblacion[0]
    mejor_distancia = calcular_aptitud(mejor_solucion, distancias, coordenadas)
    return mejor_solucion, mejor_distancia

# Ruta de predicción
@app.post("/predict/")
async def predict(data: InputData):
    try:
        input_data = np.array(data.data).reshape(
            1, -1
        )  # Asumiendo que la entrada debe ser de forma (1, num_features)
        
        num_ciudades = int(input_data[0][0])
        num_individuos = int(input_data[0][1])
        probabilidad_mutacion = float(input_data[0][2])
        num_generaciones = int(input_data[0][3])
        distancias, coordenadas = generar_distanaias_coordenadas(np.array(data.data2))
        mejor_solucion, mejor_distancia = algoritmo_genetico(num_generaciones,num_ciudades,num_individuos,probabilidad_mutacion,distancias,coordenadas)
        mejor_solucion.append(int(mejor_distancia))
        
        prediction = mejor_solucion
        #return {"prediction": prediction.tolist()}
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
