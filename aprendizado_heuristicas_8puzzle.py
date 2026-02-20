from collections import deque
import numpy as np
from scipy.signal import convolve2d as conv2
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import random
import matplotlib.pyplot as plt

class Estado:
    def __init__(self, pai=None, matriz=None):
        self.pai = pai
        self.matriz = matriz
        self.d = 0
        self.p = 0

    def __eq__(self, other):
        return np.array_equal(self.matriz, other.matriz)

    def __hash__(self):
        return hash(tuple(self.matriz.flatten()))

    def __lt__(self, other):
        return self.p < other.p

    def mostrar(self):
        for linha in self.matriz:
            for elemento in linha:
                print(" " if elemento == 9 else elemento, end=" ")
            print()
        print()

def acoes_permitidas(estado):
    adj = np.array([[0,1,0],[1,0,1],[0,1,0]])
    blank = estado.matriz == 9
    mask = conv2(blank, adj, 'same')
    return estado.matriz[np.where(mask)]


def movimentar(s, c):
    matriz = s.matriz.copy()
    matriz[np.where(s.matriz == 9)] = c
    matriz[np.where(s.matriz == c)] = 9
    return Estado(matriz=matriz)


def bfs_reverse(goal_state):
    dist = {goal_state: 0}
    queue = deque([goal_state])

    while queue:
        state = queue.popleft()
        d = dist[state]

        for c in acoes_permitidas(state):
            next_state = movimentar(state, c)
            if next_state not in dist:
                dist[next_state] = d + 1
                queue.append(next_state)

    return dist

def heuristica_manhattan(matriz_atual):
    distancia = 0
    for i in range(3):
        for j in range(3):
            valor = matriz_atual[i][j]
            if valor != 9:
                alvo_x = (valor - 1) // 3
                alvo_y = (valor - 1) % 3
                distancia += abs(i - alvo_x) + abs(j - alvo_y)
    return distancia


def heuristica_hamming(matriz_atual, matriz_objetivo):
    diferentes = matriz_atual != matriz_objetivo
    contagem = np.sum(diferentes)
    loc_vazio = np.where(matriz_atual == 9)
    if matriz_objetivo[loc_vazio] != 9:
        contagem -= 1
    return contagem


def extrair_features(estado, objetivo):
    f1 = heuristica_manhattan(estado.matriz)
    f2 = heuristica_hamming(estado.matriz, objetivo.matriz)
    return [f1, f2]

if __name__ == "__main__":

    # Definir estado objetivo
    matriz_obj = np.array([[1,2,3],[4,5,6],[7,8,9]])
    target = Estado(matriz=matriz_obj)

    print("Gerando mapa de distâncias reais (BFS)...")
    dist = bfs_reverse(target)
    print(f"Total de estados mapeados: {len(dist)}")

    # Construir dataset
    X = []
    y = []

    estados_lista = list(dist.keys())

    for estado in estados_lista:
        X.append(extrair_features(estado, target))
        y.append(dist[estado])

    X = np.array(X)
    y = np.array(y)

    # Treinar regressão linear
    print("\nTreinando modelo...")
    modelo = LinearRegression()
    modelo.fit(X, y)

    w0 = modelo.intercept_
    w_manhattan = modelo.coef_[0]
    w_hamming = modelo.coef_[1]

    print("\nFórmula aprendida:")
    print(f"h(n) = {w0:.4f} + ({w_manhattan:.4f} * Manhattan) + ({w_hamming:.4f} * Hamming)")

    # Avaliação do modelo
    y_pred = modelo.predict(X)

    mae_modelo = mean_absolute_error(y, y_pred)
    mse_modelo = mean_squared_error(y, y_pred)

    print("\n--- Avaliação do Modelo ---")
    print(f"MAE Modelo: {mae_modelo:.4f}")
    print(f"MSE Modelo: {mse_modelo:.4f}")

    # Avaliar Manhattan isolada
    y_manhattan = np.array([heuristica_manhattan(e.matriz) for e in estados_lista])
    mae_manhattan = mean_absolute_error(y, y_manhattan)

    # Avaliar Hamming isolada
    y_hamming = np.array([heuristica_hamming(e.matriz, target.matriz) for e in estados_lista])
    mae_hamming = mean_absolute_error(y, y_hamming)

    print("\n--- Comparação de Heurísticas (MAE) ---")
    print(f"MAE Manhattan: {mae_manhattan:.4f}")
    print(f"MAE Hamming: {mae_hamming:.4f}")
    print(f"MAE Modelo Aprendido: {mae_modelo:.4f}")

    # Verificar admissibilidade
    superestimativas = np.sum(y_pred > y)
    total = len(y)

    print("\n--- Admissibilidade ---")
    print(f"Estados com superestimação: {superestimativas} de {total}")
    print(f"Percentual: {(superestimativas/total)*100:.2f}%")

    # Gráfico comparativo
    plt.figure()
    plt.bar(["Manhattan", "Hamming", "Modelo"],
            [mae_manhattan, mae_hamming, mae_modelo])
    plt.ylabel("Erro Médio Absoluto (MAE)")
    plt.title("Comparação das Heurísticas")
    plt.show()

    # Teste com estado aleatório
    print("\nTeste com estado aleatório:")
    estado_teste = random.choice(estados_lista)
    estado_teste.mostrar()

    feats_teste = extrair_features(estado_teste, target)
    predicao = modelo.predict([feats_teste])[0]
    real = dist[estado_teste]

    print(f"Distância Real: {real}")
    print(f"Distância Prevista: {predicao:.4f}")
    print(f"Erro: {abs(real - predicao):.4f}")