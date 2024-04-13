import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

def executar_modelo(iteracoes, camadas):
  """
  Treina e avalia um modelo MLPRegressor com os parâmetros especificados.

  Args:
    iteracoes: Número de iterações de treinamento.
    camadas: Tamanho das camadas ocultas (tupla de inteiros).

  Retorna:
    Tupla contendo:
      - Média das médias do desempenho do modelo.
      - Média dos desvios padrões do desempenho do modelo.
  """

  media_exec = []
  desvio_padrao_exec = []

  for _ in range(10):
    regr = MLPRegressor(hidden_layer_sizes=camadas,
                        max_iter=iteracoes,
                        activation='relu',
                        solver='adam',
                        learning_rate='adaptive',
                        n_iter_no_change=50)

    # Treinamento do modelo
    regr.fit(x, y)

    # Predição e avaliação
    y_est = regr.predict(x)
    media_exec.append(np.average(y_est))
    desvio_padrao_exec.append(np.std(y_est))

    # Visualização (opcional)
    plotar_resultados(x, y, y_est, regr.loss_curve_)

  return np.average(media_exec), np.average(desvio_padrao_exec)

def plotar_resultados(x, y, y_est, loss_curve):
  """
  Plota os resultados do modelo (dados, curva de aprendizado e regressor).

  Args:
    x: Dados de entrada.
    y: Valores reais (target).
    y_est: Valores preditos pelo modelo.
    loss_curve: Curva de aprendizado do modelo.
  """

  plt.figure(figsize=[14, 7])

  # Curva de dados original
  plt.subplot(1, 3, 1)
  plt.plot(x, y)

  # Curva de aprendizado
  plt.subplot(1, 3, 2)
  plt.plot(loss_curve)

  # Curva de regressão
  plt.subplot(1, 3, 3)
  plt.plot(x, y, linewidth=1, color='yellow')
  plt.plot(x, y_est, linewidth=2)
  plt.show()

# Carregar dados
print('Carregando Arquivo de teste')
arquivo = np.load('teste5.npy')
x = arquivo[0]
y = np.ravel(arquivo[1])

# Definir parâmetros
iteracoes = 10000
camadas = (10, 10)

# Executar modelo e obter resultados
media_final, desvio_padrao_final = executar_modelo(iteracoes, camadas)

# Exibir resultados finais
print(f'Média das médias da execução com os parâmetros max_iter: {iteracoes} setados e o tamanho de camadas: {camadas} é igual a: {media_final:.2f}')
print(f'Média dos desvios padrões da execução com os parametros max_iter: {iteracoes} setados e o tamanho de camadas: {camadas} é igual a: {desvio_padrao_final:.2f}')
