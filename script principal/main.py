import numpy as np
import matplotlib.pyplot as plt
import os

from Functions.warm_up_exercises import warm_up_exercise1, warm_up_exercise2, warm_up_exercise3, warm_up_exercise4
from Functions.warm_up_exercises import warm_up_exercise5, warm_up_exercise6, warm_up_exercise7
from Functions.plot_data import plot_data
from Functions.compute_cost import compute_cost
from Functions.gradient_descent import gradient_descent


def main():

    os.makedirs("Figures", exist_ok=True)

    print('Executando o exercício de aquecimento (warm_up_exercise)...')
    print('Matriz identidade 5x5:')
    print('Executando os exercícios de aquecimento...')

    print('\nExercício 1: Matriz identidade 5x5') 
    print(warm_up_exercise1())

    print('\nExercício 2: Vetor de 1s (m=3)')
    print(warm_up_exercise2(3))

    print('\nExercício 3: Adiciona coluna de 1s ao vetor [2, 4, 6]')
    x_ex3 = np.array([2, 4, 6])
    print(warm_up_exercise3(x_ex3))

    print('\nExercício 4: Produto X @ theta')
    X_ex4 = warm_up_exercise3(x_ex3)
    theta_ex4 = np.array([1, 2])
    print(warm_up_exercise4(X_ex4, theta_ex4))

    print('\nExercício 5: Erros quadráticos entre predições e y')
    preds = warm_up_exercise4(X_ex4, theta_ex4)
    y_ex5 = np.array([5, 9, 13])
    print(warm_up_exercise5(preds, y_ex5))

    print('\nExercício 6: Custo médio')
    errors_ex6 = warm_up_exercise5(preds, y_ex5)
    print(warm_up_exercise6(errors_ex6))

    print('\nExercício 7: Cálculo do custo médio completo')
    print(warm_up_exercise7(X_ex4, y_ex5, theta_ex4))

    input("Programa pausado. Pressione Enter para continuar...")

    print('Plotando os dados...')
    data = np.loadtxt('Data/ex1data1.txt', delimiter=',')
    x = data[:, 0]
    y = data[:, 1]
    m = len(y)

    plot_data(x, y)

    input("Programa pausado. Pressione Enter para continuar...")

    x_aug = np.column_stack((np.ones(m), x))

    theta = np.zeros(2)

    iterations = 1500
    alpha = 0.01

    print('\nTestando a função de custo...')
    cost = compute_cost(x_aug, y, theta)
    print(f'Com theta = [0 ; 0]\nCusto calculado = {cost:.2f}')
    print('Valor esperado para o custo (aproximadamente): 32.07')

    cost = compute_cost(x_aug, y, np.array([-1, 2]))
    print(f'\nCom theta = [-1 ; 2]\nCusto calculado = {cost:.2f}')
    print('Valor esperado para o custo (aproximadamente): 54.24')

    input("Programa pausado. Pressione Enter para continuar...")

    print('\nExecutando a descida do gradiente...')
    theta = np.array([8.5, 4.0])
    theta, J_history, theta_history = gradient_descent(x_aug, y, theta, alpha, iterations)

    print('Parâmetros theta encontrados pela descida do gradiente:')
    print(theta)
    print('Valores esperados para theta (aproximadamente):')
    print(' -3.6303\n  1.1664')

    # Gráficos seguem conforme descrito anteriormente
    ...
    # (Todo o código de visualização permanece igual, conforme já estava completo)
    ...


if __name__ == '__main__':
    main()
