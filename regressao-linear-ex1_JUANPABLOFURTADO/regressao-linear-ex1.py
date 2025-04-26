"""
@file regressao-linear-ex1.py
@brief Implementação completa da regressão linear com visualização.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from Functions.warm_up_exercises import (
    warm_up_exercise1, warm_up_exercise2, warm_up_exercise3,
    warm_up_exercise4, warm_up_exercise5, warm_up_exercise6,
    warm_up_exercise7
)
from Functions.plot_data import plot_data
from Functions.compute_cost import compute_cost
from Functions.gradient_descent import gradient_descent

def main():
    # Garante que a pasta de figuras existe
    os.makedirs("Figures", exist_ok=True)

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
    x = data[:, 0]    # população
    y = data[:, 1]    # lucro
    m = len(y)        # número de exemplos

    plot_data(x, y)

    input("Programa pausado. Pressione Enter para continuar...")

    # Adiciona bias (coluna de 1s)
    x_aug = np.column_stack((np.ones(m), x))

    # Inicializa theta
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
    theta = np.array([0.0, 0.0])  # ou outro valor inicial, por padrão zeros

    theta, J_history, theta_history = gradient_descent(x_aug, y, theta, alpha, iterations)

    print('Parâmetros theta encontrados pela descida do gradiente:')
    print(theta)
    print('Valores esperados para theta (aproximadamente):')
    print(' -3.6303\n  1.1664')

    # 1. Gráfico da convergência da função de custo
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, iterations + 1), J_history, 'b-', linewidth=2)
    plt.xlabel('Iteração')
    plt.ylabel('Custo J(θ)')
    plt.title('Convergência da Descida do Gradiente')
    plt.savefig("Figures/convergencia_custo.png", dpi=300, bbox_inches='tight')
    plt.savefig("Figures/convergencia_custo.svg", format='svg', bbox_inches='tight')
    plt.grid(True)
    plt.show()

    # 2. Gráfico do Ajuste da Regressão Linear
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, 'rx', markersize=5, label='Dados de treino')
    plt.plot(x, x_aug @ theta, 'b-', linewidth=2, label='Regressão linear')
    plt.xlabel('População da cidade (em dezenas de mil)')
    plt.ylabel('Lucro (em dezenas de mil dólares)')
    plt.title('Ajuste da Regressão Linear')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Previsões usando theta ajustado
    predict1 = np.array([1, 3.5]) @ theta
    predict2 = np.array([1, 7.0]) @ theta
    print(f'\nPara população = 35.000, lucro previsto = ${predict1 * 10000:.2f}')
    print(f'Para população = 70.000, lucro previsto = ${predict2 * 10000:.2f}')
    input("Programa pausado. Pressione Enter para continuar...")

    # 3. Gráfico de superfície 3D da função de custo J(θ₀, θ₁)
    print('Visualizando a função J(theta_0, theta_1) – superfície 3D...')
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)
    j_vals = np.zeros((len(theta0_vals), len(theta1_vals)))
    for i, t0 in enumerate(theta0_vals):
        for j, t1 in enumerate(theta1_vals):
            j_vals[i, j] = compute_cost(x_aug, y, np.array([t0, t1]))
    j_vals = j_vals.T

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    t0_mesh, t1_mesh = np.meshgrid(theta0_vals, theta1_vals)
    ax.plot_surface(t0_mesh, t1_mesh, j_vals, cmap='viridis', edgecolor='none')
    ax.set_xlabel(r'$\theta_0$')
    ax.set_ylabel(r'$\theta_1$')
    ax.set_zlabel('Custo')
    plt.title('Superfície da Função de Custo')
    plt.show()

    # 4. Gráfico de contorno da função de custo
    print('Visualizando a função J(theta_0, theta_1) – contorno...')
    plt.figure(figsize=(8, 5))
    plt.contour(theta0_vals, theta1_vals, j_vals, levels=np.logspace(-2, 3, 20))
    plt.plot(theta[0], theta[1], 'rx', markersize=10, linewidth=2)
    plt.xlabel(r'$\theta_0$')
    plt.ylabel(r'$\theta_1$')
    plt.title('Contorno da Função de Custo')
    plt.grid(True)
    plt.show()

    # 5) Contorno da função de custo + trajetória do gradiente
    plt.figure(figsize=(8, 5))
    cs = plt.contour(theta0_vals, theta1_vals, j_vals, levels=np.logspace(-2, 3, 20))
    plt.clabel(cs, inline=1, fontsize=8)
    plt.plot(theta_history[:, 0], theta_history[:, 1], 'r.-', markersize=6, label='Trajetória do gradiente')
    plt.xlabel(r'$\theta_0$')
    plt.ylabel(r'$\theta_1$')
    plt.title('Contorno da Função de Custo com Trajetória')
    plt.legend()
    plt.grid(True)
    plt.savefig("Figures/contorno_trajetoria.png", dpi=300, bbox_inches='tight')
    plt.savefig("Figures/contorno_trajetoria.svg", format='svg', bbox_inches='tight')
    plt.show()

    # 6) Superfície da função de custo com trajetória 3D melhor visualizada
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(t0_mesh, t1_mesh, j_vals, cmap='viridis', edgecolor='none', alpha=0.6)
    ax.view_init(elev=18, azim=-18, roll=-5)
    costs = np.concatenate(([compute_cost(x_aug, y, theta_history[0])], J_history))
    ax.plot(theta_history[:, 0], theta_history[:, 1], costs, color='red', linewidth=3, marker='o', markersize=4, label='Trajetória do gradiente')
    ax.scatter(theta_history[0, 0], theta_history[0, 1], costs[0], color='blue', s=50, label='Início')
    ax.scatter(theta_history[-1, 0], theta_history[-1, 1], costs[-1], color='green', s=50, label='Convergência')
    ax.set_xlabel(r'$\theta_0$')
    ax.set_ylabel(r'$\theta_1$')
    ax.set_zlabel('Custo')
    plt.title('Superfície da Função de Custo com Trajetória 3D')
    ax.legend()
    plt.savefig("Figures/superficie_trajetoria.png", dpi=300, bbox_inches='tight')
    plt.savefig("Figures/superficie_trajetoria.svg", format='svg', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()
