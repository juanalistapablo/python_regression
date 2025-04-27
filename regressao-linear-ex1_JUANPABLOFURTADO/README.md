# Trabalho Individual: Regressão Linear

---

# ![UFMA](./ufma_logo.png)  ![Engenharia da Computação](./eng_comp_logo.png)

---


## Universidade Federal do Maranhão
### Engenharia da Computação
### Disciplina: EECP0053 - TÓPICOS EM ENGENHARIA DA COMPUTAÇÃO II - FUNDAMENTOS DE REDES NEURAIS
### Assunto: Regressão Linear
---

## 🎯 Objetivos

Este trabalho individual visa explorar o impacto da taxa de aprendizado (α) e da inicialização dos parâmetros (θ inicial) no comportamento do algoritmo de descida do gradiente para regressão linear, bem como a implementação dos componentes básicos da regressão linear.

Os objetivos específicos são:

- Avaliar a influência da taxa de aprendizado na convergência da função custo.
- Analisar a importância da inicialização dos pesos (θ) e suas implicações no processo de aprendizagem.
- Implementar os componentes fundamentais do algoritmo de regressão linear para consolidar o entendimento teórico e prático:
    - `warm_up_exercise.py`: exercícios de aquecimentos com matriz identidade
    - `plot_data.py`: visualização gráfica dos dados
    - `compute_cost.py`: cálculo da função de custo J(θ)
    - `gradient_descent.py`: execução da descida do gradiente
---

## 📚 Tópicos a serem abordados

### 1. Implementação e geração dos gráficos

- Convergência da função de custo ao longo das iterações.
- Ajuste da reta de regressão sobre os dados.
- Superfície 3D da função de custo com trajetória do gradiente.
- Contorno da função de custo com trajetória do gradiente.

### 2. Experimentos comparativos

#### 📌 Taxa de aprendizado (α)

- Escolha três valores distintos para α (ex: 0.001, 0.01 e 0.1), sem mudar os outros parâmetros
- Compare as curvas de convergência em um único gráfico.

#### 📌 Inicialização dos pesos (θ inicial)

- Fixe a taxa de aprendizado α em 0.01.
- Teste três inicializações distintas fixas (ex: `[0,0]`, `[5,5]`, `[-5,5]`) e 3 inicializações distintas de forma aleatória.
- Compare as trajetórias no gráfico de contorno (não esqueça de mudar os limites dos gráficos).

### 3. Análise escrita 

Para esta atividade, o aluno deve elaborar um texto dissertativo, formatado ABNT, explicando os achados. O aluno deve incluir
os gráficos elaborados em ambas as atividades do ítem 2. 
Obs > não esqueça de colocar legendas nas Figuras e explicá-las !!!!!!!!!!! 
- Descreva o que acontece quando α é muito grande ou muito pequeno.
- Explique a importância de uma inicialização adequada dos pesos, relacionando isso ao conceito de fine-tuning em redes neurais.

---

## 🗂️ Estrutura do Repositório GitHub

```
regressao-linear-ex1_JUANPABLOFURTADO/
│
├─ Figures/                # gráficos (.png e .svg)
│
├─ Data/
│   └─ ex1data1.txt
│
├─ Functions/
│   ├─ warm_up_exercises.py
│   ├─ plot_data.py
│   ├─ compute_cost.py
│   └─ gradient_descent.py
│
├─ README.md               # descrição do projeto
├─ regressao-linear-ex1.py # script principal
├─ ufma_logo.png           # logo da UFMA
├─ eng_comp_logo.png       # logo do curso
├─ REQUIREMENTS.txt        # bibliotecas necessárrias
├─ regressao-linear-ex1.yml# ambiente Conda, caso queria fazer uma criação automatizada com a instalação das libs necessárias
└─ setup_env.py            # script que automatiza a criação do ambiente e instalação das libs. Caso deseje, use python setup_env.py no terminal
```

## 🚀 Como executar o projeto

### ✅ Opção 1: Usando Conda (recomendado)

```bash
conda env create -f environment.yml
conda activate regressao-linear-ex1
python regressao-linear-ex1.py
```

### 🐍 Opção 2: Ambiente virtual com Python puro (mais genérico)

1. Certifique-se de ter um arquivo `requirements.txt` com as dependências mínimas:

```txt
numpy
matplotlib
```

2. Execute o script de configuração automática:

```bash
python setup_env.py
```

Esse script irá:
- Criar o ambiente virtual `regressao-linear-ex1`
- Instalar os pacotes do `requirements.txt`
- Mostrar como ativar o ambiente virtual (Windows, Linux ou MacOS)

> O script `setup_env.py` está incluído no repositório e funciona em qualquer sistema.

- Gráficos gerados ficarão na pasta `Figures/`.
- Renomeie cada figura gerada para facilitar comparações.

---


Dúvidas, estou à disposição por e-mail ou em sala.

## Reconhecimentos e Direitos Autorais

```
@autor:                Juan Pablo Furtado Mondego Macedo 
@contato:              [Seu Email]  
@data última versão:   26/04/2025  
@versão:               1.0  
@Agradecimentos:       Universidade Federal do Maranhão (UFMA),  
                       Prof. Dr. Thales Levi Azevedo Valente, thales.l.a.valente@gmail.com
                       https://www.linkedin.com/in/thalesvalente/
                       colegas de curso.
```

---

## Licença (MIT)

> Este material é resultado de um trabalho acadêmico para a disciplina *EECP0053 - TÓPICOS EM ENGENHARIA DA COMPUTAÇÃO II - FUNDAMENTOS DE REDES NEURAIS*, semestre letivo 2025.1, curso Engenharia da Computação, UFMA.

```
MIT License

Copyright (c) 20/04/2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
