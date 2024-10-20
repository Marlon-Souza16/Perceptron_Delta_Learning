# Perceptron com Regra de Aprendizado Delta

Este projeto implementa um **Perceptron** para resolver o problema das portas lógicas **OR** ou **AND**, utilizando a **Regra de Aprendizado Delta**. Ele exemplifica o funcionamento básico de uma rede neural com um único neurônio e demonstra como ajustar pesos com base no erro durante o treinamento.

## Como Funciona?

O perceptron é treinado para classificar entradas binárias de acordo com a operação lógica **OR**. Os **pesos** são ajustados a cada iteração utilizando a **Regra Delta** para minimizar o erro entre a saída prevista e a desejada.

## O que é a Regra de Aprendizado Delta?

A **Regra Delta** é um método utilizado para **ajustar os pesos** de um neurônio durante o treinamento. A ideia principal é que, a cada iteração, o peso é modificado com base no **erro** entre a saída desejada e a saída prevista. 

A fórmula para o ajuste dos pesos é:

![Aprendizado_Delta](./aprendizado_delta.png)

- **Δw**: Ajuste no peso
- **c**: Taxa de aprendizado
- **d**: Saída desejada
- **o**: Saída prevista

Esse processo é repetido até que o erro seja suficientemente pequeno ou que o número máximo de iterações seja atingido.

---

## O que é uma Função Sigmoide?

A **função sigmoide** é uma função matemática que transforma um valor de entrada em um número entre **-1 e 1** (no caso da sigmoide escalada). No contexto do perceptron, a sigmoide é usada para calcular a **saída prevista** de maneira suave e contínua.

### Fórmula da Função Sigmoide Escalada:

![Função Sigmoide](https://latex.codecogs.com/png.image?\inline&space;\huge&space;\dpi{120}\bg{white}&space;S(x)=\left(\frac{2}{1&plus;e^{-x}}\right)-1&space;)

- Se o valor de entrada for positivo e alto, a saída será próxima de **1**.
- Se o valor for negativo e baixo, a saída será próxima de **-1**.
- Essa transição suave facilita o aprendizado do perceptron e ajuda na **retropropagação** do erro durante o ajuste dos pesos.

A sigmoide é útil porque:
- **Suaviza as decisões**, fornecendo previsões contínuas.
- **Facilita o ajuste dos pesos** com sua derivada, essencial para o aprendizado com a **Regra Delta**.

### Exemplo de Aplicação da função sigmoide (com a Entrada [1, 1, −1]):

Inicialmente devemos calcular o **produto interno (u)**, para isso, consideraremos que os pesos são:
 - w = [0.5, 0.5, -0.2]

O produto interno para a entrada [1, 1, -1] seria de: 

![Calculo produto interno (u)](./images/Produto_interno.png)

Portanto, a função de ativação nesse caso ficaria assim:

![Função Ativação](./images/funcao_ativacao.png)

#### Ao aplicarmos a Função de Ativação acima, considerando 

 - e = 0.3012
 - 1 / (1 + 0.3012) = 0.7685

![Calculo f(1.2)](./images/resultado_exemplo.png)

Portanto neste exemplo a função de ativação retorna aproximadamente 0.537, um valor próximo de 1, mas ainda abaixo.

---

### Entradas e Saídas

- **Entradas**:
    ```python
    [[1, 1, -1], [1, -1, -1], [-1, 1, -1], [-1, -1, -1]]
    ```
    Cada vetor representa combinações de entradas binárias, com um elemento fictício (bias) adicionado.

- **Saídas Desejadas (Porta OR)**:
    ```python
    d = [1, 1, 1, -1]
    ```
- **Caso queira utilizar a porta AND**:
     ```python
     d = [1, -1, -1, -1]
     ```

---

## Porque as portas **OR** e **AND** tem essa sútil diferença apenas na Saída (d) ? 

A diferença nas **saídas desejadas** entre as portas **OR** e **AND** reflete o comportamento lógico de cada operação. Embora as **entradas** sejam as mesmas para ambos os casos, a lógica interna das portas OR e AND é distinta, e isso se traduz nas saídas esperada, conforme demonstrado abaixo:

### Porta OR ('OU' lógico):
A saída é **1** (ou verdadeiro) se **pelo menos uma** das entradas for 1. Caso contrário, retorna **-1** (equivalente a 0, ou falso).

#### Tabela-verdade da OR:

| **Entrada 1** | **Entrada 2** | **Saída** |
|---------------|---------------|-----------|
| 1             | 1             | 1         |
| 1             | -1            | 1         |
| -1            | 1             | 1         |
| -1            | -1            | -1        |

**Explicação:**  
Se qualquer uma das entradas for 1, a saída será **1**. Apenas quando ambas são **-1** (falso), a saída será **-1**.

---

### Porta AND ('E' lógico):
A saída é **1** somente se **todas** as entradas forem 1. Caso contrário, retorna **-1**.

#### Tabela-verdade da AND:
| **Entrada 1** | **Entrada 2** | **Saída** |
|---------------|---------------|-----------|
| 1             | 1             | 1         |
| 1             | -1            | -1        |
| -1            | 1             | -1        |
| -1            | -1            | -1        |

**Explicação:**  
A porta AND é **mais restritiva**, pois exige que ambas as entradas sejam **1** para que a saída seja **1**. Se qualquer uma das entradas for **-1**, a saída será **-1**.

## Diferença Principal
A diferença na saída reflete as **regras da lógica booleana**:

- **OR:** Mais permissiva, retorna **1** se **qualquer uma** das entradas for **1**.  
- **AND:** Mais restritiva, só retorna **1** se **todas as entradas** forem **1**.

---

## Exemplo de Execução:

```python
import numpy as np
import math

# Função de ativação sigmoide
def f(u):
    return (2 / (1 + math.exp(-u))) - 1

# Função para calcular a saída
def find_output(data, w):
    u = np.dot(w, data) 
    lamb = 0.1 
    return f(lamb * u)

# Inicialização dos dados de entrada e saída
p = np.array([[1, 1, -1], [1, -1, -1], [-1, 1, -1], [-1, -1, -1]])
d = np.array([1, 1, 1, -1])

# Inicialização aleatória dos pesos
w = np.random.rand(p.shape[1])

# Parâmetros de treinamento
c = 0.5  # Taxa de aprendizado
d_error = 0.01  # Erro desejado
max_iter = 1000  # Máximo de iterações

# Treinamento usando a Regra Delta
for iteration in range(max_iter):
    error = 0
    for i in range(len(p)):
        o = find_output(p[i], w)
        error += 0.5 * (d[i] - o) ** 2
        delta = (d[i] - o) * (1 - o * o) / 2
        w += c * delta * p[i]
    
    print(f"Iteração {iteration + 1} | Erro: {error:.4f} | Pesos: {w}")
    
    if error < d_error:
        print(f"Convergência atingida após {iteration + 1} iterações.")
        break
else:
    print("Número máximo de iterações atingido sem convergência.")

# Teste do Perceptron
for entrada in p:
    print(f"Entrada: {entrada} | Saída prevista: {find_output(entrada, w)}")
```

---

## Resultados

- Vale ressaltar que o Perceptron foi treinado por até 1000 iterações para minimizar o erro.

### Abaixo segue exemplo de saída após 1000 iterações:

``` python

Iteração 1000 | Erro: 0.0183 | Pesos: [28.404, 28.408, -28.401]
Número máximo de iterações atingido sem convergência.
Entrada: [1 1 -1] | Saída prevista: 0.9996
Entrada: [1 -1 -1] | Saída prevista: 0.8896
Entrada: [-1 1 -1] | Saída prevista: 0.8896
Entrada: [-1 -1 -1] | Saída prevista: -0.8897

```

---

## Como o Perceptron Funciona

- Função de Ativação: Usamos uma sigmoide escalada para produzir saídas contínuas entre -1 e 1.

- Ajuste dos Pesos: A cada iteração, os pesos são atualizados para minimizar o erro entre a saída prevista e a desejada.

- Erro Residual: Como o treinamento parou ao atingir 1000 iterações, a convergência completa não foi alcançada, mas as previsões estão próximas do esperado. 
