
![Icone](./images/logo/logo256x256.png "Icone")

# MLP - Implementação de Rede Neural Multicamadas (MLP) em JavaScript

Este repositório contém uma implementação independente de uma Rede Neural Multicamadas (MLP) em JavaScript, gerada com a ajuda de Inteligência Artificial. A rede foi implementada sem o uso de bibliotecas de aprendizado de máquina ou notação matricial. Ela pode ser configurada para suportar múltiplas camadas e unidades, sendo aplicada ao problema clássico do XOR.

## Visão Geral

O MLP é uma rede neural feedforward totalmente conectada com uma ou mais camadas ocultas. Este código usa a função de ativação sigmoide e implementa o algoritmo de retropropagação (backpropagation) para ajustar os pesos da rede durante o treinamento.

Esta implementação foi desenvolvida de forma independente para ser simples e didática, realizando os cálculos elemento a elemento (em vez de usar operações matriciais).

## Características

- Suporte para múltiplas camadas ocultas.
- Função de ativação sigmoide e sua derivada.
- Retropropagação (backpropagation) implementada para ajuste de pesos e vieses.
- Treinamento e teste para resolver o problema lógico do XOR.
- Não usa bibliotecas externas, tornando-o fácil de entender e modificar.
- Código gerado com a assistência de IA, mantendo total independência da implementação.

## Como Funciona

A estrutura da rede é definida por um array onde cada elemento indica o número de unidades em cada camada. Por exemplo, a rede para o problema XOR possui 2 unidades na camada de entrada, 2 na camada oculta e 1 unidade na camada de saída:

```javascript
const mlp = new MLP({
    layers: [
        { type: LayerType.Input,  inputs: 2, units: 2 }, 
        { type: LayerType.Hidden, inputs: 2, units: 2 }, 
        { type: LayerType.Final,  inputs: 2, units: 1 }
    ],
    initialization: Initialization.Random
});
```

A função `train` é usada para treinar a rede, e `estimate` é usada para realizar estimativas após o treinamento.

## Exemplo: Problema XOR

O problema XOR é um problema lógico clássico que não pode ser resolvido com um único perceptron, mas pode ser resolvido com uma rede neural com uma camada oculta.

### Entradas e Saídas Esperadas

| Entrada | Saída Esperada |
|---------|----------------|
| [0, 0]  | [0]            |
| [0, 1]  | [1]            |
| [1, 0]  | [1]            |
| [1, 1]  | [0]            |

### Uso

1. Clone o repositório:
    ```bash
    git clone https://github.com/WilliamJardim/MLP-mini
    ```

2. Navegue até a pasta do projeto:
    ```bash
    cd MLP-mini/tests
    ```

3. Execute o arquivo JavaScript com `node`:
    ```bash
    node xor-test.js
    ```

    Ou então, acesse pelo navegador rodando o arquivo 'tests/xor-test.html'

4. Veja as estimativas da rede para o problema XOR:

    ```bash
    estimativas:
    Entrada: 0,0, Previsão: 0
    Entrada: 0,1, Previsão: 1
    Entrada: 1,0, Previsão: 1
    Entrada: 1,1, Previsão: 0
    ```

## Estrutura do Código

- **MLP**: Classe que representa a Rede Neural Multicamadas.
  - `constructor(layers)`: Inicializa os pesos e vieses da rede.
  - `forward(input)`: Realiza a passagem direta através da rede.
  - `train(inputs, targets, learningRate, epochs)`: Treina a rede usando backpropagation.
  - `estimate(input)`: Retorna estimativas (estimativas) para um dado conjunto de entradas.

## Integridade dos arquivos do projeto
Este projeto é cuidadosamente testado em detalhes, isso pode ser notado nos arquivos de teste de integridade que introduzi a partir do dia 16.10.2024, garantindo a integridade do comportamento dos arquivos do dia 15.10.2024(data initial do projeto) para com as futuras versões.
Você poderá ler sobre esses testes aqui:

  [Arquivo do teste de integridade](./tests/XOR INTEGRY TEST - 15-10 verifed/detalhes teste integridade.md);


## Licença

Este projeto está licenciado sob a licença MIT. Consulte o arquivo [LICENSE](LICENSE) para mais detalhes.
