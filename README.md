
![Icone](./images/logo/logo256x256.png "Icone")

# MLP - Implementação de Rede Neural Multicamadas (MLP) em JavaScript
Este repositório contém uma implementação independente de uma Rede Neural Multicamadas (MLP) em JavaScript, gerada com a ajuda de Inteligência Artificial. A rede foi implementada sem o uso de bibliotecas de aprendizado de máquina ou notação matricial. Ela pode ser configurada para suportar múltiplas camadas e unidades, sendo aplicada ao problema clássico do XOR.

## Visão Geral
O MLP é uma rede neural feedforward totalmente conectada com uma ou mais camadas ocultas. Este código usa a função de ativação sigmoide e implementa o algoritmo de retropropagação (backpropagation) para ajustar os pesos da rede durante o treinamento.

Esta implementação foi desenvolvida de forma independente para ser simples e didática, realizando os cálculos elemento a elemento (em vez de usar operações matriciais).

## Transparência
Conforme mencionado acima, este código foi inicialmente gerado em Outubro de 2024, por uma interação com um assistente de IA (ChatGPT). A ideia era criar uma implementação manual de uma rede neural MLP para resolver o problema do XOR, oferecendo suporte a múltiplas camadas e unidades. O objetivo principal era consolidar meu entendimento sobre o funcionamento do backpropagation, utilizando uma base gerada por IA como ponto de partida, baseada em conceitos de domínio público.

Desde então, o código foi significativamente modificado e expandido para atender aos requisitos específicos do projeto, refletindo minha compreensão do tema e meu esforço em desenvolver uma implementação independente. Essa abordagem permitiu explorar e aprofundar meu conhecimento, garantindo que as soluções fossem personalizadas e alinhadas às boas práticas de aprendizado.

O processo de desenvolvimento está documentado para promover transparência e aprendizado colaborativo. Para acessar a conversa original que deu início a este projeto, clique aqui: **[Link para o Chat](https://chatgpt.com/share/676de54b-3614-8004-8f8f-9dfa2558f7e0)**. Note que o link foi gerado em 26 de dezembro de 2024, mas a conversa original ocorreu em Outubro de 2024. 

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

## Explicação aprofundada de como o algoritmo dessa implantação funciona:
- [Ler explicação](./docs/ANOTACOES_ALGORITMO.md)

- [Ver slides](./docs/slides/INFORMACOES_SLIDES.md)

## Estrutura do Código

- **MLP**: Classe que representa a Rede Neural Multicamadas.
  - `constructor(layers)`: Inicializa os pesos e vieses da rede.
  - `forward(input)`: Realiza a passagem direta através da rede.
  - `train(inputs, targets, learningRate, epochs)`: Treina a rede usando backpropagation.
  - `estimate(input)`: Retorna estimativas (estimativas) para um dado conjunto de entradas.

## Integridade dos arquivos do projeto
Este projeto é cuidadosamente testado em detalhes, isso pode ser notado nos arquivos de teste de integridade que introduzi a partir do dia 16.10.2024, garantindo a integridade do comportamento dos arquivos do dia 15.10.2024 (data inicial do projeto) para com as futuras versões.
Você poderá ler sobre esses testes aqui:

  [Caminho para o arquivo do teste de integridade](./tests/classificacao/XOR/XOR_INTEGRY_TEST_15_10_verifed/detalhes_teste_integridade.md)
  (ULTIMA VERIFICAÇÂO DE INTEGRIDADE 24/12/2024 21:40 PM)

## Licença

Este projeto está licenciado sob a licença MIT. Consulte o arquivo [LICENSE](LICENSE) para mais detalhes.
