// Estrutura da rede: 2 neurônios na entrada, 2 na camada oculta, 1 na saída
const config = {
    layers: [
        { type: LayerType.Input,  inputs: 2, units: 2 }, 
        { type: LayerType.Hidden, inputs: 2, units: 2, functions: [ 'Sigmoid', 'Sigmoid' ]  }, 
        { type: LayerType.Final,  inputs: 2, units: 1, functions: [ 'Sigmoid' ]  }
    ],
    initialization: Initialization.Random
};

const mlp = new MLP(config);

// Dados de entrada para o problema XOR
const inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
];

// Saídas esperadas para o XOR
const targets = [
    [0],
    [1],
    [1],
    [0]
];

// Treinando a rede
mlp.train(inputs, targets, 0.1, 10000);

// Testando a rede
console.log('Estimativas:');
inputs.forEach(input => {
    const output = mlp.estimate(input);
    console.log(`Entrada: ${input}, Estimativa: ${output}`);
});


