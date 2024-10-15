// Estrutura da rede: 2 neurônios na entrada, 2 na camada oculta, 1 na saída
const layers = [2, 2, 1];
const mlp = new MLP(layers);

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
console.log('Previsões:');
inputs.forEach(input => {
    const prediction = mlp.predict(input);
    console.log(`Entrada: ${input}, Previsão: ${prediction}`);
});


