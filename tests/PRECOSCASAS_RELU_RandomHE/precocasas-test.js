
// Função para calcular o Min-Max Scaling
function normalizeData(dataset) {
    // Transpor os dados para acessar colunas
    const transposed = dataset[0].map((_, colIndex) => dataset.map(row => row[colIndex]));
  
    // Encontrar os valores mínimos e máximos de cada coluna
    const minValues = transposed.map(col => Math.min(...col));
    const maxValues = transposed.map(col => Math.max(...col));
  
    // Normalizar os dados
    const normalized = dataset.map(row =>
      row.map((value, index) => {
        const min = minValues[index];
        const max = maxValues[index];
        return max === min ? 0 : (value - min) / (max - min); // Evitar divisão por zero
      })
    );
  
    return normalized;
}

// Função para desnormalizar os dados
function denormalizeData(normalizedDataset, originalDataset) {
    // Transpor os dados originais para acessar colunas
    const transposedOriginal = originalDataset[0].map((_, colIndex) => 
        originalDataset.map(row => row[colIndex])
    );

    // Encontrar os valores mínimos e máximos de cada coluna
    const minValues = transposedOriginal.map(col => Math.min(...col));
    const maxValues = transposedOriginal.map(col => Math.max(...col));

    // Desnormalizar os dados
    const denormalized = normalizedDataset.map(row =>
        row.map((value, index) => {
            const min = minValues[index];
            const max = maxValues[index];
            return value * (max - min) + min; // Desnormalização
        })
    );

    return denormalized;
}
  
// Aplicar a normalização
const normalizedData = normalizeData(dataset);
  
// Estrutura da rede: 11 neurônios na entrada, 3 na camada oculta, 1 na saída, todas usando ReLU
const config = {
    layers: [
        { type: LayerType.Input,  inputs: 11, units: 11 }, 
        { type: LayerType.Hidden, inputs: 11, units: 3, functions: [ 'ReLU', 'ReLU', 'ReLU']  }, 
        { type: LayerType.Final,  inputs: 3, units: 1, functions: [ 'ReLU' ]  }
    ],
    initialization: Initialization.RandomHeNormal
};

const mlp = new MLP(config);

mlp.importParameters({
    "weights": [
        [
            [
                0.22677741005078256,
                0.2456372833335657,
                0.26918293463221976,
                0.22519569320736615,
                -0.18432489103248736,
                0.34579040631837127,
                -0.08757992625300418,
                0.05576830245177678,
                -0.35362342289855814,
                -0.09484887191889994,
                -0.2360360831835377
            ],
            [
                -0.33216649606219517,
                -0.05894729454604437,
                -0.1765651907528107,
                -0.17381980749208345,
                -0.10555980058127111,
                -0.4016428569268778,
                -0.3657646263998327,
                -0.1037713952865656,
                -0.1746689026849487,
                -0.04748451438970722,
                0.3482923921555126
            ],
            [
                0.13421514142694038,
                -0.2543354667392169,
                -0.33140275064831115,
                0.3898004709574889,
                -0.012427134540697481,
                -0.04477495683190677,
                -0.13025215371086898,
                0.08357895521254928,
                -0.06860498089942081,
                -0.10238235722939426,
                0.25801216607846583
            ]
        ],
        [
            [
                0.02118254566620611,
                0.2401700180778615,
                -0.195938037548431
            ]
        ]
    ],
    "biases": [
        [
            0,
            0,
            0
        ],
        [
            0
        ]
    ],
    "layers": [
        11,
        3,
        1
    ],
    "generatedAt": 1734705209980
});

let datasetTratado = [];
let targets       = [];

//Treino e Teste
const numColunaPreco = 10; //O numero da coluna que queremos estimar, que no caso é o preco
const numeroAmostrasTreino = 800;
const numeroAmostrasTeste  = 200;

//Hyperparametros
const taxa_aprendizado = 0.005;

//Epocas
const numero_epocas = 110000;
const exibirACada = 1000;

//Remove a coluna target
for( let l = 0 ; l < dataset.length ; l++ )
{
    const linha = normalizedData[l];
    const novaLinha = [];

    //Ignora a ultiam coluna que vai ser a variavel que vamos estimar
    for( let c = 0 ; c < linha.length ; c++ ){
        if( c != numColunaPreco ){
            novaLinha.push( linha[c] )
        }
    }

    //Vai ter todas as features MENOS A COLUNA TARGET
    datasetTratado.push(novaLinha);
}

//Extrai APENAS os valores target
for( let l = 0 ; l < dataset.length ; l++ )
{   
    const linha = normalizedData[l];

    //Pega somente a variavel que vamos estimar
    targets.push( [linha[ numColunaPreco ]] );

}

// Dados de entrada para o problema
const inputs_modelo = datasetTratado.slice(0, numeroAmostrasTreino);

// Saídas esperadas para o XOR
const targets_modelo = targets.slice(0, numeroAmostrasTreino);;

// Treinando a rede com os dados de treino
mlp.train(inputs_modelo, targets_modelo, taxa_aprendizado, numero_epocas, exibirACada);

//Testando com dados de teste
for( let l = numeroAmostrasTreino+1 ; l < numeroAmostrasTreino+1+numeroAmostrasTeste ; l++ ){
    const valorEstimado = mlp.forward( datasetTratado[l] );
    const valorReal     = targets[ l ];
    const diff          = Math.pow( (valorEstimado - valorReal), 2 ); 

    console.log(`Amostra de teste numero ${l}: Valor Estimado ${ valorEstimado } - Valor Real: ${ valorReal }, ERRO AO QUADRADO: ${ diff }`)
}

//TODO: Desnormalizar os resultados
