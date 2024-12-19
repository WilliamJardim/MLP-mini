
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
    initialization: Initialization.Zeros
};

const mlp = new MLP(config);

//Carrega pesos iniciais que sei que vai funcionar pra esse problema
mlp.importParameters({
    "weights": [
        [
            [
                -0.02453568699326114,
                0.789170420894993,
                0.5701559345656122,
                0.7248060565878056,
                0.43603054056979085,
                0.20875285723725634,
                0.8751427805511227,
                0.7893549350170037,
                0.6319248135545701,
                0.9578649545445583,
                -0.02404991670386325
            ],
            [
                0.6111889558798751,
                -0.2805481117458055,
                0.5768812559362142,
                0.21499658959705847,
                0.223493291798039,
                -0.7068488275559957,
                -0.40135789872683336,
                0.44357510086619145,
                -0.03651367201878086,
                0.7909906270646605,
                0.7756861421704944
            ],
            [
                -0.22433343040925324,
                0.5472350772876187,
                -0.5014198770952891,
                -0.8674938891227195,
                -0.26105774781462454,
                0.1426245832189985,
                0.3853286481688927,
                0.022601040049062604,
                -0.6921627327618238,
                0.470772528794293,
                -0.7213779161215994
            ]
        ],
        [
            [
                0.6129199597665016,
                0.4582438548356986,
                -0.13517353462138537
            ]
        ]
    ],
    "biases": [
        [
            0.2825339397021591,
            -0.8618110211550665,
            -0.028659394190896137
        ],
        [
            -0.8159161177264354
        ]
    ],
    "layers": [
        11,
        3,
        1
    ],
    "generatedAt": 1734647094720
});

let datasetTratado = [];
let targets       = [];

//Treino e Teste
const numColunaPreco = 10; //O numero da coluna que queremos estimar, que no caso é o preco
const numeroAmostrasTreino = 300;
const numeroAmostrasTeste  = 50;

//Hyperparametros
const taxa_aprendizado = 0.01;

//Epocas
const numero_epocas = 12000;
const exibirACada = 10;

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
