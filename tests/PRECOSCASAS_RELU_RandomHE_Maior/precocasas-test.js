
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
  
// Estrutura da rede ajustada
const config = {
    layers: [
        { type: LayerType.Input,  inputs: 11, units: 11 }, //11 features
        { type: LayerType.Hidden, inputs: 11, units: 64, functions: Array(64).fill('ReLU') }, // 64 neurônios
        { type: LayerType.Hidden, inputs: 64, units: 32, functions: Array(32).fill('ReLU') }, // 32 neurônios
        { type: LayerType.Final, inputs: 32, units: 1, functions: ['Linear'] } // 1 saída contínua
    ],
    initialization: Initialization.RandomHeNormal // Inicialização recomendada para ReLU
};

const mlp = new MLP(config);

let datasetTratado = [];
let targets       = [];

//Treino e Teste
const numColunaPreco = 10; //O numero da coluna que queremos estimar, que no caso é o preco
const numeroAmostrasTreino = 3500;
const numeroAmostrasTeste  = 1500;

//Hyperparametros
const taxa_aprendizado = 0.001;

//Epocas
const numero_epocas = 100;
const exibirACada = 1;

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
const targets_modelo = targets.slice(0, numeroAmostrasTreino);


//Calcula a média do erro nos dados de teste ANTES DO TREINAMENTO
//Testando com dados de teste
var erroTotalTeste = 0;
var quantidadeTestes = 0;
for( let l = numeroAmostrasTreino+1 ; l < numeroAmostrasTreino+1+numeroAmostrasTeste ; l++ ){

    //Evita passar do range do dataset
    if( datasetTratado[l] )
    {

        const valorEstimado = mlp.forward( datasetTratado[l] );
        const valorReal     = targets[ l ];
        const diff          = Math.pow( (valorEstimado - valorReal), 2 ); 
        erroTotalTeste += diff;
        quantidadeTestes++;

        console.log(`Amostra de teste numero ${l}: Valor Estimado ${ valorEstimado } - Valor Real: ${ valorReal }, ERRO AO QUADRADO: ${ diff }`)

    }
}

console.log('Média do erro ao quadrado DADOS DE TESTE ANTES DO TREINAMENTO: ', erroTotalTeste/quantidadeTestes )





// Treinando a rede com os dados de treino
mlp.train(inputs_modelo, targets_modelo, taxa_aprendizado, numero_epocas, exibirACada);

//Testando com dados de teste
var erroTotalTeste = 0;
var quantidadeTestes = 0;
for( let l = numeroAmostrasTreino+1 ; l < numeroAmostrasTreino+1+numeroAmostrasTeste ; l++ ){

    //Evita passar do range do dataset
    if( datasetTratado[l] )
    {

        const valorEstimado = mlp.forward( datasetTratado[l] );
        const valorReal     = targets[ l ];
        const diff          = Math.pow( (valorEstimado - valorReal), 2 ); 
        erroTotalTeste += diff;
        quantidadeTestes++;

        console.log(`Amostra de teste numero ${l}: Valor Estimado ${ valorEstimado } - Valor Real: ${ valorReal }, ERRO AO QUADRADO: ${ diff }`)

    }
}

console.log('Média do erro ao quadrado DADOS DE TESTE APOS TREINAMENTO: ', erroTotalTeste/quantidadeTestes )

//TODO: Desnormalizar os resultados
