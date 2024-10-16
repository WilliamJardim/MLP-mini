class ActivationFunctions {
    // Torna a classe um singleton impedindo instanciamento externo
    private constructor() {}

    // Função de ativação sigmoide
    public static sigmoid(x: number): number {
        return 1 / (1 + Math.exp(-x));
    }

    // Derivada da sigmoide
    public static sigmoidDerivative(x: number): number {
        return x * (1 - x);
    }

    // Função de ativação ReLU
    public static ReLU(x: number): number {
        return Math.max(0, x);
    }

    // Derivada da ReLU
    public static ReLUDerivative(x: number): number {
        return x > 0 ? 1 : 0;
    }
}

// Função para inicializar pesos de forma aleatória
function randomWeight(): number {
    return Math.random() * 2 - 1; // Gera valores entre -1 e 1
}

// Rede Neural MLP com suporte a múltiplas camadas
class MLP {
    private layers;
    private weights;
    private biases;
    private layerActivations;

    constructor(layers: number[]) {
        // layers é um array onde cada elemento é o número de neurônios na respectiva camada
        this.layers = layers as number[];

        // Inicializando pesos e biases para todas as camadas
        this.weights = [] as number[];
        this.biases  = [] as number[];

        for (let i = 1; i < layers.length; i++) {
            // Pesos entre a camada i-1 e a camada i
            const layerWeights = [] as number[][];

            for (let j = 0; j < layers[i]; j++) {
                const neuronWeights = [] as number[];

                for (let k = 0; k < layers[i - 1]; k++) {
                    neuronWeights.push( randomWeight() );
                }
                
                layerWeights.push(neuronWeights);
            }

            this.weights.push(layerWeights);

            // Biases para a camada i
            const layerBiases = Array(layers[i]).fill(0).map(() => randomWeight()) as number[];
            this.biases.push(layerBiases);
        }
    }

    /**
    * Calcula o custo de todas as amostras de uma só vez
    * 
    * @param {Array} train_samples - Todas as amostras de treinamento
    * @returns {Number} - o custo
    */
    compute_train_cost( inputs: number[][], mytargets:number[][], estimatedValues:number[][] ): number{

        let cost = 0;
        
        inputs.forEach((input: number[], i: number) => {
            const targets                = mytargets[i];
            const estimations: number[]  = estimatedValues[i];

            for( let S = 0 ; S < estimations.length ; S++ )
            {
                cost = cost + Math.pow( (estimations[S] - targets[S]), 2 );
            }

        });

        return cost;
    }
    
    /**
    * Log the current network parameters values in a string
    * 
    * @param parameterShow - The show type
    */
    logParameters( parameterShow:string = 'verbose'): void{
        let netStr:string = '-=-=- WEIGHS OF THE NETWORK: -=-=- \n\n';
        let identSimbol = '--->';

        for( let l = 0 ; l < this.weights.length ; l++ )
        {
            netStr += `LAYER ${ l }:\n `;

            for (let j = 0; j < this.weights[l].length; j++) {
                if( parameterShow == 'verbose' ){
                    netStr += `     ${identSimbol} UNIT OF NUMBER ${ j }:\n `;

                }else if( parameterShow == 'short' ){
                    netStr += `     ${identSimbol} UNIT ${ j }:\n `;
                }

                for( let k = 0 ; k < this.weights[l][j].length ; k++ ){
                    if( parameterShow == 'verbose' ){
                        netStr += `          ${identSimbol} WEIGHT OF INPUT X${ k }: ${this.weights[l][j][k]}\n `;

                    }else if( parameterShow == 'short' ){
                        netStr += `          ${identSimbol} W${ j }${ k }: ${this.weights[l][j][k]}\n `;
                    }
                }

                netStr += `          ${identSimbol} BIAS: ${this.biases[l][j]}\n `;
                
                netStr += '\n';
            }

            netStr += '\n';
        }

        console.log(netStr);
    }

    /**
    * Export the current network parameters values into a JSON object
    * @returns {JSON}
    */
    exportParameters(): JSON{
        return ( JSON.parse( JSON.stringify( {
            weighs: [... this.weights.copyWithin() ],
            biases: [... this.biases.copyWithin()  ],

            //Other info
            generatedAt: new Date().getTime()
        } ) ) );
    }

    // Forward pass (passagem direta)
    forward(input: number[]) {
        let activations = input as number[];

        // Passar pelos neurônios de cada camada
        this.layerActivations = [activations] as number[][]; // Para armazenar as ativações de cada camada

        for (let l = 0; l < this.weights.length; l++) {
            const nextActivations = [] as number[];

            for (let j = 0; j < this.weights[l].length; j++) {

                let weightedSum:number = 0;

                for (let k = 0; k < activations.length; k++) {
                    weightedSum += activations[k] * this.weights[l][j][k];
                }

                weightedSum += this.biases[l][j];

                nextActivations.push( ActivationFunctions.sigmoid(weightedSum) );
            }

            activations = nextActivations;
            this.layerActivations.push( activations );
        }

        return activations;
    }

    // Função de treinamento com retropropagação
    train(
          inputs: number[][], 
          targets: number[][], 
          learningRate: number = 0.1, 
          epochs: number = 10000, 
          printEpochs:number = 1000

    ): void {

        console.log(`Erro inicial(ANTES DO TREINAMENTO): ${ this.compute_train_cost( inputs, targets, inputs.map( (xsis: number[]) => this.forward(xsis) ) ) }`);

        for (let epoch = 0; epoch < epochs; epoch++) {
        
            inputs.forEach((input, i) => {
                const target = targets[i];

                // Passagem direta
                const output = this.forward(input);

                // Cálculo do erro da saída
                const outputError = [];
                for (let j = 0; j < output.length; j++) {
                    const error = target[j] - output[j];
                    outputError.push(error);
                }

                // Backpropagation (retropropagação)
                const layerErrors = [outputError];

                // Cálculo dos erros das camadas ocultas, começando da última camada
                for (let l = this.weights.length - 1; l >= 1; l--) {
                    const layerError = [];
                    
                    for (let j = 0; j < this.weights[l - 1].length; j++) {
                    
                        let error = 0;
                        for (let k = 0; k < this.weights[l].length; k++) {
                            error += layerErrors[0][k] * this.weights[l][k][j];
                        }
                    
                        layerError.push(error * ActivationFunctions.sigmoidDerivative(this.layerActivations[l][j]));
                    }

                    /**
                    * Adiciona os erros das camada oculta atual como sendo o primeiro elemento do array "layerErrors", e os demais elementos que já existem no array ficam atráz dele, sequencialmente.
                    * Isso por que layerErrors[0] sempre vai retornar os erros da ultima camada calculada pelo Backpropagation
                    * 
                    * E é por isso que layerErrors[0] começa sendo os erros da camada de saida(ou seja da última camada da rede)
                    * e na segunda interação, do for "for (let l = this.weights.length - 1; l >= 1; l--) {", ao chegar nesse unshift, layerErrors[0] passa a ser os erros da penultima camada oculta
                    * e assim por diante
                    */
                    layerErrors.unshift(layerError);
                }

                // Atualização dos pesos e biases
                for (let l = this.weights.length - 1; l >= 0; l--) {
                    for (let j = 0; j < this.weights[l].length; j++) {
                        for (let k = 0; k < this.weights[l][j].length; k++) {
                            // Atualiza os pesos usando a retropropagação
                            this.weights[l][j][k] += learningRate * layerErrors[l][j] * this.layerActivations[l][k];
                        }

                        // Atualiza os biases
                        this.biases[l][j] += learningRate * layerErrors[l][j];
                    }
                }
            });

            let totalError:number = this.compute_train_cost( inputs, targets, inputs.map( (xsis: number[]) => this.forward(xsis) ) );

            // Log do erro para monitoramento
            if (epoch % printEpochs === 0) {
                console.log(`Epoch ${epoch}, Erro total: ${totalError}`);
            }
        }
    }

    // Função para prever a saída para um novo conjunto de entradas
    estimate(input: number[]): number[] {
        const output = this.forward(input) as number[];
        return output.map( (o: number) => (o > 0.5 ? 1 : 0) );
    }
}
