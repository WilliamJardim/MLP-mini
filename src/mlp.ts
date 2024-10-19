import ActivationFunctions from './utils/ActivationFunctions';
import LayerDeclaration from './interfaces/LayerDeclaration';
import MLPConfig from './interfaces/MLPConfig';
import './utils/Enums';
import ValidateStructure from './validators/ValidateStructure';
import ValidateDataset from './validators/ValidateDataset';
import ValidateLayerFunctions from './validators/ValidateLayerFunctions';
import notifyIfhasNaN from './utils/notifyIfhasNaN';

// Função para inicializar pesos de forma aleatória
function randomWeight(): number {
    return Math.random() * 2 - 1; // Gera valores entre -1 e 1
}

// Rede Neural MLP com suporte a múltiplas camadas
class MLP {
    private config             : MLPConfig;
    private layers             : number[];
    private layers_functions   : string[][];
    private weights            : number[][][];
    private biases             : number[][];
    private layerActivations   : number[][];
    private initialParameters  : DoneParameters;

    public constructor(config: MLPConfig) {
        this.config = config;

        // Aplica uma validação de estrutura 
        ValidateStructure( this.config );

        // layers é um array onde cada elemento é o número de unidades na respectiva camada
        // Essa informação será extraida do config
        this.layers = []             as number[];

        //Esse aqui é um array para armazenar os nomes das funções de ativações das unidades de cada camada, assim: Array de Array<string>
        this.layers_functions = []   as string[][];

        for( let layerIndex = 0; layerIndex < this.config.layers.length ; layerIndex++ ){
            const layerDeclaration:LayerDeclaration = this.config.layers[layerIndex];

            this.layers[ layerIndex ] = layerDeclaration.units;
        }

        //Identifica quais as funções que cada unidade de cada camada usa,
        //Ignora a camada de entrada que não possui funções
        for( let layerIndex = 1; layerIndex < this.config.layers.length ; layerIndex++ ){
            const layerDeclaration:LayerDeclaration = this.config.layers[layerIndex];

            //Usei - 1 pra ignorar a camada de entrada, e ordenar corretamente
            this.layers_functions[ layerIndex-1 ] = layerDeclaration.functions;
        }

        //Adicionar validação aqui para validar as funções das camadas
        if( this.layers_functions.length > 0 ){
            //Se tiver this.layers_functions, então ele precisa validar
            ValidateLayerFunctions( this.config );
        }

        // Inicializando pesos e biases para todas as camadas
        this.weights = [];
        this.biases  = [];

        if( config.initialization == Initialization.Random )
        {
            for (let i = 1; i < this.layers.length; i++) {
                // Pesos entre a camada i-1 e a camada i
                const layerWeights: number[][] = [];

                for (let j = 0; j < this.layers[i]; j++) {
                    const neuronWeights: number[] = [];

                    for (let k = 0; k < this.layers[i - 1]; k++) {
                        neuronWeights.push( randomWeight() );
                    }
                    
                    layerWeights.push(neuronWeights);
                }

                this.weights.push(layerWeights);

                // Biases para a camada i
                const layerBiases = Array(this.layers[i]).fill(0).map(() => randomWeight()) as number[];
                this.biases.push(layerBiases);
            }


        }else if( config.initialization == Initialization.Zeros ){

            for (let i = 1; i < this.layers.length; i++) {
                // Pesos entre a camada i-1 e a camada i
                const layerWeights: number[][] = [];

                for (let j = 0; j < this.layers[i]; j++) {
                    const neuronWeights: number[] = [];

                    for (let k = 0; k < this.layers[i - 1]; k++) {
                        neuronWeights.push( 0 );
                    }
                    
                    layerWeights.push(neuronWeights);
                }

                this.weights.push(layerWeights);

                // Biases para a camada i
                const layerBiases = Array(this.layers[i]).fill(0).map(() => 0) as number[];
                this.biases.push(layerBiases);
            }
    

        }else if( config.initialization == Initialization.Manual ){
            this.importParameters( config.parameters! );

            
        }else if( config.initialization == Initialization.Dev )
        {
            //Aqui fica por conta do programador definir os parametros antes de tentar usar o modelo
        }

        //Faz a exportação dos parametros iniciais
        this.initialParameters = this.exportParameters();
    }

    /**
    * Calcula o custo de todas as amostras de uma só vez
    * 
    * @param {Array} train_samples - Todas as amostras de treinamento
    * @returns {Number} - o custo
    */
    public static compute_train_cost( inputs: number[][], mytargets:number[][], estimatedValues:number[][] ): number{

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
    * Retorna os parametros iniciais que foram usados para inicializar a rede
    */
    public getInitialParameters(): DoneParameters{
        return this.initialParameters;
    }
    
    /**
    * Log the current network parameters values in a string
    * 
    * @param parameterShow - The show type
    */
    public logParameters( parameterShow:string = 'verbose'): void{
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
    * @returns {DoneParameters}
    */
    public exportParameters(): DoneParameters{
        return ( JSON.parse( JSON.stringify( {
            weights: [... this.weights ],
            biases: [... this.biases  ],

            layers: this.layers,

            //Other info
            generatedAt: new Date().getTime()
        } ) ) );
    }

    /**
    * Import the parameters intro this network
    * @param {parameters} - The JSON object that contain the weights and biases 
    */
    public importParameters( parameters:DoneParameters ): void{
        console.log(`Loading parameters from JSON, from date: ${ parameters.generatedAt! }`);
        this.layers  = parameters!.layers;
        this.weights = parameters!.weights;
        this.biases  = parameters!.biases;
        console.log(`Success from import JSON, from date: ${ parameters.generatedAt! }`);
    }

    // Forward pass (passagem direta)
    public forward(input: number[]) {
        let activations = input as number[];

        // Passar pelos neurônios de cada camada
        this.layerActivations = [activations]; // Para armazenar as ativações de cada camada

        for (let l = 0; l < this.weights.length; l++) {
            const nextActivations:number[] = [];

            for (let j = 0; j < this.weights[l].length; j++) {

                let weightedSum:number = 0;

                for (let k = 0; k < activations.length; k++) {
                    weightedSum += activations[k] * this.weights[l][j][k];
                }

                weightedSum += this.biases[l][j];

                //Verifica se a unidade tem uma função especificada, ou se vai usar uma função padrão
                const unidadeTemFuncao : boolean = (this.layers_functions.length > 0 && this.layers_functions[l] && this.layers_functions[l][j]) ? true : false;
                const nomeDaFuncao     : string = ( unidadeTemFuncao == true ? this.layers_functions[l][j] : 'Sigmoid' );

                nextActivations.push( ActivationFunctions[ nomeDaFuncao ]( weightedSum ) );

                /** NaN detector */
                if(notifyIfhasNaN( 'feedforward/loops', [
                    weightedSum,
                    this.biases[l][j],
                    this.weights[l][j],
                    ActivationFunctions[ nomeDaFuncao ]( weightedSum )

                ]).hasNaN){
                    debugger;
                };

            }

            activations = nextActivations;
            this.layerActivations.push( activations );
        }

        return activations;
    }

    // Função de treinamento com retropropagação
    public train(
          inputs: number[][], 
          targets: number[][], 
          learningRate: number = 0.1, 
          epochs: number = 10000, 
          printEpochs:number = 1000

    ): void {

        // Valida os dados de treinamento
        ValidateDataset( this.config, 
                         inputs, 
                         targets );

        console.log(`Erro inicial(ANTES DO TREINAMENTO): ${ MLP.compute_train_cost( inputs, targets, inputs.map( (xsis: number[]) => this.forward(xsis) ) ) }`);

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

                        //Verifica se a unidade tem uma função especificada, ou se vai usar uma função padrão
                        const unidadeTemFuncao : boolean = (this.layers_functions.length > 0 && this.layers_functions[l] && this.layers_functions[l][j]) ? true : false;
                        const nomeDaFuncao     : string = ( unidadeTemFuncao == true ? this.layers_functions[l][j] : 'Sigmoid' );
                    
                        layerError.push(error * ActivationFunctions[ `${nomeDaFuncao}Derivative` ](this.layerActivations[l][j]));
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

            let totalError:number = MLP.compute_train_cost( inputs, targets, inputs.map( (xsis: number[]) => this.forward(xsis) ) );

            // Log do erro para monitoramento
            if (epoch % printEpochs === 0) {
                console.log(`Epoch ${epoch}, Erro total: ${totalError}`);
            }
        }
    }

    // Função para prever a saída para um novo conjunto de entradas
    public estimate(input: number[]): number[] {
        const output = this.forward(input) as number[];
        return output.map( (o: number) => (o > 0.5 ? 1 : 0) );
    }
}
