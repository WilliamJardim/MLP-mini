class ActivationFunctions {
    // Torna a classe um singleton impedindo instanciamento externo
    constructor() { }
    // Função de ativação sigmoide
    static sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }
    // Derivada da sigmoide
    static sigmoidDerivative(x) {
        return x * (1 - x);
    }
    // Função de ativação ReLU
    static ReLU(x) {
        return Math.max(0, x);
    }
    // Derivada da ReLU
    static ReLUDerivative(x) {
        return x > 0 ? 1 : 0;
    }
}
var Initialization;
(function (Initialization) {
    Initialization["Zeros"] = "Zeros";
})(Initialization || (Initialization = {}));
var Task;
(function (Task) {
    Task["BinaryClassification"] = "binary_classification";
})(Task || (Task = {}));
var TrainType;
(function (TrainType) {
    TrainType["Online"] = "online";
})(TrainType || (TrainType = {}));
var LayerType;
(function (LayerType) {
    LayerType["Input"] = "input";
    LayerType["Hidden"] = "hidden";
    LayerType["Final"] = "final";
})(LayerType || (LayerType = {}));
var ActivationFunctionsNames;
(function (ActivationFunctionsNames) {
    ActivationFunctionsNames["Sigmoid"] = "sigmoid";
    ActivationFunctionsNames["ReLU"] = "ReLU";
})(ActivationFunctionsNames || (ActivationFunctionsNames = {}));
// Função para inicializar pesos de forma aleatória
function randomWeight() {
    return Math.random() * 2 - 1; // Gera valores entre -1 e 1
}
// Rede Neural MLP com suporte a múltiplas camadas
class MLP {
    constructor(config) {
        const classContext = this;
        this.config = config;
        // layers é um array onde cada elemento é o número de unidades na respectiva camada
        // Essa informação será extraida do config
        this.layers = [];
        this.config.layers.forEach(function (layerDeclaration, layerIndex) {
            classContext.layers[layerIndex] = layerDeclaration.units;
        });
        // Inicializando pesos e biases para todas as camadas
        this.weights = [];
        this.biases = [];
        for (let i = 1; i < this.layers.length; i++) {
            // Pesos entre a camada i-1 e a camada i
            const layerWeights = [];
            for (let j = 0; j < this.layers[i]; j++) {
                const neuronWeights = [];
                for (let k = 0; k < this.layers[i - 1]; k++) {
                    neuronWeights.push(randomWeight());
                }
                layerWeights.push(neuronWeights);
            }
            this.weights.push(layerWeights);
            // Biases para a camada i
            const layerBiases = Array(this.layers[i]).fill(0).map(() => randomWeight());
            this.biases.push(layerBiases);
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
    static compute_train_cost(inputs, mytargets, estimatedValues) {
        let cost = 0;
        inputs.forEach((input, i) => {
            const targets = mytargets[i];
            const estimations = estimatedValues[i];
            for (let S = 0; S < estimations.length; S++) {
                cost = cost + Math.pow((estimations[S] - targets[S]), 2);
            }
        });
        return cost;
    }
    /**
    * Retorna os parametros iniciais que foram usados para inicializar a rede
    */
    getInitialParameters() {
        return this.initialParameters;
    }
    /**
    * Log the current network parameters values in a string
    *
    * @param parameterShow - The show type
    */
    logParameters(parameterShow = 'verbose') {
        let netStr = '-=-=- WEIGHS OF THE NETWORK: -=-=- \n\n';
        let identSimbol = '--->';
        for (let l = 0; l < this.weights.length; l++) {
            netStr += `LAYER ${l}:\n `;
            for (let j = 0; j < this.weights[l].length; j++) {
                if (parameterShow == 'verbose') {
                    netStr += `     ${identSimbol} UNIT OF NUMBER ${j}:\n `;
                }
                else if (parameterShow == 'short') {
                    netStr += `     ${identSimbol} UNIT ${j}:\n `;
                }
                for (let k = 0; k < this.weights[l][j].length; k++) {
                    if (parameterShow == 'verbose') {
                        netStr += `          ${identSimbol} WEIGHT OF INPUT X${k}: ${this.weights[l][j][k]}\n `;
                    }
                    else if (parameterShow == 'short') {
                        netStr += `          ${identSimbol} W${j}${k}: ${this.weights[l][j][k]}\n `;
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
    exportParameters() {
        return (JSON.parse(JSON.stringify({
            weights: [...this.weights],
            biases: [...this.biases],
            layers: this.layers,
            //Other info
            generatedAt: new Date().getTime()
        })));
    }
    /**
    * Import the parameters intro this network
    * @param {parameters} - The JSON object that contain the weights and biases
    */
    importParameters(parameters) {
        console.log(`Loading parameters from JSON, from date: ${parameters.generatedAt}`);
        this.layers = parameters.layers;
        this.weights = parameters.weights;
        this.biases = parameters.biases;
        console.log(`Success from import JSON, from date: ${parameters.generatedAt}`);
    }
    // Forward pass (passagem direta)
    forward(input) {
        let activations = input;
        // Passar pelos neurônios de cada camada
        this.layerActivations = [activations]; // Para armazenar as ativações de cada camada
        for (let l = 0; l < this.weights.length; l++) {
            const nextActivations = [];
            for (let j = 0; j < this.weights[l].length; j++) {
                let weightedSum = 0;
                for (let k = 0; k < activations.length; k++) {
                    weightedSum += activations[k] * this.weights[l][j][k];
                }
                weightedSum += this.biases[l][j];
                nextActivations.push(ActivationFunctions.sigmoid(weightedSum));
            }
            activations = nextActivations;
            this.layerActivations.push(activations);
        }
        return activations;
    }
    // Função de treinamento com retropropagação
    train(inputs, targets, learningRate = 0.1, epochs = 10000, printEpochs = 1000) {
        console.log(`Erro inicial(ANTES DO TREINAMENTO): ${MLP.compute_train_cost(inputs, targets, inputs.map((xsis) => this.forward(xsis)))}`);
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
            let totalError = MLP.compute_train_cost(inputs, targets, inputs.map((xsis) => this.forward(xsis)));
            // Log do erro para monitoramento
            if (epoch % printEpochs === 0) {
                console.log(`Epoch ${epoch}, Erro total: ${totalError}`);
            }
        }
    }
    // Função para prever a saída para um novo conjunto de entradas
    estimate(input) {
        const output = this.forward(input);
        return output.map((o) => (o > 0.5 ? 1 : 0));
    }
}
