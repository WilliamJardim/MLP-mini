
// Conteúdo do arquivo: C:\Users\Meu Computador\Desktop\Projetos Pessoais Github\Deep Learning\MLP-mini\dist\src\interfaces\DoneParameters.js


// Conteúdo do arquivo: C:\Users\Meu Computador\Desktop\Projetos Pessoais Github\Deep Learning\MLP-mini\dist\src\interfaces\HyperParameters.js


// Conteúdo do arquivo: C:\Users\Meu Computador\Desktop\Projetos Pessoais Github\Deep Learning\MLP-mini\dist\src\interfaces\LayerDeclaration.js



// Conteúdo do arquivo: C:\Users\Meu Computador\Desktop\Projetos Pessoais Github\Deep Learning\MLP-mini\dist\src\interfaces\MLPConfig.js



// Conteúdo do arquivo: C:\Users\Meu Computador\Desktop\Projetos Pessoais Github\Deep Learning\MLP-mini\dist\src\mlp.js








// Rede Neural MLP com suporte a múltiplas camadas
class MLP {
    constructor(config) {
        this.geralMonitor = new ConsoleMonitor({
            name: 'GeralConsole'
        });
        this.config = config;
        // Aplica uma validação de estrutura 
        ValidateStructure(this.config);
        // layers é um array onde cada elemento é o número de unidades na respectiva camada
        // Essa informação será extraida do config
        this.layers = [];
        //Esse aqui é um array para armazenar os nomes das funções de ativações das unidades de cada camada, assim: Array de Array<string>
        this.layers_functions = [];
        for (let layerIndex = 0; layerIndex < this.config.layers.length; layerIndex++) {
            const layerDeclaration = this.config.layers[layerIndex];
            this.layers[layerIndex] = layerDeclaration.units;
        }
        //Identifica quais as funções que cada unidade de cada camada usa,
        //Ignora a camada de entrada que não possui funções
        for (let layerIndex = 1; layerIndex < this.config.layers.length; layerIndex++) {
            const layerDeclaration = this.config.layers[layerIndex];
            //Usei - 1 pra ignorar a camada de entrada, e ordenar corretamente
            this.layers_functions[layerIndex - 1] = layerDeclaration.functions;
        }
        //Adicionar validação aqui para validar as funções das camadas
        if (this.layers_functions.length > 0) {
            //Se tiver this.layers_functions, então ele precisa validar
            ValidateLayerFunctions(this.config);
        }
        // Inicializando pesos e biases para todas as camadas
        this.weights = [];
        this.biases = [];
        if (config.initialization == Initialization.Random) {
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
        }
        else if (config.initialization == Initialization.Zeros) {
            for (let i = 1; i < this.layers.length; i++) {
                // Pesos entre a camada i-1 e a camada i
                const layerWeights = [];
                for (let j = 0; j < this.layers[i]; j++) {
                    const neuronWeights = [];
                    for (let k = 0; k < this.layers[i - 1]; k++) {
                        neuronWeights.push(0);
                    }
                    layerWeights.push(neuronWeights);
                }
                this.weights.push(layerWeights);
                // Biases para a camada i
                const layerBiases = Array(this.layers[i]).fill(0).map(() => 0);
                this.biases.push(layerBiases);
            }
        }
        else if (config.initialization == Initialization.Manual) {
            this.importParameters(config.parameters);
        }
        else if (config.initialization == Initialization.Dev) {
            //Aqui fica por conta do programador definir os parametros antes de tentar usar o modelo
        }
        //Faz a exportação dos parametros iniciais
        this.initialParameters = this.exportParameters();
    }
    //Obtem logs
    getLogs() {
        return this.geralMonitor.getHistory();
    }
    //Obtem o console geral
    getMonitor() {
        return this.geralMonitor;
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
        return {
            weights: JSON.parse(JSON.stringify([...this.weights])),
            biases: JSON.parse(JSON.stringify([...this.biases])),
            layers: this.layers,
            //Other info
            generatedAt: new Date().getTime()
        };
    }
    /**
    * Import the parameters intro this network
    * @param {parameters} - The JSON object that contain the weights and biases
    */
    importParameters(parameters) {
        console.log(`Loading parameters from JSON, from date: ${parameters.generatedAt}`);
        this.layers = [...JSON.parse(JSON.stringify(parameters.layers))];
        this.weights = [...JSON.parse(JSON.stringify(parameters.weights))];
        this.biases = [...JSON.parse(JSON.stringify(parameters.biases))];
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
                //Verifica se a unidade tem uma função especificada, ou se vai usar uma função padrão
                const unidadeTemFuncao = (this.layers_functions.length > 0 && this.layers_functions[l] && this.layers_functions[l][j]) ? true : false;
                const nomeDaFuncao = (unidadeTemFuncao == true ? this.layers_functions[l][j] : 'Sigmoid');
                nextActivations.push(ActivationFunctions[nomeDaFuncao](weightedSum));
                /** NaN detector */
                if (notifyIfhasNaN('feedforward/loops', [
                    weightedSum,
                    this.biases[l][j],
                    this.weights[l][j],
                    ActivationFunctions[nomeDaFuncao](weightedSum)
                ]).hasNaN) {
                    debugger;
                }
                ;
            }
            activations = nextActivations;
            this.layerActivations.push(activations);
        }
        return activations;
    }
    // Função de treinamento com retropropagação
    train(inputs, targets, learningRate = 0.1, epochs = 10000, printEpochs = 1000) {
        let trainMonitor = new ConsoleMonitor({
            name: 'TrainConsole'
        });
        // Garante que os parametros iniciais sejam arquivados ANTES DO TREINAMENTO COMEÇAR
        this.initialParameters = this.exportParameters();
        // Valida os dados de treinamento
        ValidateDataset(this.config, inputs, targets);
        trainMonitor.log(`Erro inicial(ANTES DO TREINAMENTO): ${MLP.compute_train_cost(inputs, targets, inputs.map((xsis) => this.forward(xsis)))}`);
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
                        const unidadeTemFuncao = (this.layers_functions.length > 0 && this.layers_functions[l] && this.layers_functions[l][j]) ? true : false;
                        const nomeDaFuncao = (unidadeTemFuncao == true ? this.layers_functions[l][j] : 'Sigmoid');
                        layerError.push(error * ActivationFunctions[`${nomeDaFuncao}Derivative`](this.layerActivations[l][j]));
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
                trainMonitor.log(`Epoch ${epoch}, Erro total: ${totalError}`);
            }
        }
        //Integra os logs atuais do treinamento no geral
        this.geralMonitor.integrate([
            trainMonitor
        ]);
    }
    // Função para prever a saída para um novo conjunto de entradas
    estimate(input) {
        const output = this.forward(input);
        return output.map((o) => (o > 0.5 ? 1 : 0));
    }
}


// Conteúdo do arquivo: C:\Users\Meu Computador\Desktop\Projetos Pessoais Github\Deep Learning\MLP-mini\dist\src\utils\ActivationFunctions.js
class ActivationFunctions {
    // Torna a classe um singleton impedindo instanciamento externo
    constructor() { }
    // Função de ativação sigmoide
    static Sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }
    // Derivada da sigmoide
    static SigmoidDerivative(x) {
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


// Conteúdo do arquivo: C:\Users\Meu Computador\Desktop\Projetos Pessoais Github\Deep Learning\MLP-mini\dist\src\utils\ConsoleMonitor.js
class ConsoleMonitor {
    constructor(config) {
        this.lines = '';
        this.history = [];
        this.config = config;
        this.name = config.name;
    }
    getConsoleName() {
        return this.name;
    }
    asString() {
        return this.lines;
    }
    getHistory() {
        return this.history;
    }
    push(info) {
        this.getHistory().push(info);
    }
    updateString() {
        let currentHistory = this.getHistory();
        this.lines = '';
        //Para cada console vinculado
        for (let i = 0; i < currentHistory.length; i++) {
            this.lines += currentHistory[i].message + '\n';
        }
    }
    /**
    * Integra o conteudo de outros ConsoleMonitor(es) a esse
    */
    integrate(from) {
        this.isIntegrator = true;
        //Para cada console vinculado
        for (let i = 0; i < from.length; i++) {
            //Extrai as informações e acrescenta elas na lista
            let currentLogs = from[i].getHistory();
            let consoleName = from[i].getConsoleName();
            currentLogs.forEach((info) => {
                this.push(Object.assign({}, info));
            });
        }
        this.updateString();
    }
    log(message, aparence = 'white', classes = []) {
        console.log(message);
        this.lines = this.lines + message + '\n';
        this.history.push({
            aparence: aparence,
            message: message,
            classes: classes,
            timestamp: new Date().getTime(),
            date: new Date()
        });
    }
    reset() {
        this.lines = '';
        this.history = [];
    }
}


// Conteúdo do arquivo: C:\Users\Meu Computador\Desktop\Projetos Pessoais Github\Deep Learning\MLP-mini\dist\src\utils\Enums.js
var Initialization;
(function (Initialization) {
    Initialization["Zeros"] = "Zeros";
    Initialization["Manual"] = "Manual";
    Initialization["Random"] = "Random";
    Initialization["Dev"] = "Dev";
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
    LayerType["Input"] = "Input";
    LayerType["Hidden"] = "Hidden";
    LayerType["Final"] = "Final";
})(LayerType || (LayerType = {}));
var ActivationFunctionsNames;
(function (ActivationFunctionsNames) {
    ActivationFunctionsNames["Sigmoid"] = "sigmoid";
    ActivationFunctionsNames["ReLU"] = "ReLU";
})(ActivationFunctionsNames || (ActivationFunctionsNames = {}));


// Conteúdo do arquivo: C:\Users\Meu Computador\Desktop\Projetos Pessoais Github\Deep Learning\MLP-mini\dist\src\utils\isDecimalNumber.js
function isDecimalNumber(x) {
    return String(x).indexOf('.') != -1 ? true : false;
}


// Conteúdo do arquivo: C:\Users\Meu Computador\Desktop\Projetos Pessoais Github\Deep Learning\MLP-mini\dist\src\utils\notifyIfhasNaN.js
var jaFoi = {};
function notifyIfhasNaN(title, varToCheck, callback) {
    let nanValues = [];
    let hasNaN = false;
    varToCheck.forEach((val, valIndex) => {
        if (val instanceof Array) {
            let resultSub = notifyIfhasNaN(title + '_array', val);
            nanValues = [...resultSub.values, nanValues];
            hasNaN = resultSub.hasNaN;
        }
        else {
            if (isNaN(val)) {
                nanValues.push(valIndex);
                if (!jaFoi[title]) {
                    console.warn(title, 'NaN', valIndex, 'please insert debugger');
                    jaFoi[title] = true;
                }
                hasNaN = true;
            }
        }
    });
    let result = { hasNaN: hasNaN, values: nanValues };
    if (hasNaN && callback) {
        callback(result);
    }
    return result;
}


// Conteúdo do arquivo: C:\Users\Meu Computador\Desktop\Projetos Pessoais Github\Deep Learning\MLP-mini\dist\src\utils\randomWeight.js
// Função para inicializar pesos de forma aleatória
function randomWeight() {
    return Math.random() * 2 - 1; // Gera valores entre -1 e 1
}


// Conteúdo do arquivo: C:\Users\Meu Computador\Desktop\Projetos Pessoais Github\Deep Learning\MLP-mini\dist\src\validators\ValidateDataset.js

function ValidateDataset(config, train_inputs, train_targets) {
    const layers = config.layers;
    const firstLayer = layers[0];
    const lastLayer = layers[layers.length - 1];
    if (train_inputs.length != train_targets.length) {
        throw `No seu dataset voce tem ${train_inputs.length} linhas, porém, voce tem apenas ${train_targets.length} targets!`;
    }
    ;
    //Procura se não tem dados faltando
    for (let i = 0; i < train_inputs.length; i++) {
        let trainInputs = train_inputs[i];
        let targetTrainInputs = train_targets[i];
        if (trainInputs.length != firstLayer.inputs) {
            throw `O seu modelo de rede possui ${firstLayer.inputs} entradas, porém, seu dataset possui ${trainInputs.length} features na linha ${i}!. Dados precisam bater!`;
        }
        ;
        if (targetTrainInputs.length != lastLayer.units) {
            throw `A quantidade de targets da linha ${i} do seu dataset é ${targetTrainInputs.length}, sendo que na camada de saida da sua rede, voce tem ${lastLayer.units}. As quantidades precisam bater!`;
        }
        ;
    }
}


// Conteúdo do arquivo: C:\Users\Meu Computador\Desktop\Projetos Pessoais Github\Deep Learning\MLP-mini\dist\src\validators\ValidateLayerFunctions.js

function ValidateLayerFunctions(config) {
    const layers = config.layers;
    const firstLayer = layers[0];
    if (firstLayer.functions != undefined) {
        throw `A camada de entrada não pode ter o atributo 'functions' !`;
    }
    ;
    for (let i = 0; i < layers.length; i++) {
        const currentLayer = layers[i];
        if (currentLayer.functions) {
            currentLayer.functions.forEach(function (nomeFn) {
                if (!(nomeFn in ActivationFunctionsNames)) {
                    throw `${nomeFn} não é uma função valida!, veja ActivationFunctionsNames`;
                }
                ;
            });
            if (currentLayer.functions.length != currentLayer.units) {
                throw `A camada camada${i} tem ${currentLayer.functions.length} funções, sendo elas [${currentLayer.functions}], porém, essa camada possui ${currentLayer.units} unidades. A quantidade nao bate! `;
            }
            ;
        }
    }
}


// Conteúdo do arquivo: C:\Users\Meu Computador\Desktop\Projetos Pessoais Github\Deep Learning\MLP-mini\dist\src\validators\ValidateStructure.js


function ValidateStructure(config) {
    const initializationType = config.initialization;
    if (!(initializationType in Initialization)) {
        throw `O tipo de inicialização não é um tipo valido de Initialization`;
    }
    ;
    if (typeof initializationType != 'string') {
        throw `O atributo 'initialization' precisa ser do tipo 'string' `;
    }
    ;
    const layers = config.layers;
    const firstLayer = layers[0];
    const lastLayer = layers[layers.length - 1];
    if (firstLayer.type != LayerType.Input) {
        throw 'A primeira camada camada${ 0 } precisa ser a camada de entrada, do tipo LayerType.Input!';
    }
    if (lastLayer.type != LayerType.Final) {
        throw 'A ultima camada camada${ layers.length-1 } precisa ser a camada de saida final do modelo, do tipo LayerType.Final!';
    }
    for (let i = 0; i < layers.length; i++) {
        const previousLayer = layers[i - 1];
        const currentLayer = layers[i];
        if (!currentLayer.type) {
            throw ` A camada ${i} precisa ter um atributo 'type'! `;
        }
        if (!(currentLayer.type in LayerType)) {
            throw `O atributo 'type' da camada ${i} não é um valor valido de LayerType!`;
        }
        ;
        if (!currentLayer.inputs) {
            throw ` A camada ${i} precisa ter o atributo 'inputs'! `;
        }
        if (!currentLayer.units) {
            throw ` A camada ${i} precisa ter o atributo 'units'! `;
        }
        if (typeof currentLayer.type != 'string') {
            throw `O atributo 'type' da camada ${i} precisa ser do tipo 'string' `;
        }
        ;
        if (isDecimalNumber(currentLayer.inputs)) {
            throw ` O atributo 'inputs' da camada ${i} precisa ser um número inteiro! `;
        }
        if (isDecimalNumber(currentLayer.units)) {
            throw ` O atributo 'units' da camada ${i} precisa ser um número inteiro! `;
        }
        ;
        if (previousLayer && currentLayer.inputs != previousLayer.units) {
            throw ` A camada camada${i - 1} possui ${previousLayer.units} saidas, porém a camada camada${i} possui apenas ${currentLayer.inputs} entradas! `;
        }
    }
}

