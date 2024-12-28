import ActivationFunctions from './utils/ActivationFunctions';
import './utils/Enums';
import ValidateStructure from './validators/ValidateStructure';
import ValidateDataset from './validators/ValidateDataset';
import ValidateLayerFunctions from './validators/ValidateLayerFunctions';
import notifyIfhasNaN from './utils/notifyIfhasNaN';
import randomWeight from './utils/randomWeight';
import heNormal from './utils/heNormal';
import heUniform from './utils/heUniform';
import ConsoleMonitor from './utils/ConsoleMonitor';
import AddInfoToLayersAndUnits from './utils/AddInfoToLayersAndUnits';
// Rede Neural MLP com suporte a múltiplas camadas
class MLP {
    constructor(config) {
        this.layersInfo = []; //A Array of LayerInfo(s)
        this.geralMonitor = new ConsoleMonitor({
            name: 'GeralConsole'
        });
        this.config = config;
        this.hyperparameters = config.hyperparameters;
        //Se não for especificado, por padrão, ele vai calcular a derivada da camada de saida, exceto se o usuário quiser mudar isso
        if (this.hyperparameters != undefined &&
            (this.hyperparameters.derivateFinalLayer == undefined || this.hyperparameters.derivateFinalLayer == null)) {
            this.hyperparameters.derivateFinalLayer = true;
        }
        else {
            //Caso não seja nem memso passado hyper parametros, ele ja define por padrão alguns, pra evitar erros
            if (this.hyperparameters == undefined) {
                this.hyperparameters = {
                    derivateFinalLayer: true,
                    debugTrain: false
                };
            }
        }
        //Se não for especificado, por padrão, ele vai usar o Bias, exceto se o usuário quiser mudar isso
        if (this.hyperparameters != undefined &&
            (this.hyperparameters.useBias == undefined || this.hyperparameters.useBias == null)) {
            this.hyperparameters.useBias = true;
        }
        // Aplica uma validação de estrutura 
        ValidateStructure(this.config);
        // Cria descrições nas camadas pra facilitar o debugging
        AddInfoToLayersAndUnits(this.config, this.layersInfo);
        // camadas é um array onde cada elemento é o número de unidades na respectiva camada
        // Essa informação será extraida do config
        this.camadas = [];
        //Esse aqui é um array para armazenar os nomes das funções de ativações das unidades de cada camada, assim: Array de Array<string>
        this.funcoes_camadas = [];
        for (let layerIndex = 0; layerIndex < this.config.camadas.length; layerIndex++) {
            const layerDeclaration = this.config.camadas[layerIndex];
            this.camadas[layerIndex] = layerDeclaration.units;
        }
        //Identifica quais as funções que cada unidade de cada camada usa,
        //Ignora a camada de entrada que não possui funções
        for (let layerIndex = 1; layerIndex < this.config.camadas.length; layerIndex++) {
            const layerDeclaration = this.config.camadas[layerIndex];
            //Usei - 1 pra ignorar a camada de entrada, e ordenar corretamente
            this.funcoes_camadas[layerIndex - 1] = layerDeclaration.functions;
        }
        //Adicionar validação aqui para validar as funções das camadas
        if (this.funcoes_camadas.length > 0) {
            //Se tiver this.funcoes_camadas, então ele precisa validar
            ValidateLayerFunctions(this.config);
        }
        // Cria um Array de pesos e biases para armazenar os pesos e biases de cada camada
        this.pesos = [];
        this.biases = [];
        if (config.initialization == Initialization.Random) {
            for (let i = 1; i < this.camadas.length; i++) {
                // Pesos entre a camada i-1 e a camada i
                const pesosCamada = [];
                for (let j = 0; j < this.camadas[i]; j++) {
                    const pesosUnidade = [];
                    for (let k = 0; k < this.camadas[i - 1]; k++) {
                        pesosUnidade.push(randomWeight());
                    }
                    pesosCamada.push(pesosUnidade);
                }
                this.pesos.push(pesosCamada);
                // Biases para a camada i
                const biasesCamada = [];
                for (let b = 0; b < this.camadas[i]; b++) {
                    biasesCamada.push(randomWeight());
                }
                this.biases.push(biasesCamada);
            }
        }
        else if (config.initialization === Initialization.RandomHeNormal || config.initialization === Initialization.RandomHeUniform) {
            for (let i = 1; i < this.camadas.length; i++) {
                // Pesos entre a camada i-1 e a camada i
                const pesosCamada = [];
                for (let j = 0; j < this.camadas[i]; j++) {
                    const pesosUnidade = [];
                    for (let k = 0; k < this.camadas[i - 1]; k++) {
                        if (config.initialization === Initialization.RandomHeNormal) {
                            pesosUnidade.push(heNormal(this.camadas[i - 1]));
                        }
                        else if (config.initialization === Initialization.RandomHeUniform) {
                            pesosUnidade.push(heUniform(this.camadas[i - 1]));
                        }
                    }
                    pesosCamada.push(pesosUnidade);
                }
                this.pesos.push(pesosCamada);
                // Gerando Biases da camada i
                const biasesCamada = [];
                for (let b = 0; b < this.camadas[i]; b++) {
                    biasesCamada.push(0);
                }
                this.biases.push(biasesCamada);
            }
        }
        else if (config.initialization == Initialization.Zeros) {
            for (let i = 1; i < this.camadas.length; i++) {
                // Pesos entre a camada i-1 e a camada i
                const pesosCamada = [];
                for (let j = 0; j < this.camadas[i]; j++) {
                    const pesosUnidade = [];
                    for (let k = 0; k < this.camadas[i - 1]; k++) {
                        pesosUnidade.push(0);
                    }
                    pesosCamada.push(pesosUnidade);
                }
                this.pesos.push(pesosCamada);
                // Biases para a camada i
                const biasesCamada = [];
                for (let b = 0; b < this.camadas[i]; b++) {
                    biasesCamada.push(0);
                }
                this.biases.push(biasesCamada);
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
    static compute_train_cost(amostras, myMetas, valoresEstimados) {
        let custo = 0;
        amostras.forEach((amostra, numAmostra) => {
            const metas = myMetas[numAmostra];
            const estimativas = valoresEstimados[numAmostra];
            for (let numValor = 0; numValor < estimativas.length; numValor++) {
                custo = custo + Math.pow((estimativas[numValor] - metas[numValor]), 2);
            }
        });
        return custo;
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
        for (let l = 0; l < this.pesos.length; l++) {
            netStr += `LAYER ${l}:\n `;
            for (let j = 0; j < this.pesos[l].length; j++) {
                if (parameterShow == 'verbose') {
                    netStr += `     ${identSimbol} UNIT OF NUMBER ${j}:\n `;
                }
                else if (parameterShow == 'short') {
                    netStr += `     ${identSimbol} UNIT ${j}:\n `;
                }
                for (let k = 0; k < this.pesos[l][j].length; k++) {
                    if (parameterShow == 'verbose') {
                        netStr += `          ${identSimbol} WEIGHT OF INPUT X${k}: ${this.pesos[l][j][k]}\n `;
                    }
                    else if (parameterShow == 'short') {
                        netStr += `          ${identSimbol} W${j}${k}: ${this.pesos[l][j][k]}\n `;
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
            pesos: JSON.parse(JSON.stringify([...this.pesos])),
            biases: JSON.parse(JSON.stringify([...this.biases])),
            layers: this.camadas,
            //Other info
            generatedAt: new Date().getTime()
        };
    }
    /**
    * Importa os parametros para essa rede
    * @param {parameters} - O objeto JSON que contém os pesos e biases
    */
    importParameters(parameters) {
        console.log(`Loading parameters from JSON, from date: ${parameters.generatedAt}`);
        this.camadas = [...JSON.parse(JSON.stringify(parameters.layers))];
        this.pesos = [...JSON.parse(JSON.stringify(parameters.pesos))];
        this.biases = [...JSON.parse(JSON.stringify(parameters.biases))];
        console.log(`Success from import JSON, from date: ${parameters.generatedAt}`);
    }
    // Feedforward (fazer as estimativas)
    estimar(amostra) {
        let ativacoes = amostra;
        // Passar pelos neurônios de cada camada
        this.ativacoesPorCamada = [ativacoes]; // Para armazenar as ativações de cada camada
        for (let l = 0; l < this.pesos.length; l++) {
            const proximasAtivacoes = [];
            for (let j = 0; j < this.pesos[l].length; j++) {
                let acumulacao = 0;
                for (let k = 0; k < ativacoes.length; k++) {
                    acumulacao += ativacoes[k] * this.pesos[l][j][k];
                }
                //Se vai incluir o Bias
                if (this.hyperparameters.useBias == true) {
                    acumulacao += this.biases[l][j];
                }
                //Verifica se a unidade tem uma função especificada, ou se vai usar uma função padrão
                const unidadeTemFuncao = (this.funcoes_camadas.length > 0 && this.funcoes_camadas[l] && this.funcoes_camadas[l][j]) ? true : false;
                const nomeDaFuncao = (unidadeTemFuncao == true ? this.funcoes_camadas[l][j] : 'Sigmoid');
                proximasAtivacoes.push(ActivationFunctions[nomeDaFuncao](acumulacao));
                /** NaN detector */
                if (notifyIfhasNaN('feedforward/loops', [
                    acumulacao,
                    this.biases[l][j],
                    this.pesos[l][j],
                    ActivationFunctions[nomeDaFuncao](acumulacao)
                ]).hasNaN) {
                    debugger;
                }
                ;
            }
            ativacoes = proximasAtivacoes;
            this.ativacoesPorCamada.push(ativacoes);
        }
        return ativacoes;
    }
    // Função de treinamento com retropropagação
    treinar(amostras, metas, learningRate = 0.1, epocas = 10000, epocasMostrar = 1000) {
        this.trainTracker = [];
        let trainMonitor = new ConsoleMonitor({
            name: 'TrainConsole'
        });
        // Garante que os parametros iniciais sejam arquivados ANTES DO TREINAMENTO COMEÇAR
        this.initialParameters = this.exportParameters();
        // Valida os dados de treinamento
        ValidateDataset(this.config, amostras, metas);
        const erroInicialAntesTreinamento = MLP.compute_train_cost(amostras, metas, amostras.map((dadosAmostra) => this.estimar(dadosAmostra)));
        trainMonitor.log(`Erro Total inicial(ANTES DO TREINAMENTO): ${erroInicialAntesTreinamento}`);
        trainMonitor.log(`Média do Erro Total inicial(ANTES DO TREINAMENTO): ${erroInicialAntesTreinamento / metas.length}`);
        let contagemTrackedStep = 0;
        for (let epoca = 0; epoca < epocas; epoca++) {
            const contextoModelo = this;
            amostras.forEach(function (amostra, numAmostra) {
                // Calculando o erro na camada de saida
                const meta = metas[numAmostra];
                const estimativas = contextoModelo.estimar(amostra);
                const gradientesFinais = [];
                for (let j = 0; j < estimativas.length; j++) {
                    //Verifica se a unidade tem uma função especificada, ou se vai usar uma função padrão
                    const unidadeTemFuncao = (contextoModelo.funcoes_camadas.length > 0 && contextoModelo.funcoes_camadas[contextoModelo.pesos.length - 1] && contextoModelo.funcoes_camadas[contextoModelo.pesos.length - 1][j]) ? true : false;
                    const nomeDaFuncao = (unidadeTemFuncao == true ? contextoModelo.funcoes_camadas[contextoModelo.pesos.length - 1][j] : 'Sigmoid');
                    const diferencaValores = meta[j] - estimativas[j];
                    //Adiciona essa diferença acima dentro de "gradientesFinais"
                    gradientesFinais.push(
                    /** MULTIPLICADA POR <O_VALOR_DA_CONDIÇÂO_ABAIXO> **/
                    diferencaValores * (
                    /**
                    * Se é pra derivar a camada de saida, ele pega a derivada da função de ativação dessa unidade da camada de saida.
                    * Caso o contrário, ele deixa 1 para não afetar em nada a diferença
                    */
                    (contextoModelo.hyperparameters != undefined && contextoModelo.hyperparameters.derivateFinalLayer == true)
                        ? ActivationFunctions[`${nomeDaFuncao}Derivative`](estimativas[j])
                        : 1));
                }
                // Aplica o algoritmo do Backpropagation
                const gradientesPorCamada = [gradientesFinais];
                // Calcular os erros das camadas ocultas, começando da última camada
                for (let l = contextoModelo.pesos.length - 1; l >= 1; l--) {
                    const gradientesCamadaAtual = [];
                    // Para cada unidade da camada oculta atual
                    for (let j = 0; j < contextoModelo.pesos[l - 1].length; j++) {
                        //Calcula o gradiente com relação a unidade J atual da camada oculta atual
                        let numeroUnidadeOculta = j;
                        let quantidadeUnidadesCamadaSeguinte = contextoModelo.pesos[l].length;
                        let gradienteUnidade = 0;
                        /**
                        * Para cada unidade na camada seguinte
                        */
                        for (let k = 0; k < quantidadeUnidadesCamadaSeguinte; k++) {
                            /**
                            * OBS: Ao dizer gradienteUnidadeSeguinte ou seja "gradiente da unidada seguinte" aqui, estou me referindo a unidade da camada seguinte
                            * Ou seja, a camada seguinte é próxima camada que vem depois da camada oculta atual
                            */
                            const gradienteUnidadeSeguinte = gradientesPorCamada[0][k];
                            const pesosUnidadeSeguinte = contextoModelo.pesos[l][k];
                            const pesoQueLiga = pesosUnidadeSeguinte[numeroUnidadeOculta];
                            gradienteUnidade += gradienteUnidadeSeguinte * pesoQueLiga;
                        }
                        //Verifica se a unidade tem uma função especificada, ou se vai usar uma função padrão
                        const unidadeTemFuncao = (contextoModelo.funcoes_camadas.length > 0 && contextoModelo.funcoes_camadas[l - 1] && contextoModelo.funcoes_camadas[l - 1][j]) ? true : false;
                        const nomeDaFuncao = (unidadeTemFuncao == true ? contextoModelo.funcoes_camadas[l - 1][j] : 'Sigmoid');
                        gradientesCamadaAtual.push(gradienteUnidade *
                            ActivationFunctions[`${nomeDaFuncao}Derivative`](contextoModelo.ativacoesPorCamada[l][j]));
                    }
                    /**
                    * Adiciona os erros das camada oculta atual como sendo o primeiro elemento do array "gradientesPorCamada", e os demais elementos que já existem no array ficam atráz dele, sequencialmente.
                    * Isso por que gradientesPorCamada[0] sempre vai retornar os erros da ultima camada calculada pelo Backpropagation
                    *
                    * E é por isso que gradientesPorCamada[0] começa sendo os erros da camada de saida(ou seja da última camada da rede)
                    * e na segunda interação, do for "for (let l = this.pesos.length - 1; l >= 1; l--) {", ao chegar nesse unshift, gradientesPorCamada[0] passa a ser os erros da penultima camada oculta
                    * e assim por diante
                    */
                    gradientesPorCamada.unshift(gradientesCamadaAtual);
                }
                //Se for pra debugar o treinamento
                let dadosDebugAmostra = null;
                if (contextoModelo.hyperparameters != undefined && contextoModelo.hyperparameters.debugTrain == true) {
                    dadosDebugAmostra = {
                        trackIndex: contagemTrackedStep || 0,
                        timestamp: new Date().getTime(),
                        date: new Date(),
                        description: `Epoca ${epoca}`,
                        epoch: epoca, //O numero da epoca atual
                        dataset: Array.from([...amostras]), //O dataset inteiro
                        amostra: Array.from([...amostra]), //Os dados da amostra atual
                        indiceAmostra: numAmostra, //O indice da amostra atual no dataset
                        estimativas: Array.from([...estimativas]), //Os valores estimados para a amostra atual
                        metas: Array.from([...meta]), //Os valores esperados para a amostra atual
                        gradientesUltimaCamada: Array.from([...gradientesFinais]), //Os gradientes calculados da camada de saida DESSA EPOCA
                        todosGradientesJuntos: Array.from([...gradientesPorCamada]), //Os gradientes calculados pelo backpropagation, de todas as camadas, DESSA EPOCA(inclusive a camada de saida)
                        initial_parameters: contextoModelo.getInitialParameters(), //Os parametros iniciais ANTES DO TREINAMENTO COMEÇAR
                        parameters_before_update: contextoModelo.exportParameters(), //Os parametros DE ANTES DE APLICAR O GRADIENTE DESCEDENTE DESTA EPOCA
                        funcoes_camadas: contextoModelo.funcoes_camadas,
                        mlpConfig: contextoModelo.config //As configurações usadas para criar a MLP
                    };
                }
                // Atualização dos pesos e biases
                for (let l = contextoModelo.pesos.length - 1; l >= 0; l--) {
                    for (let j = 0; j < contextoModelo.pesos[l].length; j++) {
                        for (let k = 0; k < contextoModelo.pesos[l][j].length; k++) {
                            // Atualiza os pesos com o Gradiente Descedente
                            contextoModelo.pesos[l][j][k] += learningRate * gradientesPorCamada[l][j] * contextoModelo.ativacoesPorCamada[l][k];
                        }
                        //Se estiver usando o Bias
                        if (contextoModelo.hyperparameters.useBias == true) {
                            // Atualiza os biases com o Gradiente Descedentes
                            // Aqui usamos vezes 1 pois a derivada em relação ao Bias é 1, pois não tem entrada, então só sobra o propio Bias
                            contextoModelo.biases[l][j] += learningRate * gradientesPorCamada[l][j] * 1;
                        }
                    }
                }
                //Se for pra debugar o treinamento
                if (contextoModelo.hyperparameters != undefined && contextoModelo.hyperparameters.debugTrain == true) {
                    dadosDebugAmostra['parameters_after_update'] = contextoModelo.exportParameters();
                    //Se existe um passo anterior cadastrado
                    if (contagemTrackedStep > 0) {
                        dadosDebugAmostra['oldStep'] = contextoModelo.trainTracker[contagemTrackedStep - 1];
                    }
                    contextoModelo.trainTracker.push(dadosDebugAmostra);
                    contagemTrackedStep++; //Atualiza o ID do rastreio dos passos
                }
            });
            let erroTotal = MLP.compute_train_cost(amostras, metas, amostras.map((dadosAmostra) => contextoModelo.estimar(dadosAmostra)));
            // Monitorar o erro ao longo das épocas
            // Log do erro para monitoramento
            if (epoca % epocasMostrar === 0) {
                trainMonitor.log(`Epoch ${epoca}, Erro total: ${erroTotal}, Média Erro Total: ${erroTotal / metas.length}`);
            }
        }
        //Integra os logs atuais do treinamento no geral
        this.geralMonitor.integrate([
            trainMonitor
        ]);
        contagemTrackedStep = 0; //Depois que o treinamento é concluido, ele zera a variavel usada para gerar os IDs dos Trackers
    }
    // Função para fazer a estimativas de novos valores de forma mais amigavel(útil para classificação)
    estimarPronto(amostra, limiar = 0.5) {
        const estimativas = this.estimar(amostra);
        return estimativas.map((saida) => (saida > limiar ? 1 : 0));
    }
}
