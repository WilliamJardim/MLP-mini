
// Conteúdo do arquivo: C:\Users\Meu Computador\Desktop\Projetos Pessoais Github\Deep Learning\MLP-mini\dist\src\interfaces\DoneParameters.js
{};


// Conteúdo do arquivo: C:\Users\Meu Computador\Desktop\Projetos Pessoais Github\Deep Learning\MLP-mini\dist\src\interfaces\HyperParameters.js


// Conteúdo do arquivo: C:\Users\Meu Computador\Desktop\Projetos Pessoais Github\Deep Learning\MLP-mini\dist\src\interfaces\LayerDeclaration.js



// Conteúdo do arquivo: C:\Users\Meu Computador\Desktop\Projetos Pessoais Github\Deep Learning\MLP-mini\dist\src\interfaces\LayerInfo.js
{};


// Conteúdo do arquivo: C:\Users\Meu Computador\Desktop\Projetos Pessoais Github\Deep Learning\MLP-mini\dist\src\interfaces\MLPConfig.js



// Conteúdo do arquivo: C:\Users\Meu Computador\Desktop\Projetos Pessoais Github\Deep Learning\MLP-mini\dist\src\interfaces\TrackedStep.js
{};


// Conteúdo do arquivo: C:\Users\Meu Computador\Desktop\Projetos Pessoais Github\Deep Learning\MLP-mini\dist\src\interfaces\UnitInfo.js
{};


// Conteúdo do arquivo: C:\Users\Meu Computador\Desktop\Projetos Pessoais Github\Deep Learning\MLP-mini\dist\src\mlp.js











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
        /**
        * Inicio das iterações das epocas
        */
        let epoca = 0;
        /**
        * Para cada uma das epocas
        */
        while (epoca < epocas) {
            /**
            * Variavel usada para armazenar o contexto do modelo. Ou seja, o "this".
            * Ela é usada para conseguirmos acessar os pesos, biases, e outros atributos do modelo, usando contextoModelo.<ALGUMA_COISA>
            */
            const contextoModelo = this;
            /**
            * Para cada amostra
            */
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
            /**
            * Percorre novamente "a parte", todas as amostras de treinamento, e calcula o o erro total(o erro somado de todas as amostras)
            * Pra ajudar a verificar o progresso do treinamento
            */
            let erroTotal = MLP.compute_train_cost(amostras, metas, amostras.map((dadosAmostra) => contextoModelo.estimar(dadosAmostra)));
            // Monitorar o erro ao longo das épocas
            if (epoca % epocasMostrar === 0) {
                trainMonitor.log(`Epoch ${epoca}, Erro total: ${erroTotal}, Média Erro Total: ${erroTotal / metas.length}`);
            }
            //Pula pra proxima iteração de "epoca"
            epoca++;
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
    // Função de ativação Linear
    static Linear(x) {
        return x;
    }
    // Derivada da ativação Linear
    static LinearDerivative(x) {
        return 1;
    }
}


// Conteúdo do arquivo: C:\Users\Meu Computador\Desktop\Projetos Pessoais Github\Deep Learning\MLP-mini\dist\src\utils\AddInfoToLayersAndUnits.js
/**
*
* @param mlpConfig  - As configurações onde onde as descrições serão obtidas
* @param layersInfo - Objeto de dentro da MLP que vai armazenar os dados
*/
function AddInfoToLayersAndUnits(mlpConfig, layersInfo) {
    const infoConfig = mlpConfig.layerInfo || [];
    if (infoConfig.length > 0) {
        infoConfig.forEach((infoCamada, indiceLayerInfo) => {
            //Validações e tratamentos se necessário
            const infoAdicionarCamada = Object.assign({}, infoCamada);
            infoAdicionarCamada['layerIndex'] = indiceLayerInfo;
            if (!infoCamada['title']) {
                infoAdicionarCamada['title'] = `Camada ${indiceLayerInfo + 1}`;
            }
            if (!infoCamada['description']) {
                infoAdicionarCamada['description'] = `A camada ${indiceLayerInfo + 1}, cujo indice é ${indiceLayerInfo}. ${infoCamada['title'] ? 'Ela tem o titulo: "' + String(infoCamada['title']) + '"' : ''}`;
            }
            infoAdicionarCamada['layerIndex'] = indiceLayerInfo;
            layersInfo.push(infoAdicionarCamada);
        });
        //Caso nenhuma configuração seja passada via 'layerInfo'
    }
    else {
        //Percorre cada definição de camada, procurando por titulos
        if (mlpConfig.camadas) {
            mlpConfig.camadas.forEach((layerDeclaration, indiceLayerDeclaration) => {
                const tituloCamada = layerDeclaration.title || null;
                const infoAdicionarCamada = {
                    title: tituloCamada || `Camada ${indiceLayerDeclaration + 1}`,
                    layerIndex: indiceLayerDeclaration
                };
                //Gerado automaticamente
                infoAdicionarCamada['description'] = `A camada ${indiceLayerDeclaration + 1}, cujo indice é ${indiceLayerDeclaration}. ${tituloCamada ? 'Ela tem o titulo: "' + String(tituloCamada) + '"' : ''}`;
                layersInfo.push(infoAdicionarCamada);
            });
        }
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
    Initialization["RandomHeNormal"] = "RandomHeNormal";
    Initialization["RandomHeUniform"] = "RandomHeUniform";
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
    ActivationFunctionsNames["Linear"] = "Linear";
})(ActivationFunctionsNames || (ActivationFunctionsNames = {}));


// Conteúdo do arquivo: C:\Users\Meu Computador\Desktop\Projetos Pessoais Github\Deep Learning\MLP-mini\dist\src\utils\heNormal.js
// Função para inicializar pesos usando He Normal
function heNormal(nIn) {
    return Math.random() * Math.sqrt(2 / nIn) * (Math.random() < 0.5 ? -1 : 1);
}


// Conteúdo do arquivo: C:\Users\Meu Computador\Desktop\Projetos Pessoais Github\Deep Learning\MLP-mini\dist\src\utils\heUniform.js
// Função para inicializar pesos usando He Uniform
function heUniform(nIn) {
    const limit = Math.sqrt(6 / nIn);
    return Math.random() * 2 * limit - limit; // Gera valores entre -limit e +limit
}


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
    const camadas = config.camadas;
    const firstLayer = camadas[0];
    const lastLayer = camadas[camadas.length - 1];
    if (train_inputs.length != train_targets.length) {
        throw `No seu dataset voce tem ${train_inputs.length} linhas, porém, voce tem apenas ${train_targets.length} targets!`;
    }
    ;
    //Procura se não tem dados faltando
    for (let i = 0; i < train_inputs.length; i++) {
        let trainInputs = train_inputs[i];
        let targetTrainInputs = train_targets[i];
        const targetsIsArray = targetTrainInputs instanceof Array;
        if (!targetsIsArray) {
            throw `A variavel targets precisa ser um Array com ${lastLayer.units} elementos!`;
        }
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
    const camadas = config.camadas;
    const firstLayer = camadas[0];
    if (firstLayer.functions != undefined) {
        throw `A camada de entrada não pode ter o atributo 'functions' !`;
    }
    ;
    for (let i = 0; i < camadas.length; i++) {
        const currentLayer = camadas[i];
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
    const camadas = config.camadas;
    const firstLayer = camadas[0];
    const lastLayer = camadas[camadas.length - 1];
    if (firstLayer.type != LayerType.Input) {
        throw 'A primeira camada camada${ 0 } precisa ser a camada de entrada, do tipo LayerType.Input!';
    }
    if (lastLayer.type != LayerType.Final) {
        throw 'A ultima camada camada${ layers.length-1 } precisa ser a camada de saida final do modelo, do tipo LayerType.Final!';
    }
    for (let i = 0; i < camadas.length; i++) {
        const previousLayer = camadas[i - 1];
        const currentLayer = camadas[i];
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

