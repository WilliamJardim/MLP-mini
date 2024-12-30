import '../utils/Enums';
// Interpreta o config tanto em portugues quanto em ingles
export default function TranslateStructure(config) {
    //Obtem as propriedades seja em ingles ou em portugues
    const camadas = config.layers || config.camadas;
    const inicializacao = config.initialization || config.inicializacao;
    const tarefa = config.task || config.tarefa;
    const treino = config.traintype || config.treino;
    const hiperparametros = config.hyperparameters || config.hiperparametros;
    const parametros = config.parameters || config.parametros;
    const infoCadastras = config.layerInfo || config.infoCamadas;
    //Define os valores no objeto MLPConfig
    if (camadas && JSON.stringify(config.camadas) != JSON.stringify(camadas)) {
        console.info('Definido camadas!');
        config.camadas = camadas;
    }
    if (inicializacao && config.initialization != inicializacao) {
        console.info('Definido inicialização!');
        config.initialization = inicializacao;
    }
    if (tarefa && config.task != tarefa) {
        console.info('Definido tarefa!');
        config.task = tarefa;
    }
    if (treino && config.traintype != treino) {
        console.info('Definido tipo de treinamento!');
        config.traintype = treino;
    }
    if (hiperparametros && JSON.stringify(config.hyperparameters) != JSON.stringify(hiperparametros)) {
        console.info('Definido hiperparametros!');
        config.hyperparameters = hiperparametros;
    }
    if (parametros && JSON.stringify(config.parameters) != JSON.stringify(parametros)) {
        console.info('Definido parametros!');
        config.parameters = parametros;
    }
    if (infoCadastras && JSON.stringify(config.layerInfo) != JSON.stringify(infoCadastras)) {
        console.info('Definido info camadas!');
        config.layerInfo = infoCadastras;
    }
}
