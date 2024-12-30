import MLPConfig from "../interfaces/MLPConfig";
import HyperParameters from '../interfaces/HyperParameters';
import LayerDeclaration from "../interfaces/LayerDeclaration";
import DoneParameters from "../interfaces/DoneParameters";
import LayerInfo from "../interfaces/UnitInfo";
import '../utils/Enums';

// Interpreta o config tanto em portugues quanto em ingles
export default function TranslateStructure( config: any ): void 
{
    //Obtem as propriedades seja em ingles ou em portugues
    const camadas: Array<LayerDeclaration>  = config.layers           || config.camadas;
    const inicializacao: Initialization     = config.initialization   || config.inicializacao;
    const tarefa:Task                       = config.task             || config.tarefa;
    const treino:TrainType                  = config.traintype        || config.treino;
    const hiperparametros:HyperParameters   = config.hyperparameters  || config.hiperparametros;
    const parametros:DoneParameters         = config.parameters       || config.parametros;
    const infoCadastras:LayerInfo[]         = config.layerInfo        || config.infoCamadas;

    //Define os valores no objeto MLPConfig
    if(camadas && JSON.stringify(config.camadas) != JSON.stringify(camadas) ){
        console.info('Definido camadas!');
        config.camadas = camadas;
    }
    if(inicializacao && config.initialization != inicializacao){
        console.info('Definido inicialização!');
        config.initialization = inicializacao;
    }
    if(tarefa && config.task != tarefa){
        console.info('Definido tarefa!');
        config.task = tarefa;
    }
    if(treino && config.traintype != treino){
        console.info('Definido tipo de treinamento!');
        config.traintype = treino;
    }
    if(hiperparametros && JSON.stringify(config.hyperparameters) != JSON.stringify(hiperparametros)){
        console.info('Definido hiperparametros!');
        config.hyperparameters = hiperparametros;
    }
    if(parametros && JSON.stringify(config.parameters) != JSON.stringify(parametros)){
        console.info('Definido parametros!');
        config.parameters = parametros;
    }
    if(infoCadastras && JSON.stringify(config.layerInfo) != JSON.stringify(infoCadastras)){
        console.info('Definido info camadas!');
        config.layerInfo = infoCadastras;
    }
}