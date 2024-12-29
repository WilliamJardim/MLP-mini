
import MLPConfig from "../interfaces/MLPConfig";
import HyperParameters from '../interfaces/HyperParameters';

export default function ValidateCriterioParada( config:MLPConfig, hyperparameters: HyperParameters ){
    if( !hyperparameters.debugTrain && hyperparameters.criterioParada ){
        throw `Para usar um critério de parada você precisa ativar o Hyper Parametro "debugTrain", definindo ele como true.`;
    }
}