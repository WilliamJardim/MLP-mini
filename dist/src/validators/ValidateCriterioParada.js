export default function ValidateCriterioParada(config, hyperparameters) {
    if (!hyperparameters.debugTrain && hyperparameters.criterioParada) {
        throw `Para usar um critério de parada você precisa ativar o Hyper Parametro "debugTrain", definindo ele como true.`;
    }
}
