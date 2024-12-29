// Função para inicializar pesos de forma aleatória
export default function VerificarCriteriosParada(epocaAtual, trainTracker, ultimoPasso, funcaoCriterio) {
    if (funcaoCriterio) {
        //Se ele retorna "true" ele para o loop do treinamento
        return funcaoCriterio(epocaAtual, ultimoPasso, trainTracker) == true;
    }
    //Caso não existe um críterio de parada, por padrão o retorno vai ser "false" pra não interferir no loop
    return false;
}
