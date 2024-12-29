import TrackedStep from "../interfaces/TrackedStep";

// Função para inicializar pesos de forma aleatória
export default function VerificarCriteriosParada( epocaAtual: number, 
                                                  trainTracker: TrackedStep[], 
                                                  ultimoPasso: TrackedStep,
                                                  funcaoCriterio?: Function ): boolean 
                                                  {
    if( funcaoCriterio ){
        //Se ele retorna "true" ele para o loop do treinamento
        return funcaoCriterio( epocaAtual, ultimoPasso, trainTracker ) == true;
    }

    //Caso não existe um críterio de parada, por padrão o retorno vai ser "false" pra não interferir no loop
    return false;
}