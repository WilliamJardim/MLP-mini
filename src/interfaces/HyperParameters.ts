export default interface HyperParameters{
    learningRate?: number,

    /**
    * Caso definnido, pode controlar se queremos derivar a camada de saida ou não
    */
    derivateFinalLayer?: boolean,

    /**
    * Se debugTrain for 'true', então, ele vai armazenar cada de treinamento passo dentro de um Array
    */
    debugTrain?: boolean,

    /**
    * Se liteTrack for 'true', então, ele não vai exportar parametros a cada passo de TrackedStep, para poupar memória
    */
    liteTrack?: boolean,
    
    /**
    * Caso seja definido, pode controlar se vamos ou não usar Bias no nosso modelo
    */
    useBias?: boolean,

    /**
    * Caso seja definido, pode ser usado para interromper o loop de treinamento caso a condição seja atendida
    */
    criterioParada?: Function 
}