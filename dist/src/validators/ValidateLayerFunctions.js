import '../utils/Enums';
export default function ValidateLayerFunctions(config) {
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
