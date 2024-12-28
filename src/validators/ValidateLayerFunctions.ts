import LayerDeclaration from "../interfaces/LayerDeclaration";
import MLPConfig from "../interfaces/MLPConfig";
import '../utils/Enums';
import isDecimalNumber from "../utils/isDecimalNumber";

export default function ValidateLayerFunctions( config:MLPConfig ){
    const camadas:Array<LayerDeclaration> = config.camadas;
    const firstLayer:LayerDeclaration = camadas[0];

    if( firstLayer.functions != undefined ){ throw `A camada de entrada não pode ter o atributo 'functions' !` };

    for( let i = 0 ; i < camadas.length ; i++ )
    {
        const currentLayer  = camadas[  i  ];

        if( currentLayer.functions )
        {
            currentLayer.functions.forEach(function( nomeFn ){
                if( !(nomeFn in ActivationFunctionsNames) ){ throw `${nomeFn} não é uma função valida!, veja ActivationFunctionsNames` };
            });

            if( currentLayer.functions.length != currentLayer.units ){ throw `A camada camada${i} tem ${currentLayer.functions.length} funções, sendo elas [${currentLayer.functions}], porém, essa camada possui ${ currentLayer.units } unidades. A quantidade nao bate! ` };
        }
    }
}