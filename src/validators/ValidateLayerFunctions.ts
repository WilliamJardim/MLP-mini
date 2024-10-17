import LayerDeclaration from "../interfaces/LayerDeclaration";
import MLPConfig from "../interfaces/MLPConfig";
import '../utils/Enums';
import isDecimalNumber from "../utils/isDecimalNumber";

export default function ValidateLayerFunctions( config:MLPConfig ){
    const layers:Array<LayerDeclaration> = config.layers;

    for( let i = 0 ; i < layers.length ; i++ )
    {
        const currentLayer  = layers[  i  ];

        if( currentLayer.functions && currentLayer.functions.length != currentLayer.units ){
            throw ` A camada camada${i} tem ${currentLayer.functions.length} funções, sendo elas [${currentLayer.functions}], porém, essa camada possui ${ currentLayer.units } unidades. A quantidade nao bate! `;
        }
    }
}