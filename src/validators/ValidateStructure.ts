import LayerDeclaration from "../interfaces/LayerDeclaration";
import MLPConfig from "../interfaces/MLPConfig";
import '../utils/Enums';
import isDecimalNumber from "../utils/isDecimalNumber";

export default function ValidateStructure( config: MLPConfig ): void
{
    const initializationType:Initialization = config.initialization;

    if( !( initializationType in Initialization ) ){ throw `O tipo de inicialização não é um tipo valido de Initialization` };
    if( typeof initializationType != 'string' ){ throw `O atributo 'initialization' precisa ser do tipo 'string' ` };

    const layers:Array<LayerDeclaration> = config.layers;
    const firstLayer = layers[ 0 ];
    const lastLayer  = layers[layers.length-1];

    if( firstLayer.type != LayerType.Input ){
        throw 'A primeira camada camada${ 0 } precisa ser a camada de entrada, do tipo LayerType.Input!';
    }

    if( lastLayer.type != LayerType.Final ){
        throw 'A ultima camada camada${ layers.length-1 } precisa ser a camada de saida final do modelo, do tipo LayerType.Final!';
    }

    for( let i = 0 ; i < layers.length ; i++ )
    {
        const previousLayer = layers[i - 1];
        const currentLayer  = layers[  i  ];

        if( !currentLayer.type   ){ throw ` A camada ${ i } precisa ter um atributo 'type'! ` }
        if( !(currentLayer.type in LayerType) ){ throw `O atributo 'type' da camada ${ i } não é um valor valido de LayerType!` };
        if( !currentLayer.inputs ){ throw ` A camada ${ i } precisa ter o atributo 'inputs'! ` }
        if( !currentLayer.units  ){ throw ` A camada ${ i } precisa ter o atributo 'units'! ` }

        if( typeof currentLayer.type != 'string' ){ throw `O atributo 'type' da camada ${ i } precisa ser do tipo 'string' ` };
        if( isDecimalNumber(currentLayer.inputs) ){ throw ` O atributo 'inputs' da camada ${ i } precisa ser um número inteiro! ` }
        if( isDecimalNumber(currentLayer.units) ){ throw ` O atributo 'units' da camada ${ i } precisa ser um número inteiro! ` };

        if( previousLayer && currentLayer.inputs != previousLayer.units ){
            throw ` A camada camada${ i-1 } possui ${ previousLayer.units } saidas, porém a camada camada${ i } possui apenas ${ currentLayer.inputs } entradas! `;
        }
    }
}