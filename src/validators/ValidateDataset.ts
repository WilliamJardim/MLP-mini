import LayerDeclaration from "../interfaces/LayerDeclaration";
import MLPConfig from "../interfaces/MLPConfig";
import '../utils/Enums';
import isDecimalNumber from "../utils/isDecimalNumber";

export default function ValidateDataset( config: MLPConfig, train_inputs: number[][], train_targets: number[][] ): void
{
    const layers:Array<LayerDeclaration> = config.layers;
    const firstLayer = layers[ 0 ];
    const lastLayer  = layers[layers.length-1];

    if( train_inputs.length != train_targets.length ){ throw `No seu dataset voce tem ${ train_inputs.length } linhas, porém, voce tem apenas ${ train_targets.length } targets!` };
    
    //Procura se não tem dados faltando
    for( let i = 0 ; i < train_inputs.length ; i++ )
    {
        let trainInputs       = train_inputs[i];
        let targetTrainInputs = train_targets[i];

        if( trainInputs.length != firstLayer.inputs ){ throw `O seu modelo de rede possui ${ firstLayer.inputs } entradas, porém, seu dataset possui ${ trainInputs.length } features na linha ${ i }!. Dados precisam bater!` };
        if( targetTrainInputs.length != lastLayer.units ){ throw `A quantidade de targets da linha ${i} do seu dataset é ${ targetTrainInputs.length }, sendo que na camada de saida da sua rede, voce tem ${ lastLayer.units }. As quantidades precisam bater!` };
    }
}