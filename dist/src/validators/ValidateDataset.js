import '../utils/Enums';
export default function ValidateDataset(config, train_inputs, train_targets) {
    const camadas = config.camadas;
    const firstLayer = camadas[0];
    const lastLayer = camadas[camadas.length - 1];
    if (train_inputs.length != train_targets.length) {
        throw `No seu dataset voce tem ${train_inputs.length} linhas, porém, voce tem apenas ${train_targets.length} targets!`;
    }
    ;
    //Procura se não tem dados faltando
    for (let i = 0; i < train_inputs.length; i++) {
        let trainInputs = train_inputs[i];
        let targetTrainInputs = train_targets[i];
        const targetsIsArray = targetTrainInputs instanceof Array;
        if (!targetsIsArray) {
            throw `A variavel targets precisa ser um Array com ${lastLayer.units} elementos!`;
        }
        if (trainInputs.length != firstLayer.inputs) {
            throw `O seu modelo de rede possui ${firstLayer.inputs} entradas, porém, seu dataset possui ${trainInputs.length} features na linha ${i}!. Dados precisam bater!`;
        }
        ;
        if (targetTrainInputs.length != lastLayer.units) {
            throw `A quantidade de targets da linha ${i} do seu dataset é ${targetTrainInputs.length}, sendo que na camada de saida da sua rede, voce tem ${lastLayer.units}. As quantidades precisam bater!`;
        }
        ;
    }
}
