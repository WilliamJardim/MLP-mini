/**
*
* @param mlpConfig  - As configurações onde onde as descrições serão obtidas
* @param layersInfo - Objeto de dentro da MLP que vai armazenar os dados
*/
export default function AddInfoToLayersAndUnits(mlpConfig, layersInfo) {
    const infoConfig = mlpConfig.layerInfo || [];
    if (infoConfig.length > 0) {
        infoConfig.forEach((infoCamada, indiceLayerInfo) => {
            //Validações e tratamentos se necessário
            const infoAdicionarCamada = Object.assign({}, infoCamada);
            if (!infoCamada['title']) {
                infoAdicionarCamada['title'] = `Camada ${indiceLayerInfo + 1}`;
            }
            if (!infoCamada['description']) {
                infoAdicionarCamada['description'] = `A camada ${indiceLayerInfo + 1}, cujo indice é ${indiceLayerInfo}`;
            }
            infoAdicionarCamada['layerIndex'] = indiceLayerInfo;
            layersInfo.push(infoAdicionarCamada);
        });
    }
}
