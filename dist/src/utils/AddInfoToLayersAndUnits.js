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
            infoAdicionarCamada['layerIndex'] = indiceLayerInfo;
            if (!infoCamada['title']) {
                infoAdicionarCamada['title'] = `Camada ${indiceLayerInfo + 1}`;
            }
            if (!infoCamada['description']) {
                infoAdicionarCamada['description'] = `A camada ${indiceLayerInfo + 1}, cujo indice é ${indiceLayerInfo}. ${infoCamada['title'] ? 'Ela tem o titulo: "' + String(infoCamada['title']) + '"' : ''}`;
            }
            infoAdicionarCamada['layerIndex'] = indiceLayerInfo;
            layersInfo.push(infoAdicionarCamada);
        });
        //Caso nenhuma configuração seja passada via 'layerInfo'
    }
    else {
        //Percorre cada definição de camada, procurando por titulos
        if (mlpConfig.camadas) {
            mlpConfig.camadas.forEach((layerDeclaration, indiceLayerDeclaration) => {
                const tituloCamada = layerDeclaration.title || null;
                const infoAdicionarCamada = {
                    title: tituloCamada || `Camada ${indiceLayerDeclaration + 1}`,
                    layerIndex: indiceLayerDeclaration
                };
                //Gerado automaticamente
                infoAdicionarCamada['description'] = `A camada ${indiceLayerDeclaration + 1}, cujo indice é ${indiceLayerDeclaration}. ${tituloCamada ? 'Ela tem o titulo: "' + String(tituloCamada) + '"' : ''}`;
                layersInfo.push(infoAdicionarCamada);
            });
        }
    }
}
