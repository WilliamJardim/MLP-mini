import LayerDeclaration from '../interfaces/LayerDeclaration';
import DoneParameters from '../interfaces/DoneParameters';
import '../utils/Enums';
import LayerInfo from './UnitInfo';
import HyperParameters from './HyperParameters';

export default interface MLPConfig{
    
    initialization: Initialization,
    task: Task,
    traintype: TrainType,

    hyperparameters: HyperParameters,

    //Structure
    camadas: Array<LayerDeclaration>,

    parameters: DoneParameters,

    //Opcionais
    layerInfo: LayerInfo[]
}
