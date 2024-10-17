import LayerDeclaration from '../interfaces/LayerDeclaration';
import '../utils/Enums';

export default interface MLPConfig{
    
    initialization: Initialization,
    task: Task,
    traintype: TrainType,

    hyperparameters: HyperParameters,

    //Structure
    layers: Array<LayerDeclaration>,

    parameters: DoneParameters
}
