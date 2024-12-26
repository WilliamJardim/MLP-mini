import '../utils/Enums';
import LayerInfo from './LayerInfo';

export default interface LayerDeclaration{
    type       : LayerType,
    inputs     : number,
    units      : number,
    functions  : Array<ActivationFunctionsNames>,
    title?     : string
}