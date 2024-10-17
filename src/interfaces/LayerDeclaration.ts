import '../Enums';

export default interface LayerDeclaration{
    type      : LayerType,
    inputs    : number,
    units     : number,
    functions : Array<ActivationFunctionsNames>
}
