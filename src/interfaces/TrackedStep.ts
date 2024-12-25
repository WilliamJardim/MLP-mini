import DoneParameters from './DoneParameters';
import MLPConfig from './MLPConfig';

export default interface TrackedStep{
    timestamp: number,
    date: Date,
    description: string,
    epoch: number,
    dataset: number[][],
    amostra: number[],
    indiceAmostra: number,
    output: number[],
    target: number[],
    finalLayerGradients: number[], //Os gradientes apenas da camada final
    allLayersGradients: number[][], //OS gradientes de todas as camadas(incluindo a camada final)
    initial_parameters: DoneParameters,
    parameters_before_update: DoneParameters,
    parameters_after_update?: DoneParameters,
    layers_functions: string[][],
    mlpConfig: MLPConfig
}