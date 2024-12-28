import DoneParameters from './DoneParameters';
import MLPConfig from './MLPConfig';

export default interface TrackedStep{
    trackIndex: number,
    timestamp: number,
    date: Date,
    description: string,
    epoch: number,
    dataset: number[][],
    amostra: number[],
    indiceAmostra: number,
    estimativas: number[],
    metas: number[],
    gradientesUltimaCamada: number[], //Os gradientes apenas da camada final
    todosGradientesJuntos: number[][], //OS gradientes de todas as camadas(incluindo a camada final)
    initial_parameters: DoneParameters,
    parameters_before_update: DoneParameters,
    parameters_after_update?: DoneParameters,
    funcoes_camadas: string[][],
    mlpConfig: MLPConfig,
    oldStep?: TrackedStep
}