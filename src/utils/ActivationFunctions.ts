export default class ActivationFunctions {
    // Torna a classe um singleton impedindo instanciamento externo
    private constructor() {}

    // Função de ativação sigmoide
    public static Sigmoid(x: number): number {
        return 1 / (1 + Math.exp(-x));
    }

    // Derivada da sigmoide
    public static SigmoidDerivative(x: number): number {
        return x * (1 - x);
    }

    // Função de ativação ReLU
    public static ReLU(x: number): number {
        return Math.max(0, x);
    }

    // Derivada da ReLU
    public static ReLUDerivative(x: number): number {
        return x > 0 ? 1 : 0;
    }
}
