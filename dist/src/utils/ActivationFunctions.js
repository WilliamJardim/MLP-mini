export default class ActivationFunctions {
    // Torna a classe um singleton impedindo instanciamento externo
    constructor() { }
    // Função de ativação sigmoide
    static Sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }
    // Derivada da sigmoide
    static SigmoidDerivative(x) {
        return x * (1 - x);
    }
    // Função de ativação ReLU
    static ReLU(x) {
        return Math.max(0, x);
    }
    // Derivada da ReLU
    static ReLUDerivative(x) {
        return x > 0 ? 1 : 0;
    }
    // Função de ativação Linear
    static Linear(x) {
        return x;
    }
    // Derivada da ativação Linear
    static LinearDerivative(x) {
        return 1;
    }
}
