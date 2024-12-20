// Função para inicializar pesos usando He Uniform
export default function heUniform(nIn: number): number {
    const limit = Math.sqrt(6 / nIn);
    return Math.random() * 2 * limit - limit; // Gera valores entre -limit e +limit
}