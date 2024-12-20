// Função para inicializar pesos usando He Normal
export default function heNormal(nIn) {
    return Math.random() * Math.sqrt(2 / nIn) * (Math.random() < 0.5 ? -1 : 1);
}
