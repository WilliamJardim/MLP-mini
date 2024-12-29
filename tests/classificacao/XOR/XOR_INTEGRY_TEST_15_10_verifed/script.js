// Estrutura da rede: 2 neurônios na entrada, 2 na camada oculta, 1 na saída
const mlp = new MLP({
    camadas: [
        { type: LayerType.Input,  inputs: 2, units: 2, title: 'Camada de entrada' },
        { type: LayerType.Hidden, inputs: 2, units: 2, title: 'Camada oculta' },
        { type: LayerType.Final,  inputs: 2, units: 1, title: 'Camada de saida' }
    ],
    initialization: Initialization.Dev,

    //Extra parameters
    hyperparameters: {
        /**
        * Esse parametro determina Se vai aplicar a derivada na camada de saida ou não.
        * Caso você use 'false', você vai ter os resultados do teste de integridade de antes da atualização do dia 24/12/2024, é uma forma rápida de você testar isso se quiser
        * 
        * LEIA SOBRE ISSO EM: https://github.com/WilliamJardim/MLP-mini/blob/main/tests/classificacao/XOR/XOR_INTEGRY_TEST_15_10_verifed/VERS%C3%82O%20DE%2023-12%20pra%20traz/NOTAS-RETRO-INTEGRIDADE.md
        */
        derivateFinalLayer: true,

        /**
        * Esse parametro quando for 'true', permite capturar informações durante o treinamento,
        * ele captura tudo, deis de pesos iniciais, pesos atuais ANTES DE APLICAR O GRADIENTE DESCEDENTE, gradientes calculados DESSA EPOCA, features das amostras, valores esperados para a amostra, valores estimados para a amostra 
        */
        debugTrain: true,

        /**
        * Esse parametro determina se vai usar Bias ou não 
        */
        useBias: true,

        /**
        * Configura uma condição de parada(além do limite de épocas) 
        * Exemplo de uso: funcaoCriterio( epocaAtual, ultimoPasso, trainTracker );
        */
        criterioParada: function( epocaAtual, ultimoPasso, trainTracker ){
            return false; //Não existe nenhum critério definido
        }
    },

});

var pesosIniciais = [
    [
        [
            0.8228850033675079,
            -0.314907800152612
        ],
        [
            0.001901923545564177,
            0.6076617485704823
        ]
    ],
    [
        [
            0.21803494362838416,
            0.13302177857890918
        ]
    ]
];
mlp.pesos = [...pesosIniciais.copyWithin()];

var biasesIniciais = [
    [
        -0.6336502693201962,
        0.9156237345346292
    ],
    [
        -0.02445825279113123
    ]
]
mlp.biases = [...biasesIniciais.copyWithin()];

// Dados de entrada para o problema XOR
const inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
];

// Saídas esperadas para o XOR
const targets = [
    [0],
    [1],
    [1],
    [0]
];

// Treinando a rede
mlp.treinar(inputs, targets, 0.1, 10000);

// Testando a rede
console.log('Estimativas:');
inputs.forEach(input => {
    const output = mlp.estimar(input);
    console.log(`Entrada: ${input}, Estimativa: ${output}`);
});

/**
 * O RESULTADO PRECISA BATER COM ISSO 
 
    Erro Total inicial(ANTES DO TREINAMENTO): 1.006463576017077
    bundle.js:409 Média do Erro Total inicial(ANTES DO TREINAMENTO): 0.25161589400426926
    bundle.js:409 Epoch 0, Erro total: 1.0058870340846515, Média Erro Total: 0.2514717585211629
    bundle.js:409 Epoch 1000, Erro total: 0.9982959622034995, Média Erro Total: 0.24957399055087487
    bundle.js:409 Epoch 2000, Erro total: 0.9905320150991395, Média Erro Total: 0.24763300377478487
    bundle.js:409 Epoch 3000, Erro total: 0.9221247945990458, Média Erro Total: 0.23053119864976146
    bundle.js:409 Epoch 4000, Erro total: 0.7722593299006424, Média Erro Total: 0.1930648324751606
    bundle.js:409 Epoch 5000, Erro total: 0.6760491879689282, Média Erro Total: 0.16901229699223205
    bundle.js:409 Epoch 6000, Erro total: 0.1774392307956196, Média Erro Total: 0.0443598076989049
    bundle.js:409 Epoch 7000, Erro total: 0.054887941979764945, Média Erro Total: 0.013721985494941236
    bundle.js:409 Epoch 8000, Erro total: 0.03004105365104298, Média Erro Total: 0.007510263412760745
    bundle.js:409 Epoch 9000, Erro total: 0.02022987983142562, Média Erro Total: 0.005057469957856405
    script.js:62 Estimativas:
    script.js:65 Entrada: 0,0, Estimativa: 0
    script.js:65 Entrada: 0,1, Estimativa: 1
    script.js:65 Entrada: 1,0, Estimativa: 1
    script.js:65 Entrada: 1,1, Estimativa: 0
    script.js:209 Hash 1 desse resultado:  e41f848d5c5d266ea8b0033faf2abdd2ece76c59b0d5af26fa5c347b2bc47de5
    script.js:217 Hash 2 desse resultado:  9763f8b2f72727cfb3ed08053775cf74cc1ac48cc8d936b6a3be4706839e38f6

    NOTA: Esse é o arquivo atualizado hoje de 24/12/2024 após a correção do pequeno BUG e da melhoria no calculo dos gradientes na camada de saida.

    Todas as versões anteriores desse teste de integridade que sempre foi feito por mim a cada atualizado, serviram para confirmar a exatidão do código até o dia 23/12/2024. 
    E hoje apenas algumas mudanças e correções de bugs foram feitas. Então, nada critico mudou, apenas o calculo do gradiente que eu melhorei. Então os resultados do teste de integridade vão mudar a partir de hoje 24/12/2024, e vou precisar salvar a nova versão que será usada daqui pra frente.

 */

/**
Erro Total inicial(ANTES DO TREINAMENTO): 1.006463576017077
bundle.js:409 Média do Erro Total inicial(ANTES DO TREINAMENTO): 0.25161589400426926
bundle.js:409 Epoch 0, Erro total: 1.0058870340846515, Média Erro Total: 0.2514717585211629
bundle.js:409 Epoch 1000, Erro total: 0.9982959622034995, Média Erro Total: 0.24957399055087487
bundle.js:409 Epoch 2000, Erro total: 0.9905320150991395, Média Erro Total: 0.24763300377478487
bundle.js:409 Epoch 3000, Erro total: 0.9221247945990458, Média Erro Total: 0.23053119864976146
bundle.js:409 Epoch 4000, Erro total: 0.7722593299006424, Média Erro Total: 0.1930648324751606
bundle.js:409 Epoch 5000, Erro total: 0.6760491879689282, Média Erro Total: 0.16901229699223205
bundle.js:409 Epoch 6000, Erro total: 0.1774392307956196, Média Erro Total: 0.0443598076989049
bundle.js:409 Epoch 7000, Erro total: 0.054887941979764945, Média Erro Total: 0.013721985494941236
bundle.js:409 Epoch 8000, Erro total: 0.03004105365104298, Média Erro Total: 0.007510263412760745
bundle.js:409 Epoch 9000, Erro total: 0.02022987983142562, Média Erro Total: 0.005057469957856405
script.js:62 Estimativas:
script.js:65 Entrada: 0,0, Estimativa: 0
script.js:65 Entrada: 0,1, Estimativa: 1
script.js:65 Entrada: 1,0, Estimativa: 1
script.js:65 Entrada: 1,1, Estimativa: 0
script.js:209 Hash 1 desse resultado:  e41f848d5c5d266ea8b0033faf2abdd2ece76c59b0d5af26fa5c347b2bc47de5
script.js:217 Hash 2 desse resultado:  9763f8b2f72727cfb3ed08053775cf74cc1ac48cc8d936b6a3be4706839e38f6

    NOTA: Esse é o arquivo atualizado hoje de 24/12/2024 após a correção do pequeno BUG e da melhoria no calculo dos gradientes na camada de saida.

    Todas as versões anteriores desse teste de integridade que sempre foi feito por mim a cada atualizado, serviram para confirmar a exatidão do código até o dia 23/12/2024. 
    E hoje apenas algumas mudanças e correções de bugs foram feitas. Então, nada critico mudou, apenas o calculo do gradiente que eu melhorei. Então os resultados do teste de integridade vão mudar a partir de hoje 24/12/2024, e vou precisar salvar a nova versão que será usada daqui pra frente.

 */


/**
 * os pesos finais precisam ser
 * 
 * [
    [
        [
            5.726599606168386,
            -5.717709974814535
        ],
        [
            -4.691954578251698,
            4.404570163172981
        ]
    ],
    [
        [
            6.821772619520277,
            6.991271031905877
        ]
    ]
]

NOTA: Esses pesos finais foram reanotados, após a atualização de hoje de 24/12/2024 após a correção do pequeno BUG e da melhoria no calculo dos gradientes na camada de saida.
Todas as versões anteriores desse teste de integridade que sempre foi feito por mim a cada atualizado, serviram para confirmar a exatidão do código até o dia 23/12/2024. 
E hoje apenas algumas mudanças e correções de bugs foram feitas. Então, nada critico mudou, apenas o calculo do gradiente que eu melhorei. Então os resultados do teste de integridade vão mudar a partir de hoje 24/12/2024, e vou precisar salvar a nova versão que será usada daqui pra frente. 

PORÈM, PERCEBI ALGO INTERESSANTE NOS PESOS FINAIS
embora com a adição da derivada no calculo do gradiente na camada de saida,.. embora os pesos finais tenham mudado um pouco,.. a proporção e o sentido continuam o mesmo
por exemplo, no primeiro peso "5.726599606168386" antes era "7.35..." o numero continuou positivo.
outro exemplo, no segundo peso " -5.717709974814535" antes era "-7.51", o numero continuou negativo
outro exemplo, no terceiro peso "-4.691954578251698" antes era "-7.017", o o numero continuou negativo
outro exemplo, no quarto peso "4.404570163172981" antes era "6.69", o numero continuou positivo
outro exemplo, no quinto peso "6.821772619520277" antes era "12.72", o numero continuou positivo
outro exemplo, no sexto peso "6.991271031905877" antes era "12.85", o numero continuou positivo

EXATAMENTE A MESMA COISA ACONTECEU PARA OS BIASES
* 
 */


/**
 * os biases finais precisam ser
 * 
 *[
    [
        -3.357152585406333,
        -2.452189841946596
    ],
    [
        -3.4036318948427606
    ]
]

NOTA: assim como os pesos, Esses biases finais foram reanotados, após a atualização de hoje de 24/12/2024 após a correção do pequeno BUG e da melhoria no calculo dos gradientes na camada de saida.
Todas as versões anteriores desse teste de integridade que sempre foi feito por mim a cada atualizado, serviram para confirmar a exatidão do código até o dia 23/12/2024. 
E hoje apenas algumas mudanças e correções de bugs foram feitas. Então, nada critico mudou, apenas o calculo do gradiente que eu melhorei. Então os resultados do teste de integridade vão mudar a partir de hoje 24/12/2024, e vou precisar salvar a nova versão que será usada daqui pra frente. 

PORÈM COMO MENCIONEI NAS ANOTAÇÔES ACIMA, PERCEBI ALGO INTERESSANTE NOS BIASES FINAIS
embora com a adição da derivada no calculo do gradiente na camada de saida,.. embora os biases finais tenham mudado um pouco,.. a proporção e o sentido continuam o mesmo

Por exemplo, no primeiro bias "-3.357152585406333" antes era "-4.016", o número continuou negativo.
Outro exemplo, no segundo bias: "-2.452189841946596" antes era "-3.63", o número continuou negativo.
Outro exemplo, no terceiro bias "-3.4036318948427606" antes era "-6.28", o número continuou negativo.

Isso é muito interessante, e refleta proporção e sentido mantidos!
 
 */


async function generateHash(input) {
    const encoder = new TextEncoder();
    const data = encoder.encode(input);
    const hashBuffer = await crypto.subtle.digest('SHA-256', data);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    const hashHex = hashArray.map(byte => byte.toString(16).padStart(2, '0')).join('');
    return hashHex;
}

// Exemplo de geraçao de hash do resultado
generateHash( String( inputs ) + ' ' +
              String( targets ) +  ' ' +
              String( mlp.pesos ) +
              String( mlp.biases ) ).then(hash => console.log('Hash 1 desse resultado: ', hash));
  
// Exemplo de geraçao de hash do resultado
generateHash( String( inputs ) + ' ' +
              String( targets ) +  ' ' +
              String( pesosIniciais ) +
              String( biasesIniciais ) + 
              String( mlp.pesos ) +
              String( mlp.biases ) ).then(hash => console.log('Hash 2 desse resultado: ', hash));
 