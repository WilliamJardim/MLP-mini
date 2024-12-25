# Arquivo Referencia
O arquivo "MLP-mini-faeef37cb0d6ac351cc03b30f5ff14507aea15a9.zip" contém um cópia completa do MLP-Mini, do dia 21/12/2024 

# Resumo do BUG corrigido e da melhoria feita
Hoje dia 24/12/2024, eu fiz uma nota atualização, e lancei a release "alpha1.0.0.13" do MLP-Mini. Nessa nova versão fiz uma melhoria no calculo dos gradientes da camada de saida, introduzindo a derivada da função de ativação das unidades da camada de saida. Ao introduzir esse novo elemento no cálculo dos gradientes, obviamente o teste de integridade do XOR sofreu a alteração, e precisou ser atualizado para continuar sendo usado nas proximas versões.
Tomei todo o cuidado de revisar cada linha do commit, pra garantir que nada importante foi alterado.
Em resumo a unica coisa que fiz foi adicionar 2 novas linhas de código e editar 1 linha, na parte que calcula os gradientes da camada de saida, conforme mencionado, e além disso, corrigi um BUG antigo, que fazia com que as funções de ativação da camada de saida fossem aplicadas nas camadas ocultas, devido a um BUG de indices. Felizmente, esse BUG não afetou nenhum teste do XOR e nenhum teste de classificação de digitos(do projeto [WilliamJardim/classificadordigitos](https://github.com/WilliamJardim/classificadordigitos)).

# Teste de integridade reforçado ainda mais!
Eu fui no meu repositório desse projeto MLP-Mini, no dia 21/12/2024, e baixei o arquivo zip do repositório do MLP-Mini ""MLP-mini-faeef37cb0d6ac351cc03b30f5ff14507aea15a9.zip", que continua os códigos mais atualizados ANTES DESSA GRANDE ATUALIZAÇÂO FEITA, para poder consultar os arquivos antigos do teste de integridade que estava antes da release "alpha1.0.0.13", com o intuito manter salvo os resultados de antes, a fim de comparar posteriormente. Eu sabia que, para o problema do XOR, o BUG que existia na versão release "alpha1.0.0.12" não afetou os testes de classificação como o classificação de digitos nem os do XOR, nem mesmo o teste de integridade, pois todos esses testes usavam função de ativação Sigmoid, então por isso não foram afetados pelo BUG, pois, se o indice da função de ativação não existisse no Array, ou se ele pegasse o nome da função de ativação errado, nada disso importaria, visto que todas as unidades nesses testes usam função de ativação Sigmoid, e tanto se não existir, quanto se acessar errado, o programa iria pegar a função Sigmoid, então por isso não afetou, por que a Sigmoid já era a função padrão. E eu sabia que se a unica coisa que mudou com a atualização que impactou o teste de integridade foi a melhoria no calculo dos gradientes da camada de saida por adicionar a derivada, então, eu tambem sabia que, se eu removesse essas 2 novas linhas e removesse o trecho "ActivationFunctions[ `${nomeDaFuncao}Derivative` ]( output[j] )", ficando assim igual estava antes:

## Trecho que estava antes
```javascript
    for (let j = 0; j < output.length; j++) {
        const error = target[j] - output[j] ;
        outputError.push(error);
    }
```

Então, eu sabia que isso teria que fazer o MLP-Mini voltar a produzir os resultados antigos do teste de integridade do XOR, os mesmos resultados de antes da atualização. E de fato isso aconteceu. Deu certo!

**Esses foram os resultados:**
```javascript
Erro Total inicial(ANTES DO TREINAMENTO): 1.006463576017077
bundle.js:409 Média do Erro Total inicial(ANTES DO TREINAMENTO): 0.25161589400426926
bundle.js:409 Epoch 0, Erro total: 1.0042949265382282, Média Erro Total: 0.25107373163455704
bundle.js:409 Epoch 1000, Erro total: 0.7573042453719532, Média Erro Total: 0.1893260613429883
bundle.js:409 Epoch 2000, Erro total: 0.006762246801441626, Média Erro Total: 0.0016905617003604065
bundle.js:409 Epoch 3000, Erro total: 0.0009645604240061526, Média Erro Total: 0.00024114010600153815
bundle.js:409 Epoch 4000, Erro total: 0.0003579708746998038, Média Erro Total: 0.00008949271867495095
bundle.js:409 Epoch 5000, Erro total: 0.0001837263766912989, Média Erro Total: 0.00004593159417282472
bundle.js:409 Epoch 6000, Erro total: 0.00011114722652080309, Média Erro Total: 0.000027786806630200772
bundle.js:409 Epoch 7000, Erro total: 0.00007427321273021063, Média Erro Total: 0.000018568303182552657
bundle.js:409 Epoch 8000, Erro total: 0.00005305475269371219, Média Erro Total: 0.000013263688173428048
bundle.js:409 Epoch 9000, Erro total: 0.000039753414805063484, Média Erro Total: 0.000009938353701265871
script.js:62 Estimativas:
script.js:65 Entrada: 0,0, Estimativa: 0
script.js:65 Entrada: 0,1, Estimativa: 1
script.js:65 Entrada: 1,0, Estimativa: 1
script.js:65 Entrada: 1,1, Estimativa: 0
script.js:209 Hash 1 desse resultado:  7e349ea735552563926e9fcbfae337b6d405abf916457af9bc6cce94e97e2c7e
script.js:217 Hash 2 desse resultado:  e727b358b88d8152b034da93750bdff29ae14e66a73b3f0649bf3abb5c029706
```

**Que foram exatamente os mesmos resultados que estavam antes(Que peguei do repositório):**
```javascript
Erro inicial(ANTES DO TREINAMENTO): 1.006463576017077
bundle.js:251 Epoch 0, Erro total: 1.0042949265382282
bundle.js:251 Epoch 1000, Erro total: 0.7573042453719532
bundle.js:251 Epoch 2000, Erro total: 0.006762246801441626
bundle.js:251 Epoch 3000, Erro total: 0.0009645604240061526
bundle.js:251 Epoch 4000, Erro total: 0.0003579708746998038
bundle.js:251 Epoch 5000, Erro total: 0.0001837263766912989
bundle.js:251 Epoch 6000, Erro total: 0.00011114722652080309
bundle.js:251 Epoch 7000, Erro total: 0.00007427321273021063
bundle.js:251 Epoch 8000, Erro total: 0.00005305475269371219
bundle.js:251 Epoch 9000, Erro total: 0.000039753414805063484
script.js:62 Estimativas:
script.js:65 Entrada: 0,0, Estimativa: 0
script.js:65 Entrada: 0,1, Estimativa: 1
script.js:65 Entrada: 1,0, Estimativa: 1
script.js:65 Entrada: 1,1, Estimativa: 0
script.js:163 Hash 1 desse resultado:  7e349ea735552563926e9fcbfae337b6d405abf916457af9bc6cce94e97e2c7e
script.js:171 Hash 2 desse resultado:  e727b358b88d8152b034da93750bdff29ae14e66a73b3f0649bf3abb5c029706
```

# VERIFIQUEI TAMBEM OS PESOS FINAIS LOGO EM SEGUIDA PRA VER SE BATEM COM OS DE ANTES
# PESOS
```javascript
[
    [
        [
            7.356551391331864,
            -7.512464010167511
        ],
        [
            -7.017645475366182,
            6.69359492863455
        ]
    ],
    [
        [
            12.728437039164444,
            12.855089565130744
        ]
    ]
]
```

# BIASES
```javascript
[
    [
        -4.016713934290134,
        -3.6312859187312987
    ],
    [
        -6.289294653592655
    ]
]
```

**Tanto os pesos quanto os biases são exatamente os que estavam antes tambem!**

# Conclusão
O algoritmo do backpropagation desta implantação continua exatamente o mesmo que era antes, não houve nenhuma alteração. E se remover o trecho que foi adicionado na atualização, os resultados do teste de integridade serão exatamente esses, que estavam antes da atualização!.

Depois de confirmar isso, eu voltei as linhas que eu tinha apagado, dando um rollback, pois quero manter. Mais isso serviu para confirmar esse fato.

Então, enquanto os resultados do teste de integridade forem estes abaixo:
```javascript
Erro Total inicial(ANTES DO TREINAMENTO): 1.006463576017077
bundle.js:405 Média do Erro Total inicial(ANTES DO TREINAMENTO): 0.25161589400426926
bundle.js:405 Epoch 0, Erro total: 1.0058870340846515, Média Erro Total: 0.2514717585211629
bundle.js:405 Epoch 1000, Erro total: 0.9982959622034995, Média Erro Total: 0.24957399055087487
bundle.js:405 Epoch 2000, Erro total: 0.9905320150991395, Média Erro Total: 0.24763300377478487
bundle.js:405 Epoch 3000, Erro total: 0.9221247945990458, Média Erro Total: 0.23053119864976146
bundle.js:405 Epoch 4000, Erro total: 0.7722593299006424, Média Erro Total: 0.1930648324751606
bundle.js:405 Epoch 5000, Erro total: 0.6760491879689282, Média Erro Total: 0.16901229699223205
bundle.js:405 Epoch 6000, Erro total: 0.1774392307956196, Média Erro Total: 0.0443598076989049
bundle.js:405 Epoch 7000, Erro total: 0.054887941979764945, Média Erro Total: 0.013721985494941236
bundle.js:405 Epoch 8000, Erro total: 0.03004105365104298, Média Erro Total: 0.007510263412760745
bundle.js:405 Epoch 9000, Erro total: 0.02022987983142562, Média Erro Total: 0.005057469957856405
script.js:62 Estimativas:
script.js:65 Entrada: 0,0, Estimativa: 0
script.js:65 Entrada: 0,1, Estimativa: 1
script.js:65 Entrada: 1,0, Estimativa: 1
script.js:65 Entrada: 1,1, Estimativa: 0
script.js:163 Hash 1 desse resultado:  e41f848d5c5d266ea8b0033faf2abdd2ece76c59b0d5af26fa5c347b2bc47de5
script.js:171 Hash 2 desse resultado:  9763f8b2f72727cfb3ed08053775cf74cc1ac48cc8d936b6a3be4706839e38f6
```

**Então, enquanto você ver esse resultado, esta tudo ok com a integridade, que permaneceu a mesma deis do dia 15.10.2024(dia em que foi feito o primeito commit do projeto).**