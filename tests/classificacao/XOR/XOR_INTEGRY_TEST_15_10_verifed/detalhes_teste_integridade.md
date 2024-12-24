Eu fiz esse teste hoje 16.10
eu baixei os arquivos do primeiro commit da minha rede neural MLP no meu github, do primeiro commit de ontem dia 15.10, antes da refatoração de hoje.

EU USEI ESSE PONTO DO MEU RESPOTÒRIO:
[ Link para o repositório na versão inicial dele de 15.10.2024 ](https://github.com/WilliamJardim/MLP-mini/tree/91bbd8bee309b5ae4cc27c234724a6cfc07941fe)

Eu gerei 9 valores aleatórios na mão para serem os pesos e biases estaticos que vou usar no teste

# Código usado para gerar os pesos e biases
Para isso, usei o seguinte código JavaScript mesmo diretamente no console do navegador:
```javascript
for( let p = 0 ; p < 10 ; p++){ 
    console.log( Math.random() * 2 - 1 ) 
};
```
Ai eu fui no editor de código, e fui pegando sequencialmente do console do navegador os números, e fui colocando lá nas posições do array weights e biases

Os resultados dos pesos e bias em sequencia(que eu obtive na geração aleatorio descrita cima) e defini como sendo pesos e biases foram as seguintes:
0.8228850033675079,
-0.314907800152612
0.001901923545564177,
0.6076617485704823
0.21803494362838416,
0.13302177857890918
-0.6336502693201962,
0.9156237345346292
-0.02445825279113123
(Eu me lembro muito bem desses valores de hoje quando eu estava construindo esse teste)

Eu organizei da seguinte forma:
```json
mlp.weights = [
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

mlp.biases = [
    [
        -0.6336502693201962,
        0.9156237345346292
    ],
    [
        -0.02445825279113123
    ]
]
```

esses são os pesos iniciais(que vão ser sempre os mesmos de forma estatica/fixa )

e criei esse arquivo script.html que serve para testar a integridade
eu coloquei esses 9 pesos estaticos nos pesos e biases(que eu gerei aleatorio na mão pelo metodo Math.random() * 2 - 1 )
e coloquei o dataset do XOR

e depois, rodei, vi que o treinamento começou com erro 1.006 e foi caindo gradualmente até ficar bem baixinho 0.000000 alguma coisa
como notei que esse resultado estaria legal para ser um exemplo, eu salvei os pesos iniciais(antes do treinamento QUE SÂO OS PESOS QUE EU GEREI ALEATORIAMENTE) e os pesos finiais(após o treinamento)
salvei essas informações no script.js, que contem os parametros iniciais, e os parametros finais, bem como o resultado que deve continuar sendo exatamente o mesmo em todas as futuras executações

CONCLUSAO:
Eu fiz o script.js dentro da pasta do primeiro commit, que eu baixei( fiz o scrips.js do "new integry" ), conforme expliquei acima, detalhando os passos de criação do script,
abri o arquivo de anotações que eu havia criado la dentro da pasta do commit antigo(anotando as informações da executação do script.js do modelo antigo antes das minhas mudanças de hoje dia 16.10) conform expliquei 
e comparei valor por valor, a olho mesmo, e posso garantir pra mim mesmo que NADA MUDOU, está tudo ok, os resultados da executação do script.js no modelo MLP antigo de ontem dia 15.10, foram exatamente os mesmos resultados da execução após as atualizações e refatorações que fiz hoje 16.10
confirmei que o treinamento e backpropagation não sofreram nenhuma alteração acidental.

as mudanças que fiz não afetaram em nada os resultados que já estavam sendo produzidos antes.

# Hashes do teste
Os hashs estão abaixo:
- Hash 1: 7e349ea735552563926e9fcbfae337b6d405abf916457af9bc6cce94e97e2c7e
- Hash 2: e727b358b88d8152b034da93750bdff29ae14e66a73b3f0649bf3abb5c029706

Eles não devem mudar!

# Resultados esperados que toda execução deve produzir para estar intacta
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

# Esclarecimentos
NOTA: Esse é o arquivo atualizado hoje de 24/12/2024 após a correção do pequeno BUG e da melhoria no calculo dos gradientes na camada de saida.

Todas as versões anteriores desse teste de integridade que sempre foi feito por mim a cada atualizado, serviram para confirmar a exatidão do código até o dia 23/12/2024. 
E hoje apenas algumas mudanças e correções de bugs foram feitas. Então, nada critico mudou, apenas o calculo do gradiente que eu melhorei. Então os resultados do teste de integridade vão mudar a partir de hoje 24/12/2024, e vou precisar salvar a nova versão que será usada daqui pra frente.

Nos testes anteriores, eu posso comprovar que o algoritmo do backpropagation dessa implantação não sofreu alterações, ele deu exatamente o mesmo resultado do custo ao longo das epocas que o script do teste da versão do dia 15.10.2024
o resultado é o mesmo, eu tenho os arquivos iniciais que usei pra fazer esse teste pela primeira vez, dentro da pasta do ZIP extraido do commit antigo que eu mencionei nas outras notas.
Isso é comprovado.

# Exatidão e Integridade mantidas deis do inicio!
Para obter esse historico de erro ao longo das epocas, eu rodei novamente o "script.html"[Veja ele aqui](./script.html) do teste de integridade 
Eu rodei novamente, e como de esperado, como não houve nenhuma alteração no algoritmo, o resultado permaneceu exatamente o mesmo descrito no arquivo(resultados deis do primeiro dia do projeto até o dia 23/12/2024) [Compare aqui](./resultado-atualizado.txt) e tambem exatamente o mesmo das anotações iniciais dos resultados do dia 15.10.2024

# Detalhes técnicos de como está implantação funciona
Eu criei um documento que contém uma analise completa que descreve como essa minha implantação exatamente funciona, passo a passo, detalhe por detalhe, com o objetivo de facilitar a compreensão dos calculos e lógica envolvida. Para ver esse documento, clique no link abaixo:

[Ver ANOTACOES DO ALGORITMO](../../../../docs/ANOTACOES_ALGORITMO.md)

