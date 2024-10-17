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
- Hash 1: 7e349ea735552563926e9fcbfae337b6d405abf916457af9bc6cce94e97e2c7e
- Hash 2: e727b358b88d8152b034da93750bdff29ae14e66a73b3f0649bf3abb5c029706

# Resultados esperados que toda execução deve produzir para estar intacta
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

# Esclarecimentos
Para obter esse historico de erro ao longo das epocas, eu rodei novamente o "script.html"[Veja ele aqui](./script.html) do teste de integridade 
Eu rodei novamente, e como de esperado, como não houve nenhuma alteração no algoritmo, o resultado permaneceu exatamente o mesmo descrito no arquivo [Compare aqui](./resultado-atualizado.txt) e tembem exatamente o mesmo das anotações iniciais dos resultados do dia 15.10.2024