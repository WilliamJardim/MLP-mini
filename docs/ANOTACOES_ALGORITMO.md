# EXPLICAÇÂO DO FUNCIONAMENTO DO BACKPROPAGATION DESTA IMPLANTAÇÂO

**DATA: 21/12/2024 10:12 AM**

**EXPLICAÇÂO DETALHADA DE COMO O ALGORITMO DO BACKPROPAGATION FUNCIONA COM BASE NESSA IMPLANTAÇÂO.**

**PROBLEMA ESCOLHIDO PARA ESSA ANALISE: XOR, DO TESTE DE INTEGRIDADE, DEVIDO A SIMPLICIDADE DA ESTRUTURA USADA:** 

[Ver Problema Escolhido](../tests/classificacao/XOR/XOR_INTEGRY_TEST_15_10_verifed/detalhes_teste_integridade.md)

# Explicação do Backpropagation na Implementação XOR

Nesta análise, explicaremos como o algoritmo de backpropagation é implementado para resolver o problema XOR. Este guia detalhado descreve cada etapa do cálculo dos gradientes, desde a camada de saída até as camadas ocultas, destacando como as estruturas de dados `this.layers` e `this.weights` são usadas.

# Estrutura da Rede
- `this.layers`: Define a quantidade de unidades em cada camada, do início ao fim.
- `this.weights`: Contém os pesos da rede, organizados em uma matriz tridimensional:
- `this.weights[l][j][k]`: representa o peso entre a unidade `j` da camada `l` e a unidade `k` da camada seguinte.

# Inicio do texto explicativo

O `this.layers` contém quantas unidades existem em cada camada,
em ordem do primeiro até o ultimo mesmo.

bem como o `this.weights`, que é uma matriz , ou seja, um vetor que contém outros vetores. 

A primeira dimensão dessa matriz é a camada. Ou seja, cada linha dentro dela[isso é, da matriz] é um vetor de pesos que corresponde aos pesos de uma unidade dessa camada.
Por exemplo `this.weights[0]` retornará todos os vetores de pesos da camada oculta cujo indice é 0, OU SEJA, LOGO `this.weights[0][0]` vai retornar o vetor de pesos da unidade cujo índice é 0 da camada oculta cujo índice é 0

Outro exemplo `this.weights[1]` retornará todos os vetores de pesos da camada de saida, cujo indice é 1, OU SEJA, LOGO `this.weights[1][0]` vai retornar o vetor de pesos da unidade cujo índice é 0 da camada oculta cujo índice é 1

A sintaxe para acessar os vetores de peso é sempre assim: `this.weights[ INDICE_CAMADA ][ INDICE_UNIDADAE ]`
E por extensão, a sintaxe para acessar um peso especifico é assim: `this.weights[ INDICE_CAMADA ][ INDICE_UNIDADAE ][ INDICE_PESO ]`

Como você pode ver, a matrix `this.weights` está organizado em ordem da primeira camada até a ultima camada tambem.


## NOTAS SOBRE ESSA MATRIX this.weights:
  - Ela no problema do XOR tem tamanho(propriedade length) de 2, pois, ela tem 3 elementos: 
     - (index 0) NUMERO_ENTRADAS, 
     - (index 1) NUMERO_UNIDADES_CAMADA_OCULTA, 
     - (index 2) NUMERO_UNIDADES_SAIDA
  
  - Ou seja:
     - O indice 0 é a camada de entrada.
     - O indice 1 é a quantidade de unidades da camada ocultas
     - E o ultimo indice, o indice 2, é a quantidade de saidas da rede(ou seja a quantidade de unidades na camada de saida)


Os erros da camada de saída são calculados pela subtração entre os valores estimados e os valores desejados, das unidades da camada de saída.

Existe uma variável criada chamada `layerErrors`, que nesse ponto(após ter sido calculado esses erros da camada de saida), começa como uma matriz que contém esses erros `const layerErrors = [outputError];`

E a variavel `outputError` é um vetor de erros(isso é, os erros que foram calculados da camada de saída, conforme descrito acima)


# Primeiro laço FOR 

No primeiro laço FOR, Quando vai iterar sobre as camadas no backpropagation NO TRECHO: `for (let l = this.weights.length - 1; l >= 1; l--) {`
Nessa iteração a variavel `l` começa com o valor de `this.weights.length - 1` por que o código ignora a camada de saída, pois já calculamos os gradientes da camada de saída

## NOTAS SOBRE A CONDIÇÂO DESSE FOR: 
   - E também a condição de parada do loop é enquanto `l >= 1`(enquanto `l` for maior ou igual que 1) por que eu quero que ele ignore a camada de entrada. Ou seja, ele só vai fazer até a primeira camada oculta, mais ai no indice 0, ele para, pois o indice zero é a camada de entrada, e ela precisa ser ignorada nesse processo.

   - NOTA SOBRE ISSO: Se eu nao ignorasse a camada entrada coisas estranhas iriam acontecer: O código tentaria calcular os gradientes das entradas, sendo que as entradas não tem função de ativação, não tem derivada, ou seja, elas não são unidades e não tem pesos lá, são apenas números!. Então, isso causaria erros. Por isso ignoro a camada de entrada, por que faz todo sentido ignorar dessa forma.

   - Além disso, um outro fato é que, como esse problema do XOR tem apenas uma camada oculta e a camada de saida,
   então, ele só irá calcular os gradientes dessa camada oculta. Ou seja, no laço for que itera sobre as camadas, ele já começa com o valor de `l = this.weights.length - 1`, que nesse caso do problema do XOR, vai ter valor de 1. E como ele faz o loop ENQUANTO `l >= 1`, ele vai fazer apenas dessa camada oculta, conforme explicado. E quando ele subtraisse 1 da variavel `l` para ir para a proxima iteração, a condição `l >= 1` retornaria false, e ele encerraria o loop. 

## OUTRAS NOTAS: 
   - O valor de `l` nessa iteração começa sendo 1, ou seja, ao acessar `this.weights[l - 1]`, estamos pegando a matrix dos pesos da camada oculta atual

   - Por outro lado Se fosse `this.weights[l]` eu estaria acessando a matrix dos pesos da camada de saída(output)

   - Isso faz todo sentido pois `l` na primeira iteração vai ter o valor `1`, que aponta para a camada de saída, e `l - 1`, ou seja `0`, aponta para a camada oculta(que está atráz da camada `1`)

   - SUB-NOTA: Como esse modelo só tem uma camada oculta, isso fica muito fácil de entender.


**DENTRO DESSE PRIMEIRO FOR:**

Para cada iteração de `l`:

   Ele cria uma variável `const layerError = [];` pra armazenar os gradientes da camada `l` atual

   **LOGO ABAIXO É O MOMENTO EM QUE O SEGUNDO FOR È CRIADO:**

   em seguida ele roda um outro laço FOR: ` for (let j = 0; j < this.weights[l - 1].length; j++) {`

   Esse outro laço FOR percorre cada "vetor de pesos" da matriz da camada oculta atual `this.weights[l - 1]`

   O `j` é o índice do "vetor de pesos"(que contém os pesos da unidade), então na realidade, a cada iteração de `j`, ele vai acessar uma unidade(isso é um vetor de pesos que corresponde aos pesos da unidade) cujo índice é `j` da camada oculta atual( `this.weights[l - 1]` )
   

# Segundo laço FOR, alinhado(dentro do primeiro)   

**DENTRO DESSE SEGUNDO FOR:**

  Na primeira iteração a variável `j` começa sendo `0`, o que significa que estamos acessando a primeira unidade(ou melhor dizendo, o primeiro vetor de pesos) da matrix `this.weights[l - 1]`, a saber, esse índice `j=0` aponta para a seguinte posição: `this.weights[l - 1][0]`, que é justamente o vetor de pesos da unidade `j=0` que contém dois pesos: `[0.8228850033675079, -0.314907800152612]`.

  ele cria uma variável chamada `let error = 0;`, que armazena o gradiente da unidade `j` atual, que pertence a camada oculta atual( `this.weights[l - 1]` )

  LOGO ABAIXO, ELE VAI RODAR AINDA OUTRO TERCEIRO LOOP FOR: `for (let k = 0; k < this.weights[l].length; k++) {`
  

# Terceiro laço FOR, alinhado(dentro do segundo)  

**DENTRO DESSE TERCEIRO LOOP FOR:**

   Esse outro loop for, que itera sobre o índice `k`, ele corresponde aos vetores de pesos da camada seguinte(isso é, da camada oculta seguinte à camada oculta atual)

   Na primeira iteração esse valor `k` começa sendo `0`, o que significa que estamos na primeira unidade da camada seguinte(isso é, o primeiro vetor de pesos da camada seguinte) da matriz `this.weights[l]`, ou seja, estamos na posição `this.weights[l][0]`, que é justamente o seguinte vetor de pesos: `[0.21803494362838416, 0.13302177857890918]`

   Esse FOR vai somando a variável `error`, acumulando assim os gradientes, da seguinte forma: "multiplicando o gradiente da unidade `k` PELO peso cujo índice é `j` da unidade `k`", 
   NOTA: lembrando que a unidade `k` está na camada seguinte. 

   IMPORTANTE: Na linha 278, no primeiro termo, isso é `layerErrors[0]`, nós não estamos acessando os gradientes da primeira camada da rede, PELO CONTRARIO, ao usar `layerErrors[0]`, estamos na realidade acessando os gradientes DA ULTIMA CAMADA OCULTA CALCULADA, OU SEJA, DA CAMADA SEGUINTE 

   IMPORTANTE: Da mesma forma, Nessa mesma linha 278, no segundo termo, isso é `this.weights[l][k][j]`, nós estamos acessando os gradientes das unidades da camada seguinte(pois estamos usando o índice `l` que corresponde a camada seguinte), ... desse modo, nesse primeiro trecho linha ao acessar `this.weights[l][k]`, estamos acessando os gradientes da UNIDADE `k` da camada seguinte. Ou seja,... o segunto termo isso é `this.weights[l][k][j]`, que pega o peso de conexão, ou seja, o peso da unidade `k`(do terceiro for) da camada seguinte `l`(do primeiro for) cujo índice é "`j`(da iteração do segundo for)" 

  Ai logo em seguida, quando termina todas as iterações desse terceiro for, com a variável `error` já calculada, ele obtem qual é a função de ativação usada pela unidade(nesse caso do problema do XOR é a Sigmoid para todas!)
  
  E LOGO EM SEGUIDA, ele faz `layerError.push(error * ActivationFunctions[`${nomeDaFuncao}Derivative`](this.layerActivations[l][j]));`, para adicionar na lista `layerError` o erro da unidade `j` da camada o oculta atual MULTIPLICADO pela derivada da função de ativação da unidade `j` da camada o oculta atual


# Ao final de cada iteração do segundo laço FOR

**[APOS TERMINAR TODAS AS ITERAÇÔES DO SEGUNDO LOOP QUE ITERA SOBRE J]**
Nesse ponto do código, Ele faz `layerErrors.unshift(layerError);`, isso é, ele joga o vetor de gradientes da camada oculta atual `l` para o INICIO DO VETOR usando o método unshift que é o inverso do push. 

Desse modo, da próxima vez que formos acessar o `layerErrors[0]` dentro do TERCEIRO LOOP, se estivéssemos calculando os gradientes de uma outra camada oculta anterior a essa, estaríamos na verdade acessando os gradientes da ULTIMA CAMADA OCULTA CALCULADA, OU SEJA, ESSES GRADIENTES QUE ESTAMOS ADICIONANDO COM O UNSHIFT






    


  


   

   

   



