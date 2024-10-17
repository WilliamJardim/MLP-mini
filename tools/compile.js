/**
* Uma rotina automatizada para compilar, empacotar, e testar o empacotamento 
*/
const { exec } = require('child_process');
const path = require('path');

// Caminho para o arquivo compile.bat
const batFilePath = path.join(__dirname, '../repository-scripts', 'compile.bat');

// Função para rodar o arquivo compile.bat
function runBatchFile() {
  exec(`"${batFilePath}"`, (error, stdout, stderr) => {
    if (error) {
      console.error(`Erro ao executar compile.bat: ${error.message}`);
      return;
    }

    if (stderr) {
      console.error(`Erro no script compile.bat: ${stderr}`);
      return;
    }

    console.log(`Saída do compile.bat:\n${stdout}`);
    console.log('compile.bat executado com sucesso!');
  });
}

// Chama a função
runBatchFile();

// Gerar o pack
require('./pack.js');

// Gerar o pack
require('./checkpack.js');