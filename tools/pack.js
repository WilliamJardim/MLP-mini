const fs   = require('fs');
const path = require('path');

// Diretório onde estão os arquivos .js que serão concatenados
const distDir = path.join(__dirname, '../dist');
// Arquivo de saída final
const outputFile = path.join(distDir, 'bundle.js');

// Função que une todos os arquivos .js em um só
function bundleJSFiles() {
  // Verifica se o diretório dist existe
  if (!fs.existsSync(distDir)) {
    console.error('Diretório dist não encontrado!');
    return;
  }

  // Se o arquivo bundle.js já existe, remove-o para evitar duplicidade de conteúdo
  if (fs.existsSync(outputFile)) {
    console.log(`Removendo o arquivo existente: ${outputFile}`);
    fs.unlinkSync(outputFile);
  }

  // Lê o conteúdo da pasta dist
  const files = fs.readdirSync(distDir);

  // Filtra apenas arquivos .js
  const jsFiles = files.filter(file => path.extname(file) === '.js');

  if (jsFiles.length === 0) {
    console.error('Nenhum arquivo .js encontrado no diretório dist.');
    return;
  }

  // Variável para armazenar o conteúdo combinado
  let combinedContent = '';

  // Itera sobre cada arquivo .js, lê o conteúdo e concatena
  jsFiles.forEach(file => {
    const filePath = path.join(distDir, file);
    const fileContent = fs.readFileSync(filePath, 'utf8');
    combinedContent += `\n// Conteúdo do arquivo: ${file}\n${fileContent}\n`;
  });

  // Escreve o conteúdo concatenado no bundle.js
  fs.writeFileSync(outputFile, combinedContent, 'utf8');
  console.log(`Arquivos combinados em ${outputFile}`);
}

// Chama a função para combinar os arquivos
bundleJSFiles();