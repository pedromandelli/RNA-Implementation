# Guia de Git para o Projeto RNA

Este guia contém instruções básicas para o uso de Git no projeto de implementação da Rede Neural Artificial.

## Comandos Básicos

### Verificar Status do Repositório
```bash
git status
```

### Adicionar Alterações ao Stage
```bash
# Adicionar um arquivo específico
git add nome_do_arquivo.py

# Adicionar todos os arquivos modificados
git add .
```

### Criar um Commit
```bash
git commit -m "Mensagem descritiva das alterações"
```

### Enviar Alterações para o Repositório Remoto
```bash
git push
```

### Atualizar o Repositório Local
```bash
git pull
```

## Boas Práticas para Mensagens de Commit

1. **Seja descritivo e conciso**:
   ```
   # Bom
   git commit -m "Implementada função de ativação Sigmoid"
   
   # Ruim
   git commit -m "Atualização"
   ```

2. **Use o modo imperativo**:
   ```
   # Bom
   git commit -m "Adiciona testes para backpropagation"
   
   # Ruim
   git commit -m "Adicionando testes para backpropagation"
   ```

3. **Mencione o componente afetado**:
   ```
   git commit -m "network.py: Implementa método fit com gradiente descendente"
   ```

## Trabalhando com Branches

### Criar uma Nova Branch
```bash
git checkout -b nome-da-branch
```

### Trocar de Branch
```bash
git checkout nome-da-branch
```

### Listar Branches
```bash
git branch
```

### Mesclar Branch para a Main
```bash
# Primeiro, vá para a branch main
git checkout main

# Depois, faça o merge
git merge nome-da-branch
```

## Fluxo de Trabalho Sugerido

1. Antes de começar a trabalhar, atualize seu repositório:
   ```bash
   git pull
   ```

2. Crie uma branch para a funcionalidade ou correção:
   ```bash
   git checkout -b implementa-sigmoid
   ```

3. Faça as alterações necessárias e teste-as.

4. Adicione e faça commit das alterações:
   ```bash
   git add .
   git commit -m "Implementa função de ativação Sigmoid"
   ```

5. Envie a branch para o repositório remoto:
   ```bash
   git push -u origin implementa-sigmoid
   ```

6. Quando a funcionalidade estiver pronta, mescle com a main:
   ```bash
   git checkout main
   git merge implementa-sigmoid
   git push
   ```

## Lidando com Conflitos

Se ocorrerem conflitos durante um merge, você precisará resolvê-los manualmente:

1. Os arquivos com conflitos serão marcados no `git status`.
2. Abra esses arquivos e procure por marcações como `<<<<<<< HEAD`, `=======` e `>>>>>>> branch-name`.
3. Edite o arquivo para resolver os conflitos.
4. Adicione os arquivos resolvidos com `git add`.
5. Complete o merge com `git commit`.

## Configurando seu Nome e Email

Se você ainda não configurou seu nome e email no Git:

```bash
git config --global user.name "Seu Nome"
git config --global user.email "seu.email@exemplo.com"
``` 