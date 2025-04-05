#!/bin/bash

# Tabela de Re -> U
declare -A velocities=(
  [0.01]=1.50E-07
  [0.05]=7.50E-07
  [0.10]=1.50E-06
  [0.40]=6.00E-06
  [0.60]=9.00E-06
  [0.80]=1.20E-05
  [1.00]=1.50E-05
  [1.50]=2.25E-05
  [2.00]=3.00E-05
  [3.00]=4.50E-05
  [4.00]=6.00E-05
  [5.00]=7.50E-05
)

# Etapa 1: criar diretórios e atualizar campo U
for Re in "${!velocities[@]}"; do
  dir="../cases/Re_$Re"
  echo "Criando caso $dir com U = ${velocities[$Re]}"
  cp -r ../template "$dir"
  sed -i "s|uniform (.* 0 0);|uniform (${velocities[$Re]} 0 0);|" "$dir/0/U"
done

# Etapa 2: gerar lista de casos
echo "${!velocities[@]}" | tr ' ' '\n' > casos.txt

# Etapa 3: função para rodar 1 caso
run_case() {
  Re=$1
  dir="../cases/Re_$Re"
  echo "Executando $dir"
  (cd "$dir" && simpleFoam > log.simpleFoam 2>&1)
}

export -f run_case  # exportar função para xargs

# Etapa 4: executar até 12 casos em paralelo
cat casos.txt | xargs -n 1 -P 12 -I {} bash -c 'run_case "$@"' _ {}