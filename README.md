# Entregas 1 e 2 — GoBR

## 1. Descrições comerciais

`dados/entradas/v1/regra_001.txt` até `regra_246.txt` são as 246 descrições
comerciais usadas nesta entrega. Cada uma corresponde ao arquivo de mesmo nome
em `dados/referencias/`. A cobertura foi conferida: 246 descrições, 246 fontes
GoBR, nenhum arquivo vazio ou sem par.

O notebook [gerar_explicacoes_gobr.ipynb](gerar_explicacoes_gobr.ipynb) registra
o fluxo de geração com Qwen e manual. Seu resultado anterior em
`dados/entradas/v2/explicacoes.jsonl` contém apenas 233 rascunhos, sem revisão
humana. Ele **não** é a base da entrega 2: algumas explicações são genéricas e
há divergências com o código. Por exemplo, em `regra_153`, o rascunho fala em
R$ 200 para BK, enquanto a condição GoBR e a descrição de `v1` usam R$ 80.

## 2. Indexação das descrições

Abra [indexacao_explicacoes.ipynb](indexacao_explicacoes.ipynb) no Colab e
execute as células em ordem. Ele baixa automaticamente o repositório público
do GitHub quando os dados não estão disponíveis na sessão. O notebook lê as
246 descrições de `v1`, confere os pares com os `.gobr`, cria embeddings com `BAAI/bge-m3` (FP16 e lotes de 4 na T4, limite de tokens ajustado ao maior texto),
persiste o índice e mostra buscas e três casos de avaliação. O texto vetorizado
é a descrição comercial. O código GoBR fica ligado pelo ID e é devolvido junto
com a descrição na busca.

Antes de baixar o modelo, execute a célula de diagnóstico do notebook.
Resultado esperado: `246` descrições e `0` fontes sem explicação. O índice
vetorial é criado no Colab na primeira execução e reutilizado nas seguintes.
Os resultados precisam ser inspecionados: similaridade não prova que uma regra
está correta ou completa para uma nova solicitação.

O manual `dados/govendas-rules.pdf` é utilizado no fluxo de geração. Sua
indexação pertence à terceira entrega e ainda não faz parte deste índice.
