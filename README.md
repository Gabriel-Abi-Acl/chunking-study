# chunking-study

Estudo técnico que compara estratégias de chunking para recuperação em RAG. O corpus, as perguntas e o harness ficam versionados. Este repositório ainda não tem números de uma execução com embedder real. O PR4 é quem publica recall, MRR, contenção do vão, tamanho do índice e custo de embedding.

Complementa o [rag-study](https://github.com/Gabriel-Abi-Acl/rag-study). Lá a pergunta é o efeito do contexto na resposta do modelo. Aqui a recuperação fica fixa e o que varia é o corte do documento. Geração de resposta fica fora, por enquanto.

O corpus deste PR é sintético (Cooperativa Rio Claro, fictícia) e serve só de placeholder. Não há chave de API no git.

## Objetivo

Medir, no mesmo conjunto de perguntas com vão de ouro, a diferença entre quatro cortes:

| estratégia | corte |
| --- | --- |
| `fixed` | janela fixa de caracteres ou tokens, sem overlap |
| `fixed_overlap` | a mesma janela, com overlap |
| `sentence_paragraph` | frase e parágrafo, empilhados até um tamanho |
| `semantic` | quebra onde a similaridade entre embeddings de frases vizinhas cai |

Para cada estratégia, no mesmo embedder e no mesmo `top_k`: recall@k, MRR, se o vão de ouro cabe inteiro num chunk recuperado, quantidade de chunks e volume embeddado.

## Hipótese

Hipótese a testar, ainda sem evidência neste repositório: o corte muda a chance de o vão de ouro caber inteiro num chunk recuperado, e também muda a quantidade de chunks e o volume embeddado.

Overlap deve recuperar um vão que atravessa a fronteira de uma janela fixa, desde que o vão caiba na janela. Não recupera um vão maior que a janela: o chunk continua limitado a `size`. Corte por frase ou parágrafo deve evitar partir um vão que cabe numa unidade. Corte semântico deve agrupar frases do mesmo assunto e separar a troca de assunto, com embeddings extras só para achar a quebra.

Confirmar ou refutar isso é o trabalho do PR4. O embedder de hash não entra nessa comparação.

## Metodologia planejada

1. Corpus sintético ou público, com perguntas e vãos de ouro (`start` inclusivo, `end` exclusivo, em caracteres do documento). O placeholder em `data/placeholder/` não é esse corpus.
2. As quatro estratégias acima, mesmo embedder, mesmo `top_k`.
3. Índice em memória, similaridade de cosseno. A classe `InMemoryVectorIndex` é o lugar de um índice persistente se o corpus deixar de caber em RAM.
4. Métricas abaixo. Não há geração de resposta neste estudo, por enquanto.
5. Tabela e gráfico só no PR4, com embedder real. Nenhum número de qualidade entra no README antes disso.

Decisões já fixadas no código:

- `fixed` e `fixed_overlap` são a mesma janela. O passo é `size - overlap`. A unidade é `chars` (índice de `str`, não byte) ou `tokens` (`\S+`). O offset gravado é sempre em caracteres, senão o vão não mapeia. Com overlap 0 a concatenação dos chunks reconstrói o documento.
- `sentence_paragraph` fecha unidade em `.`, `!`, `?` ou numa linha em branco. O ponto de abreviatura também fecha: não é um parser linguístico. Unidades são empilhadas até `size` caracteres. Overlap repete frases inteiras quando elas cabem no overlap e no tamanho; não corta frase para completar o overlap. Unidade maior que `size` cai na janela de caracteres.
- `semantic` embeda cada frase e quebra quando o cosseno com a frase anterior é estritamente menor que `SEMANTIC_SIMILARITY_THRESHOLD`. Não há janela de frases vizinhas em volta do embedding. O `size` ainda limita o chunk. `CHUNK_UNIT=tokens` não se aplica a estas duas estratégias.
- Contenção: o chunk cobre o vão quando `chunk.doc_id == span.doc_id`, `chunk.start <= span.start` e `chunk.end >= span.end`. Não há match aproximado de string.
- recall@k é a fração dos vãos da pergunta cobertos por algum chunk dos k primeiros. Sem vão, o valor é nulo, não zero.
- O recíproco da posição é `1/rank` do primeiro chunk recuperado que cobre algum vão, ou 0 se nenhum cobre. O MRR do run é a média desses valores. A média ignora nulo e conserva o zero (`mean_defined`).
- `span_contained` é verdadeiro só se cada vão da pergunta cabe inteiro em algum chunk recuperado. O runner usa o top-k, não o índice inteiro. A mesma função serve para perguntar se o corte sequer produziu um chunk que cubra o vão; o PR3 pode reportar as duas taxas.
- `embedded_texts` conta textos enviados ao embedder: frases da quebra semântica, chunks do índice e perguntas. O corte semântico embeda a frase para decidir a quebra e o índice embeda o chunk de novo. Não é preço em dólar e não há cache neste PR.
- O embedder `fake` é um saco de palavras com hash SHA-256, normalizado. Existe para teste e fumaça offline. Não é baseline de qualidade.

O SDK de Gemini ou OpenAI só é importado na primeira chamada de `embed`. Sem o pacote extra, a chamada falha com mensagem de instalação. A chave não entra no `repr` de `Settings` nem nas mensagens de erro.

## Como rodar

Requisito: Python 3.11+ e [uv](https://docs.astral.sh/uv/).

Na raiz do repositório, offline, sem chave:

```bash
uv sync
uv run python -m chunking_study
```

O comando lê `data/placeholder/`, recupera com o embedder fake e imprime um JSON por pergunta (`strategy`, `chunk_count`, `embedded_texts`, `retrieved_ids`, `recall_at_k`, `reciprocal_rank`, `span_contained`). Isso é fumaça do harness. Não é avaliação e o número impresso não é resultado do estudo.

Para sobrescrever o padrão, copie `.env.example` para `.env`. O `.env` não entra no git. Com `env` explícito, `load_settings` ignora o processo e o arquivo, para o teste não herdar chave.

Embedder real, ainda sem experimento gravado:

```bash
uv sync --extra gemini
```

`EMBEDDING_PROVIDER=gemini` e `GEMINI_API_KEY` ficam no ambiente ou no `.env`. OpenAI é o extra `openai`. A CI não instala esses extras e não chama rede.

Testes e lint, os mesmos da CI:

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

## Arquitetura

```
documentos → chunker → embedder → InMemoryVectorIndex
pergunta → embedder → top-k por cosseno → métricas
```

| módulo | papel |
| --- | --- |
| `chunking_study.chunkers` | `fixed`, `fixed_overlap`, `sentence_paragraph`, `semantic` |
| `chunking_study.embeddings` | hash, Gemini ou OpenAI; SDK só na primeira chamada |
| `chunking_study.index` | cosseno em memória |
| `chunking_study.metrics` | recall@k, recíproco da posição, contenção do vão |
| `chunking_study.runner` | um registro por pergunta |
| `chunking_study.corpus` | `.txt`/`.md` e JSONL com passagens literais |
| `chunking_study.config` | variáveis do `.env.example` |

## Roadmap

- **PR1 — esqueleto (este).** Interfaces, embedder fake, índice, métricas mínimas, placeholder e CI offline. Sem resultado.
- **PR2 — corpus e gabarito.** Corpus sintético ou público, em escala de estudo, e perguntas com vãos de ouro. Sem resposta de modelo.
- **PR3 — avaliação.** Agregar as métricas por estratégia (recall@k, MRR, contenção no top-k e no índice, chunks, volume embeddado) num formato reproduzível. Sem tabela preenchida por embedder real.
- **PR4 — experimentos.** Rodar o embedder real, gráficos e relatório no README. Só então entram números.

## O que ainda não está feito

Não há execução com embedder real no git, não há tabela preenchida e não há gráfico. O placeholder não é o corpus da comparação. O hash não mede qualidade de recuperação. Qualquer número de qualidade citado antes do PR4 está fora deste estudo.
