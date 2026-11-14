# chunking-study

Estudo técnico que compara estratégias de chunking para recuperação em RAG. O corpus, as perguntas e o harness ficam versionados. Este repositório ainda não tem números de uma execução com embedder real. O PR4 é quem publica recall, MRR, contenção do vão, tamanho do índice e custo de embedding.

Complementa o [rag-study](https://github.com/Gabriel-Abi-Acl/rag-study). Lá a pergunta é o efeito do contexto na resposta do modelo. Aqui a recuperação fica fixa e o que varia é o corte do documento. Geração de resposta fica fora, por enquanto.

Há dois conjuntos no git, os dois sintéticos (Cooperativa Rio Claro, fictícia). `data/placeholder/` é a fumaça do harness. `data/eval/v1/` é o corpus da comparação. Não há chave de API no git. Não há número de recuperação neste README.

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

1. Corpus sintético, com perguntas e vãos de ouro (`start` inclusivo, `end` exclusivo, em caracteres do documento). Está em `data/eval/v1/`. O placeholder em `data/placeholder/` não é esse corpus.
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

O comando lê `data/placeholder/`, recupera com o embedder fake e imprime um JSON por pergunta (`strategy`, `chunk_count`, `embedded_texts`, `retrieved_ids`, `recall_at_k`, `reciprocal_rank`, `span_contained`). Isso é fumaça do harness. Não é avaliação e o número impresso não é resultado do estudo. O corpus da comparação é `data/eval/v1/` e não entra nesse comando.

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

## Conjunto de avaliação (v1)

O corpus da comparação fica em `data/eval/v1/`. São 12 documentos e 48 perguntas, em português. Ainda não foi medido: não há recall, MRR nem contenção publicada aqui.

Os textos foram escritos para este estudo. Não há fonte externa, log de usuário nem documento público. Cada passagem de ouro é um trecho literal. `start` é inclusivo e `end` é exclusivo, os dois como índice de `str` (caractere, não byte). O offset foi calculado em Python na gravação do JSONL. `load_eval_queries` recusa o arquivo se `texto[start:end]` divergir da passagem, se o vão for vazio ou se cair fora do documento. `load_eval_corpus` e `load_eval_queries` estão em `chunking_study.corpus`.

### Documentos

O tamanho varia de propósito. O mais curto tem 1204 caracteres (`biblioteca-faq`) e o mais longo tem 2696 (`estatuto`). A soma é 19795 caracteres. Há parágrafo longo, lista curta, passo numerado, pergunta e resposta, e tabela em texto. O ponto de `Passo 1.` também fecha frase no chunker deste repositório. O vão de ouro não inclui esse número.

| documento | caracteres | forma |
| --- | --- | --- |
| `acervo` | 2645 | parágrafos longos |
| `agua` | 1448 | parágrafos e lista curta |
| `assembleia` | 1482 | ata em parágrafos |
| `biblioteca-faq` | 1204 | perguntas e respostas |
| `cozinha` | 1305 | passos numerados |
| `estatuto` | 2696 | parágrafos longos |
| `faq-associacao` | 1281 | perguntas e respostas |
| `horta-turno` | 1298 | passos numerados |
| `mutirao` | 1271 | lista curta |
| `oficina-precos` | 1394 | tabela em texto |
| `plantio` | 2421 | parágrafos longos |
| `transporte` | 1350 | tabela em texto |

### Perguntas

Cada documento tem uma pergunta de cada categoria. O campo `category` do JSONL é conferido pelo teste.

| categoria | perguntas | vão |
| --- | --- | --- |
| `single_sentence` | 12 | uma frase, de 41 a 72 caracteres. Cinco perguntas têm dois vãos no mesmo documento |
| `two_sentences` | 12 | duas frases vizinhas no mesmo parágrafo, de 201 a 271 caracteres |
| `paragraph_boundary` | 12 | duas frases separadas por uma linha em branco, de 191 a 301 caracteres |
| `paraphrase` | 12 | uma frase, de 54 a 80 caracteres, com pouca sobreposição lexical |

São 53 vãos. Nas três primeiras categorias a pergunta reaproveita palavras da passagem. Na paráfrase a pergunta foi escrita com outras palavras, e a resposta continua sendo a frase literal.

A sobreposição lexical do teste conta tokens em minúsculas, com letra ou dígito, de comprimento pelo menos 4, fora de uma lista curta de funcionais (por exemplo "para", "quando", "qual", "antes"). A razão é a fração desses tokens da pergunta que também aparecem na passagem. O teste exige no máximo 0,25 na paráfrase e no mínimo 0,50 nas outras. Neste arquivo as doze paráfrases ficaram em 0 e as demais entre 0,75 e 1. Isso é critério de superfície, não um julgamento de modelo.

Um vão de uma frase cabe numa janela de 120 caracteres, o `CHUNK_SIZE` padrão do harness. Um vão de duas frases, ou um vão que atravessa a linha em branco, passa de 120 caracteres, então essa janela não o cobre inteiro. Isso descreve o gabarito. Não é resultado de recuperação.

### Limitações

Uma organização fictícia, um idioma, e só os gêneros da tabela. Não representa código, diálogo, PDF ou peça de um processo real. A paráfrase é lexical: uma pergunta pode ser difícil para um embedder e ainda assim passar deste critério, ou o contrário. Não há pergunta cuja resposta esteja em dois documentos. Não há resposta de modelo. Nenhum embedder real rodou sobre este conjunto.

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
| `chunking_study.corpus` | `.txt`/`.md`, JSONL do placeholder e JSONL do eval com offset conferido |
| `chunking_study.config` | variáveis do `.env.example` |

## Roadmap

- **PR1 — esqueleto.** Interfaces, embedder fake, índice, métricas mínimas, placeholder e CI offline. Sem resultado.
- **PR2 — corpus e gabarito (este).** `data/eval/v1/`, com 12 documentos e 48 perguntas. Sem resposta de modelo e sem métrica preenchida.
- **PR3 — avaliação.** Agregar as métricas por estratégia (recall@k, MRR, contenção no top-k e no índice, chunks, volume embeddado) num formato reproduzível. Sem tabela preenchida por embedder real.
- **PR4 — experimentos.** Rodar o embedder real, gráficos e relatório no README. Só então entram números.

## O que ainda não está feito

Não há execução com embedder real no git, não há tabela preenchida e não há gráfico. O placeholder não é o corpus da comparação. O corpus v1 também ainda não foi medido. O hash não mede qualidade de recuperação. Qualquer número de qualidade citado antes do PR4 está fora deste estudo.
