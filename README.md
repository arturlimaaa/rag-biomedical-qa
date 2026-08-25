# Retrieval-Augmented Generation for Biomedical Question Answering

A RAG pipeline over the [BioASQ](https://huggingface.co/datasets/rag-datasets/rag-mini-bioasq/)
biomedical corpus, built up from a plain FAISS baseline to a hybrid retrieval pipeline with
cross-encoder reranking, contextual compression and an adaptive retry policy.

Biomedical QA is a hard setting for retrieval: questions are multi-part, answers are
information-dense, and a confident wrong answer costs more than an abstention. The notebook
also shows why the usual string-overlap metrics do not work here.

## Pipeline

| Stage | Implementation |
| --- | --- |
| Retrieval | Dense biomedical SBERT embeddings in FAISS, combined with BM25 |
| Reranking | Cross-encoder scoring of every (query, snippet) pair, top-k kept |
| Compression | Sentence-level filtering against the query by cosine similarity, bounded by threshold τ and `max_sents` |
| Generation | `google/gemma-3-1b-it`, constrained to the retrieved context, with an explicit "I don't know" option |
| Retry | On refusal, re-run with a wider candidate pool and looser compression |

## Results

BERTScore rates the no-retrieval baseline at P 0.811 / R 0.845 / F1 0.827 while every one of
its answers is factually wrong. That gap is the reason the notebook adds an LLM-as-a-judge
pass weighted so that abstention is penalised less than a confident error.

Against that judge the full pipeline beats the baseline: it answers binary and mechanism
questions correctly and turns one baseline hallucination into a correct answer. It still
drifts on formatting in one case and misses a rare term in another.

## Running it

Needs a GPU; developed on a Colab L4. Generation uses a gated model, so accept the
[Gemma-3-1B licence](https://huggingface.co/google/gemma-3-1b-it) and supply a Hugging Face
read token as `HF_TOKEN`. All other dependencies install from the first cell.
