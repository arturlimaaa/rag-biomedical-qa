import ast
import pathlib
import re

import pytest

from conftest import calls_named, kwargs_of, load, notebook_source

class Doc:
    """Stands in for langchain.schema.Document, which is not installed here."""

    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


PURE = load(
    [
        "_SENT_SPLIT", "sent_tokenize_quick", "make_snippets",
        "hybrid_candidates", "rerank_snippets", "compress",
        "token_truncate", "clean_answer_echo", "is_refusal", "_REFUSAL_MARKERS",
        "DISCARD_LINES_PAT", "clean_output",
    ],
    env={"re": re, "List": list, "Any": object, "Document": Doc},
)


class FakeBM25:
    """Mirrors BM25Retriever: returns at most `k` documents, k defaulting to 4."""

    def __init__(self, docs, k=4):
        self.docs = docs
        self.k = k

    def get_relevant_documents(self, query):
        return self.docs[: self.k]

    def invoke(self, query):
        return self.docs[: self.k]


class FakeDense:
    def __init__(self, docs):
        self.docs = docs

    def get_relevant_documents(self, query):
        return self.docs

    def invoke(self, query):
        return self.docs


# --- retrieval -------------------------------------------------------------

def test_hybrid_candidates_honours_the_sparse_budget():
    dense = FakeDense([Doc(f"dense passage {i}", {"doc_id": f"d{i}"}) for i in range(10)])
    sparse = FakeBM25([Doc(f"sparse passage {i}", {"doc_id": f"s{i}"}) for i in range(120)])

    merged = PURE["hybrid_candidates"](
        "a biomedical question", dense, sparse, bm25_fetch_k=120
    )

    from_sparse = [d for d in merged if d.metadata["doc_id"].startswith("s")]
    assert len(from_sparse) == 120


def test_hybrid_candidates_deduplicates_shared_documents():
    shared = Doc("identical passage text", {"doc_id": "x1"})
    merged = PURE["hybrid_candidates"](
        "q", FakeDense([shared]), FakeBM25([shared], k=1), bm25_fetch_k=10
    )
    assert len(merged) == 1


def test_make_snippets_preserves_every_sentence():
    text = " ".join(f"Sentence number {i}." for i in range(10))
    snippets = PURE["make_snippets"]([Doc(text, {"doc_id": 7})], max_sents_per_snip=3)

    assert [s.metadata["doc_id"] for s in snippets] == [7] * len(snippets)
    rejoined = " ".join(s.page_content for s in snippets)
    assert rejoined == text


# --- compression -----------------------------------------------------------

class FakeEncoder:
    """Scores a sentence by how many query words it contains, as a unit vector."""

    def encode(self, text, normalize_embeddings=True):
        import numpy as np

        def vec(s):
            v = np.zeros(26, dtype="float32")
            for ch in re.findall(r"[a-z]", s.lower()):
                v[ord(ch) - 97] += 1.0
            n = np.linalg.norm(v)
            return v / n if n else v

        if isinstance(text, str):
            return vec(text)
        return np.stack([vec(t) for t in text])


def _compress(**kw):
    ns = load(
        ["compress", "sent_tokenize_quick", "_SENT_SPLIT"],
        env={"re": re, "List": list, "Document": Doc, "_comp_enc": FakeEncoder()},
    )
    return ns["compress"](**kw)


def test_compress_caps_the_number_of_sentences():
    docs = [Doc(" ".join(f"aaa bbb sentence {i}." for i in range(20)), {"doc_id": 1})]
    sents, ids, insufficient = _compress(q="aaa bbb", docs_kept=docs, max_sents=4, tau=-1.0)

    assert not insufficient
    assert len(sents) == 4
    assert len(ids) == 4


def test_compress_reports_insufficient_when_nothing_clears_tau():
    docs = [Doc("Totally unrelated wording here.", {"doc_id": 1})]
    sents, ids, insufficient = _compress(q="aaa bbb", docs_kept=docs, max_sents=8, tau=0.999)

    assert (sents, ids, insufficient) == ([], [], True)


def test_compress_drops_duplicate_sentences():
    """Overlapping snippets repeat sentences; the context budget should not pay twice."""
    repeated = "Inclisiran inhibits PCSK9."
    docs = [
        Doc(f"{repeated} Filler one.", {"doc_id": 1}),
        Doc(f"{repeated} Filler two.", {"doc_id": 2}),
    ]
    sents, ids, insufficient = _compress(q="Inclisiran PCSK9", docs_kept=docs, max_sents=8, tau=-1.0)

    assert sents.count(repeated) == 1


# --- answer handling -------------------------------------------------------

def test_token_truncate_respects_the_budget():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    text = " ".join(["biomedical"] * 500)

    out = PURE["token_truncate"](text, tok, 50)

    assert len(tok.encode(out, add_special_tokens=False)) <= 50


def test_clean_answer_echo_strips_the_prompt_scaffold():
    q = "What does inclisiran inhibit?"
    raw = f"Question: {q}\n\nContext:\nSome retrieved text.\n\nAnswer: It inhibits PCSK9."

    assert PURE["clean_answer_echo"](raw, q) == "It inhibits PCSK9."


@pytest.mark.parametrize(
    "text",
    [
        "I don't know.",
        "I do not know.",
        "I don't know the answer to that.",
        "The context does not contain the answer.",
        "There is insufficient evidence in the provided context.",
        "I cannot answer this question.",
        "Unable to answer based on the given context.",
        "",
    ],
)
def test_is_refusal_recognises_common_refusals(text):
    assert PURE["is_refusal"](text) is True


@pytest.mark.parametrize(
    "text",
    [
        "Inclisiran is an siRNA drug that inhibits PCSK9.",
        "The known risk factors are age and smoking.",
        "Yes, the association is well established.",
    ],
)
def test_is_refusal_does_not_fire_on_real_answers(text):
    assert PURE["is_refusal"](text) is False


def test_clean_output_drops_disclaimers_and_echoed_question():
    q = "What is PCSK9?"
    raw = f"{q}\nDisclaimer: This is not medical advice.\n- PCSK9 is a protease."

    assert PURE["clean_output"](raw, q) == "PCSK9 is a protease."


# --- notebook-level invariants --------------------------------------------

def test_bertscore_goes_through_one_helper():
    """Every reported score must come from the same code path, or the numbers the prose
    compares sit on different scales."""
    src = notebook_source()
    assert "def bertscore_report" in src

    helper_body = src.split("def bertscore_report", 1)[1].split("\nbaseline_scores", 1)[0]
    raw_calls = [c for c in calls_named("score") if "lang" in kwargs_of(c)]
    assert len(raw_calls) == 2, "score() should only be called inside the helper"
    assert helper_body.count("score(") >= 2

    reports = [c for c in calls_named("bertscore_report")]
    assert len(reports) >= 2, "both the baseline and the RAG run must use the helper"


def test_bertscore_helper_reports_both_scales():
    """Raw BERTScore flatters wrong answers; the rescaled figure is what discriminates."""
    src = notebook_source()
    helper = src.split("def bertscore_report", 1)[1].split("\nbaseline_scores", 1)[0]
    assert "rescale_with_baseline=True" in helper
    assert '"raw"' in helper and '"rescaled"' in helper


def test_greedy_decoding_does_not_pass_a_temperature():
    """temperature is meaningless with do_sample=False and raises on newer transformers."""
    offenders = []
    for call in calls_named("pipeline"):
        kw = kwargs_of(call)
        if kw.get("do_sample") == "False" and "temperature" in kw:
            offenders.append(kw)
    for call in calls_named("generate"):
        kw = kwargs_of(call)
        if kw.get("do_sample") == "False" and "temperature" in kw:
            offenders.append(kw)
    assert offenders == []


def test_cuda_is_only_named_once_behind_a_capability_check():
    """One detection site is fine; every other use must go through DEVICE."""
    src = notebook_source()
    tree = ast.parse(src)
    literals = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value == "cuda"
    ]
    assert len(literals) == 1, "cuda is hardcoded somewhere other than the DEVICE check"
    assert 'DEVICE = "cuda" if torch.cuda.is_available() else "cpu"' in src


def test_cuda_cache_clearing_is_guarded():
    src = notebook_source()
    for line_no, line in enumerate(src.split("\n")):
        if "torch.cuda.empty_cache()" in line:
            window = "\n".join(src.split("\n")[max(0, line_no - 3):line_no])
            assert "is_available()" in window, "empty_cache() called without a CUDA check"


def test_rank_bm25_is_installed_before_bm25_is_constructed():
    nb_text = _raw_notebook_text().lower()
    install_at = nb_text.find("rank_bm25")
    use_at = nb_text.find("bm25retriever.from_documents")
    assert install_at != -1, "rank_bm25 is never installed"
    assert install_at < use_at, "BM25Retriever is constructed before rank_bm25 is installed"


def test_no_leftover_coursework_notes():
    text = _raw_notebook_text()
    for marker in ["554", "maybe make them", "TODO", "HINT", "pts)"]:
        assert marker not in text, f"leftover scaffolding in notebook: {marker!r}"


def test_no_deprecated_retriever_api():
    assert "get_relevant_documents" not in notebook_source()


def _raw_notebook_text():
    import json
    import pathlib

    from conftest import NOTEBOOK

    nb = json.loads(pathlib.Path(NOTEBOOK).read_text())
    return "\n".join(
        "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
    )


# --- dependency and environment invariants --------------------------------

def test_no_langchain_v1_removed_imports():
    """langchain 1.x dropped langchain.schema and langchain.text_splitter."""
    src = notebook_source()
    assert "from langchain.schema import" not in src
    assert "from langchain.text_splitter import" not in src


def test_install_cell_pins_langchain():
    """Unpinned, a fresh runtime resolves langchain 1.x and the notebook dies on import."""
    import json
    from conftest import NOTEBOOK

    nb = json.loads(pathlib.Path(NOTEBOOK).read_text())
    install = next(
        "".join(c["source"]) for c in nb["cells"]
        if c["cell_type"] == "code" and "pip install" in "".join(c["source"])
    )
    assert re.search(r'"langchain[>=<][^"]*"', install), install


def test_gated_model_is_authenticated_before_it_is_loaded():
    """google/gemma-3-1b-it is licence-gated; an anonymous load returns 401."""
    text = _raw_notebook_text()
    login_at = text.find("login(")
    load_at = text.find("AutoModelForCausalLM.from_pretrained")
    assert login_at != -1, "the notebook never authenticates with Hugging Face"
    assert login_at < load_at, "the gated model is loaded before login"


def test_baseline_dense_index_normalizes_embeddings():
    """multi-qa-mpnet-base-dot-v1 is a dot-product model; FAISS defaults to L2."""
    src = notebook_source()
    block = src.split("multi-qa-mpnet-base-dot-v1", 1)[1].split("FAISS.from_documents", 1)[0]
    assert "normalize_embeddings" in block


def test_bm25_is_constructed_with_an_explicit_k():
    """BM25Retriever.k defaults to 4, which silently caps the whole sparse arm."""
    calls = calls_named("from_documents")
    bm25 = [
        c for c in calls
        if isinstance(c.func, ast.Attribute)
        and isinstance(c.func.value, ast.Name)
        and c.func.value.id == "BM25Retriever"
    ]
    assert bm25, "BM25Retriever.from_documents is never called"
    assert all("k" in kwargs_of(c) for c in bm25)


# --- adaptive retry --------------------------------------------------------

def test_retry_fires_when_compression_finds_no_evidence():
    """Empty compression is the strongest signal that the evidence net is too narrow."""
    calls = {"n": 0}

    def fake_compress(q, docs_kept, max_sents, tau):
        calls["n"] += 1
        if calls["n"] == 1:
            return [], [], True                      # strict pass finds nothing
        return ["supporting sentence"], ["d1"], False  # relaxed pass finds evidence

    ns = load(
        ["qa_pipeline"],
        env={
            "List": list, "Any": object, "CrossEncoder": object,
            "CFG": {"bm25_k": 10, "idk_str": "I don't know."},
            "hybrid_candidates": lambda q, d, b, k: [Doc("candidate", {"doc_id": "d1"})],
            "rerank_snippets": lambda q, docs, rr, k: docs,
            "compress": fake_compress,
            "answer_with_context": lambda q, sents, ids, max_ctx_tokens: ("PCSK9.", ids),
            "is_refusal": PURE["is_refusal"],
        },
    )

    preds, ids = ns["qa_pipeline"](
        questions=["what does inclisiran inhibit?"],
        dense_retriever=None, bm25=None, reranker=None,
        k_final=8, k_retry=12, tau=0.5, tau_retry=0.47,
        max_sents=8, max_sents_retry=10, max_ctx_tokens=1400,
    )

    assert preds == ["PCSK9."]


def test_load_llm_uses_its_argument():
    """The parameter was shadowed by a module global, so callers could not switch models."""
    src = notebook_source()
    body = src.split("def load_llm", 1)[1].split("\nllm = ", 1)[0]
    assert "MODEL_ID" not in body, "load_llm ignores model_id and reads the global"
    assert "model_id" in body


def test_the_generation_model_is_only_loaded_once():
    """A second full copy of gemma-3-1b-it doubles VRAM and OOMs a 16GB card."""
    src = notebook_source()
    assert src.count("AutoModelForCausalLM.from_pretrained") == 1


def test_bm25_preprocessing_lowercases_and_strips_punctuation():
    ns = load(["bm25_preprocess"], env={"re": re})
    assert ns["bm25_preprocess"]("Inclisiran inhibits PCSK9, strongly.") == [
        "inclisiran", "inhibits", "pcsk9", "strongly",
    ]


def test_compress_groups_sentences_by_source_document():
    """A similarity-ordered shuffle makes the context incoherent to read."""
    docs = [
        Doc("aaa one. zzz filler. aaa two.", {"doc_id": "A"}),
        Doc("aaa three. aaa four.", {"doc_id": "B"}),
    ]
    sents, ids, _ = _compress(q="aaa", docs_kept=docs, max_sents=6, tau=-1.0)

    first_index = {d: ids.index(d) for d in dict.fromkeys(ids)}
    for doc, start in first_index.items():
        block = [i for i, d in enumerate(ids) if d == doc]
        assert block == list(range(block[0], block[0] + len(block))), ids


def test_unanswered_question_yields_an_explicit_abstention():
    """An empty string is not a usable abstention marker downstream."""
    ns = load(
        ["qa_pipeline"],
        env={
            "List": list, "Any": object, "CrossEncoder": object,
            "CFG": {"bm25_k": 10, "idk_str": "I don't know."},
            "hybrid_candidates": lambda q, d, b, k: [Doc("c", {"doc_id": "d1"})],
            "rerank_snippets": lambda q, docs, rr, k: docs,
            "compress": lambda q, docs_kept, max_sents, tau: (["s"], ["d1"], False),
            "answer_with_context": lambda q, s, i, max_ctx_tokens: ("", i),
            "is_refusal": PURE["is_refusal"],
        },
    )
    preds, _ = ns["qa_pipeline"](
        questions=["q"], dense_retriever=None, bm25=None, reranker=None,
        k_final=8, k_retry=12, tau=0.5, tau_retry=0.47,
        max_sents=8, max_sents_retry=10, max_ctx_tokens=1400,
    )
    assert preds == ["I don't know."]
