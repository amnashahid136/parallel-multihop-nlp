# Parallelized NLP Pipeline for Multi-Hop Reasoning and Topic Classification

**Scaling Beyond the GIL with Ray, FAISS, and Dense Tensor Embeddings**

Amna Shahid ([i21-1713@nu.edu.pk](mailto:i21-1713@nu.edu.pk)) · Maheen Kamal ([i21-1351@nu.edu.pk](mailto:i21-1351@nu.edu.pk))
Department of Data Science, National University of Computer and Emerging Sciences (FAST-NUCES), Islamabad

Course project — Parallel and Distributed Computing (PDC)

---

## Overview

Standard Python NLP pipelines are bottlenecked by the Global Interpreter Lock (GIL), leaving multi-core hardware underutilized for CPU-bound preprocessing, retrieval, and classification. This project replaces every sequential bottleneck in a multi-hop reasoning + topic-classification pipeline with a parallelism-native alternative:

- **Ray** distributed task graphs for text preprocessing
- **MiniLM** (Sentence-Transformers) batched dense embeddings for vectorization
- **FAISS** (C++, BLAS-optimized) for batched multi-hop retrieval
- **XGBoost** (`n_jobs=-1`, `tree_method=hist`) for distributed classification

Evaluated on **NaturalQuestions, TriviaQA, WebQuestions, and FEVER** over a 50,000-document corpus.

## Key Results

| Stage | Sequential | Distributed | Speedup |
|---|---|---|---|
| Preprocessing (Ray) | 18.47 s | 5.38 s | **3.43×** |
| Retrieval (FAISS) | 91.23 s | 2.01 s | **45.39×** |
| Classification (XGBoost) | 6.84 s | 2.21 s | **3.10×** |

**FEVER fact-verification:** 98.0% accuracy, 0.9801 weighted F1 (vs. 97.5% / 0.974 sequential baseline).

Ablation confirms the gains come from two compounding factors: migrating to Ray eliminates GIL-bound preprocessing overhead, while replacing TF-IDF with FAISS + MiniLM delivers both the dominant 45.4× retrieval speedup *and* a 2.7-point accuracy improvement over the sparse-representation baseline — i.e., no speed/accuracy trade-off.

## Repository Contents
.
├── paper.pdf          # Full write-up: motivation, related work, methodology, results
├── pipeline.ipynb      # End-to-end implementation (preprocessing, embedding, retrieval, classification)
├── requirements.txt    # Pinned dependencies
└── README.md

## Pipeline Architecture

1. **Distributed Preprocessing** — corpus partitioned into 16 chunks, dispatched to Ray remote workers (`@ray.remote`), aggregated via `ray.get()`.
2. **Dense Vectorization** — `all-MiniLM-L6-v2` batch-encodes documents into 384-dim vectors (`batch_size=256`), ~1,057 docs/sec.
3. **FAISS Retrieval** — `IndexFlatIP` over the corpus; all query embeddings submitted as a single batched matrix for one C++-level `index.search()` call.
4. **Two-Hop Reasoning** — Hop 1 retrieves the closest document; Hop 2 re-encodes `query ⊕ hop1_doc` and retrieves again, enabling ComposeRAG-style compositional reasoning without sequential loops.
5. **Distributed Classification** — XGBoost (`n_jobs=-1`, `tree_method=hist`, `n_estimators=150`) trained on MiniLM embeddings for FEVER's 3-class fact-verification task.

## Setup & Usage

```bash
git clone https://github.com/amnashahid136/parallel-multihop-nlp.git
cd parallel-multihop-nlp
pip install -r requirements.txt
jupyter notebook pipeline.ipynb
```

The notebook was developed and benchmarked on **Google Colab** (4 physical CPU cores, ~12 GB RAM, no GPU). No special hardware is required to reproduce the results — the entire speedup comes from architecture, not hardware.

## Datasets

Loaded via Hugging Face `datasets`: NaturalQuestions, TriviaQA, WebQuestions, FEVER (500 samples each; retrieval corpus expanded to 50,000 documents). No raw dataset files are stored in this repo — they're pulled at runtime.

## Citation

If you reference this work, please cite the accompanying paper (`paper.pdf`):
Shahid, A., Kamal, M. (2025). Parallelized NLP Pipeline for Multi-Hop Reasoning
and Topic Classification: Scaling Beyond the GIL with Ray, FAISS, and Dense
Tensor Embeddings. FAST-NUCES Islamabad.

## License

MIT — see [LICENSE](LICENSE).
