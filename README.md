# tok — bounded-vocabulary n-gram tokenizer experiments (C++ / CUDA / Python)

**What:** A personal scratch lab for **vocabulary construction**: streaming
n-gram counters (C++17, 12-thread `mmap`+`unordered_map`) that keep an unbounded
n-gram stream bounded to a target vocabulary by a min-heap "reduce" step, in four
reduction flavours (drop-last-character, split-in-half, length×count scoring), a
naive **CUDA port** of the same algorithm (device hash map, tombstone deletion,
device binary heap, level-by-level parallel merge), and Python/NumPy notebooks
that turn the resulting gram counts into **HuggingFace `tokenizers` model JSON**
(gram-split -> merge list -> `vocab`).

**Why it exists:** Notes-to-self from trying to build a tokenizer whose lexicon is
emitted directly from count-min heaps with a hard vocab cap, instead of running
BPE to convergence. The recurring idea across `tok.cpp`/`tok2.cpp`/`tok4.cpp`/
`tok5.cpp`/`unigram.{cpp,cu}` is the same: count, and when the working set
overflows, *shrink the grams* (drop the tail, or halve them) by the lowest
`len × count` score, then re-merge — an approximation of the standard BPE merge
loop implemented with priority queues. It is not a library and has no consumers.

**How to run:** there is no build system or single entrypoint; each C++ file is an independent `main`.
```
g++ -O2 -pthread -o /tmp/tok tok.cpp && /tmp/tok <corpus> <ngram> <target_vocab> <max_working_set>
g++ -O2 -pthread -o /tmp/uni unigram.cpp && /tmp/uni <gram> <vocab> <vocabMax> <corpus>
nvcc -O2 -o /tmp/uni_cu unigram.cu && /tmp/uni_cu <gram> <vocab> <vocabMax> <vocabCap> <block> <threads> <corpus>
```
`tok5.cpp` uses a subcommand interface (`count` / `reduce`) and is **not
buildable as committed**. Notebook order:
`tok.ipynb` (byte-level + HF BPE baseline), `tok1.ipynb`/`tok2.ipynb` (gram
counting and heap reduction in NumPy), `tok3.ipynb` (turn grams + merges into a
`tokenizers` JSON), `tok4.ipynb` (pure-Python prototype of the same reduce/count/
merge loop over a `mmap`ed `../../wikitext2.txt`). The notebooks expect corpora and
intermediate files (`py.txt`, `wikitext2.txt`, `py.64`, `py256.json`) that live
**outside this directory** and are not committed.

**Status:** deprecation check — this is **current (2025–2026 era)** code: C++17
(`std::thread`, `std::unordered_map`, `sliding_window_view`-style NumPy), CUDA 12
(`cudaMalloc`/`cudaMemcpy`, `__global__` kernels), Python 3.12 + NumPy +
HuggingFace `tokenizers`. No TensorFlow/fastai/Vue/deprecated framework is
involved. The caveat is not age but **completeness**: it is an unpolished,
partly broken scratch archive, so treat it as
`deprecated (2025 era) — kept as an archive entry` in tone and as
`archive_only` for migration.

**License:** the repository's own code is licensed under [AGPL-3.0-only](LICENSE).
