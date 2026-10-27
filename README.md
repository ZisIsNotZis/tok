**English** | [简体中文](README.zh-CN.md)

<p align="center"><img src="assets/logo.svg" alt="tok" width="200"></p>

# tok

Bounded-vocabulary n-gram tokenizer experiments in C++17, CUDA and NumPy.

**Count every n-gram in a corpus, then shrink the overflowing working set by dropping or halving its cheapest grams — so a fixed-size vocabulary falls straight out of raw counts, without running BPE to convergence.**

This is a scratch lab, not a library: six independent `main`s and five notebooks, all circling the same idea. There is no build system, no test suite and no package.

## Quick start

No dependencies beyond a C++17 compiler — the corpus is just a file, and any file will do.

```sh
git clone https://github.com/ZisIsNotZis/tok.git && cd tok
g++ -O2 -pthread -o /tmp/tok tok.cpp
/tmp/tok README.md 5 4096 65536     # corpus, ngram, target_vocab, max_working_set
```

It prints a JSON object of `{"gram": count}` to stdout and streams `count`/`reduce`/`sync` progress to stderr. Exit code `1` means fewer than four arguments.

## What it can do now

- **Streaming n-gram counting** over an `mmap`'d corpus, split across 12 threads, each with its own `unordered_map` flushed into a shared map at the end (`tok.cpp`, `tok2.cpp`, `tok4.cpp`).
- **A hard working-set cap.** Counts never grow unbounded: when a thread-local map reaches `max_size`, it is reduced back to `target_size` in place (`tok.cpp`, `tok4.cpp`).
- **Four reduction flavours**, all "erase the cheapest gram and fold its count into something shorter":
  - drop the last character (`tok.cpp`, `tok5.cpp`),
  - split a gram in half, both halves kept when they are at least 2 bytes (`tok2.cpp`),
  - prune by `len × count` score (`tok4.cpp`, `tok5.cpp`, `unigram.cpp`),
  - the same by `count` first and length as tie-break (`tok.cpp`, `tok2.cpp`).
- **Versioned heaps.** Every entry carries the count/score it was pushed with; an entry whose live value no longer matches is dropped, so grams can be re-pushed after they grow without ever rebuilding the queue.
- **O(1) shortening via rolling hashes.** A 131-base rolling hash makes dropping the last byte a subtraction and a divide (`tok.cpp`); `tok5.cpp` uses a shift-xor hash with a matching `pop()`.
- **A CUDA port of the same loop** (`unigram.cu`): open-addressing device hash map with tombstones, a device binary heap, scan per level and a level-by-level parallel merge back into one map.
- **A two-phase CLI** (`tok5.cpp`): `count` emits `[u8 len][u32 count][gram bytes]` records, `reduce` merges many such shards with one global reduction — the shape a multi-machine version would need.
- **Notebooks that turn grams into a tokenizer**: `tok3.ipynb` builds the merge list and `tokenizers` model JSON from gram counts, `tok1.ipynb`/`tok2.ipynb` prototype counting and heap reduction in NumPy, `tok.ipynb` trains an HF BPE as the baseline, `tok4.ipynb` is a pure-Python mirror of the C++ reduce/count/merge loop over a `mmap`'d corpus.

## Architecture

Everything is one loop with three knobs: how grams are counted, what a gram becomes when it is pruned, and how the pruned counts are merged back. `tok.cpp` is the reference; the later files change one knob each — `tok2.cpp` splits instead of truncating, `tok4.cpp` scores by `len × count` and uses `std::async` instead of raw threads, `unigram.cpp` shrinks the key to a `(hash, size)` pair and `unigram.cu` moves that to the device. The notebooks sit downstream: they consume gram counts and emit tokenizer artefacts.

```
tok/
├── tok.cpp        reference: mmap + 12 threads + drop-last-character reduce
├── tok2.cpp       split-in-half reduce
├── tok4.cpp       len×count score, std::async, FNV-1a keys
├── tok5.cpp       count/reduce subcommands + binary shard format (does not compile as committed)
├── unigram.cpp    hash+size keys, tiled counting (CPU)
├── unigram.cu     the same on CUDA: device hash map, device heap, level merges
├── tok*.ipynb     NumPy prototypes, HF BPE baseline, gram-count → tokenizers JSON
├── assets/logo.svg
└── docs/project-status.md
```

The gram key never owns its bytes: it is an offset into the mapped corpus plus a length, with equality done by `memcmp` and hashing done over the bytes at that offset. That is what keeps a multi-million-entry map of long keys affordable, and it is why the corpus must stay mapped for the program's whole life.

## How to develop

There is no Makefile, CMake, `requirements.txt` or test runner — each source file is its own entry point and you compile the one you want:

```sh
g++ -O2 -pthread -o /tmp/tok   tok.cpp   && /tmp/tok   <corpus> <ngram> <target_vocab> <max_working_set>
g++ -O2 -pthread -o /tmp/tok2  tok2.cpp  && /tmp/tok2  <corpus> <ngram> <target_vocab> <max_working_set>
g++ -O2 -pthread -o /tmp/tok4  tok4.cpp  && /tmp/tok4  <corpus> <gram> <vocab>
g++ -O2 -pthread -o /tmp/uni   unigram.cpp && /tmp/uni <gram> <vocab> <vocabMax> <corpus>
nvcc -O2 -o /tmp/uni_cu unigram.cu && /tmp/uni_cu <gram> <vocab> <vocabMax> <vocabCap> <block> <threads> <corpus>
```

`tok5.cpp` is committed mid-edit and does **not** build: `output()` closes over names that are not in scope. Treat it as the design of the two-phase interface rather than as working code.

The notebooks need Jupyter, NumPy and HuggingFace `tokenizers`, and they expect a corpus and intermediates that are **not** in this repo — `py.txt`, `wikitext2.txt`, `py.64`, `py256.json`, read from `../../`. Re-running them means supplying those yourself; the cell order in each notebook is the pipeline.

Two rules that keep this directory readable: change one algorithm knob at a time and compare the resulting vocabulary, because these files exist to be compared against each other; and keep the gram-as-offset invariant, since a key that copies its own bytes changes memory use by an order of magnitude.

## How to contribute

See [CONTRIBUTING.md](CONTRIBUTING.md). Short version: this is an unmaintained archive, so open an issue before a large PR, and keep changes to one file's algorithm per PR so they stay comparable.

## Roadmap

**Implemented**

- Reference counter with a bounded working set and drop-last-character reduction (`tok.cpp`).
- Split-in-half reduction (`tok2.cpp`) and `len × count` scoring (`tok4.cpp`, `unigram.cpp`).
- CUDA port with a device hash map and device heap (`unigram.cu`).
- Two-phase `count` / `reduce` shard format, unbuilt (`tok5.cpp`).
- NumPy prototypes, HF BPE baseline and gram-count → `tokenizers` JSON conversion (notebooks).

**Planned — not started**

- Make `tok5.cpp` compile and finish the shard round-trip, so the two-phase idea is actually runnable.
- One build file and one entry point over the four CPU variants, for side-by-side comparisons on the same corpus.
- A small committed corpus plus a smoke test, so a checkout can be verified without the external `../../` data.

The longer-term interest is the one the notes keep returning to: whether a vocabulary emitted directly from count-min heaps with a hard cap can stand in for a tokenizer trained to convergence, and how much quality the drop-or-halve approximation costs against BPE at equal vocabulary size. Nothing here answers that yet; the pieces to measure it are what these files are.

## License

[AGPL-3.0-only](LICENSE) © ZisIsNotZis.
