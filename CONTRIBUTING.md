# Contributing to tok

This directory is an **unmaintained archive** of tokenizer-vocabulary
experiments: six independent `main`s and five notebooks, no build system, no
test suite. Bugs are not being fixed on a schedule. Contributions are welcome
anyway, under three rules.

## Scope

In scope:

- fixing code that does not compile — `tok5.cpp` is currently broken; see
  [`README.md`](README.md#how-to-develop);
- fixing a wrong reduction invariant or a corrupted output record;
- adding a comparison between two reduction flavours on the same corpus.

Out of scope: turning this into a library, adding a package/build system, or
porting it to another language. Those are separate projects, not polish.

## Before you open a PR

- Open an issue describing the corpus, the parameters and the observed vs
  expected output. Timing or quality numbers need the corpus name and size.
- One algorithm knob per PR, and one file per PR where possible: these files
  exist to be compared against each other, so a diff that changes two of them at
  once cannot be checked.
- Keep the gram-as-offset invariant. A key that copies its own bytes changes
  memory use by orders of magnitude; that is a redesign, not a fix.

## Checks

There is nothing to install and no test runner. State in the PR body which of
these you ran and what came out:

```sh
g++ -O2 -pthread -o /tmp/tok  tok.cpp  && /tmp/tok  <corpus> <ngram> <target_vocab> <max_working_set>
g++ -O2 -pthread -o /tmp/tok2 tok2.cpp && /tmp/tok2 <corpus> <ngram> <target_vocab> <max_working_set>
g++ -O2 -pthread -o /tmp/tok4 tok4.cpp && /tmp/tok4 <corpus> <gram> <vocab>
g++ -O2 -pthread -o /tmp/uni  unigram.cpp && /tmp/uni <gram> <vocab> <vocabMax> <corpus>
nvcc -O2 -o /tmp/uni_cu unigram.cu && /tmp/uni_cu <gram> <vocab> <vocabMax> <vocabCap> <block> <threads> <corpus>
```

Any UTF-8 text file works as a smoke corpus; the notebooks queue in
`tok*.ipynb` additionally need Jupyter, NumPy, HuggingFace `tokenizers` and
corpora that are **not** in this repo (they read from `../../`).

## Issue etiquette

Include the command line you ran, the compiler version (`g++ --version`,
`nvcc --version`) and whether the result was a crash, a wrong count or a
vocabulary that never reaches the target size. "Does not converge" reports
should say how far the working set got and on what corpus.
