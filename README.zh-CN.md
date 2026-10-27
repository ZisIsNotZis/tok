**简体中文** | [English](README.md)

<p align="center"><img src="assets/logo.svg" alt="tok" width="200"></p>

# tok

用 C++17、CUDA 和 NumPy 做的「有界词表 n-gram 分词器」实验。

**先把语料里出现的每个 n-gram 都数一遍，一旦工作集超出上限，就把最便宜的那些 gram 丢掉或对半砍，让词表规模自动收敛到指定大小——不跑 BPE 到底也能直接得到词表。**

这是一间草稿实验室，不是库：六个各自独立的 `main`、五份 notebook，围着同一个想法转。没有构建系统、没有测试、也没有打包。

## 快速开始

除 C++17 编译器外不需要任何依赖——语料就是一个文件，随便哪个文件都行。

```sh
git clone https://github.com/ZisIsNotZis/tok.git && cd tok
g++ -O2 -pthread -o /tmp/tok tok.cpp
/tmp/tok README.md 5 4096 65536     # 语料、ngram、目标词表、工作集上限
```

结果以 `{"gram": count}` 形式的 JSON 打到 stdout，`count`/`reduce`/`sync` 的过程信息走 stderr。参数少于四个时退出码为 `1`。

## 目前能做什么

- **流式 n-gram 统计**：把语料 `mmap` 进来，切成 12 个线程，各自维护一张 `unordered_map`，最后汇总到共享表（`tok.cpp`、`tok2.cpp`、`tok4.cpp`）。
- **硬性工作集上限**：计数永远不会无限膨胀——线程本地的表涨到 `max_size` 就地缩回 `target_size`（`tok.cpp`、`tok4.cpp`）。
- **四种缩减口味**，本质都是「删掉最便宜的 gram，把它的计数并进更短的 gram」：
  - 去掉最后一个字符（`tok.cpp`、`tok5.cpp`）；
  - 对半劈开，两半长度都 ≥2 时都保留（`tok2.cpp`）；
  - 按 `len × count` 分数淘汰（`tok4.cpp`、`tok5.cpp`、`unigram.cpp`）；
  - 先按 `count`、再按长度做次级排序（`tok.cpp`、`tok2.cpp`）。
- **带版本的堆**：每个堆元素都记着入堆时的计数/分数，与实际表里对不上就直接丢弃，所以 gram 涨大以后可以重新入堆，堆永远不用重建。
- **滚动哈希带来的 O(1) 缩短**：131 进制的滚动哈希让「去掉最后一个字节」变成一次减法和一次除法（`tok.cpp`）；`tok5.cpp` 用移位异或哈希，并配一个对应的 `pop()`。
- **同一套循环的 CUDA 版**（`unigram.cu`）：设备端开放寻址哈希表（带墓碑标记）、设备端二叉堆、逐层扫描，再逐层并行合并回一张表。
- **两阶段的命令行**（`tok5.cpp`）：`count` 输出 `[u8 len][u32 count][gram 字节]` 记录，`reduce` 把许多这样的分片合并成一次全局缩减——分布式版本需要的就是这个形状。
- **把 gram 变成分词器的 notebook**：`tok3.ipynb` 从 gram 计数拼出 merge 列表和 `tokenizers` 模型 JSON；`tok1.ipynb`/`tok2.ipynb` 用 NumPy 复刻计数与堆缩减；`tok.ipynb` 训练一个 HF BPE 当基线；`tok4.ipynb` 是纯 Python 版，在 `mmap` 的语料上跑同一套 reduce/count/merge。

## 结构

所有东西都是同一个循环，只是三个旋钮不同：gram 怎么数、被淘汰时变成什么、缩减后的计数怎么并回去。`tok.cpp` 是基准，后面的文件各改一个旋钮——`tok2.cpp` 改成劈开而不是截断，`tok4.cpp` 改成 `len × count` 打分并用 `std::async` 代替裸线程，`unigram.cpp` 把键缩成 `(hash, size)`，`unigram.cu` 把它搬到设备上。notebook 在这之下游：吃 gram 计数，吐出分词器产物。

```
tok/
├── tok.cpp        基准：mmap + 12 线程 + 去掉末字符的缩减
├── tok2.cpp       对半劈开的缩减
├── tok4.cpp       len×count 打分、std::async、FNV-1a 键
├── tok5.cpp       count/reduce 两个子命令 + 二进制分片格式（按现状无法编译）
├── unigram.cpp    hash+size 键，分块计数（CPU）
├── unigram.cu     同一套东西上 CUDA：设备哈希表、设备堆、逐层合并
├── tok*.ipynb     NumPy 原型、HF BPE 基线、gram 计数 → tokenizers JSON
├── assets/logo.svg
└── docs/project-status.md
```

gram 键从不自己持有字节：它只是「在映射进来的语料里的偏移 + 长度」，相等靠 `memcmp`，哈希就在那个偏移上现算。这让一张几百万条长键的表才扛得住，也意味着语料必须在整个进程生命周期里保持映射。

## 开发方式

没有 Makefile、CMake、`requirements.txt`，也没有测试运行器——每个源文件都是自己的入口，用到哪个就编哪个：

```sh
g++ -O2 -pthread -o /tmp/tok   tok.cpp   && /tmp/tok   <语料> <ngram> <目标词表> <工作集上限>
g++ -O2 -pthread -o /tmp/tok2  tok2.cpp  && /tmp/tok2  <语料> <ngram> <目标词表> <工作集上限>
g++ -O2 -pthread -o /tmp/tok4  tok4.cpp  && /tmp/tok4  <语料> <gram> <词表>
g++ -O2 -pthread -o /tmp/uni   unigram.cpp && /tmp/uni <gram> <词表> <词表上限> <语料>
nvcc -O2 -o /tmp/uni_cu unigram.cu && /tmp/uni_cu <gram> <词表> <词表上限> <词表容量> <block> <threads> <语料>
```

`tok5.cpp` 是改到一半就提交的，**编不过**：`output()` 里用了不在作用域里的名字。把它当作两阶段接口的设计稿，而不是能跑的代码。

notebook 需要 Jupyter、NumPy 和 HuggingFace `tokenizers`，并且依赖本仓库**没有**的语料和中间文件——`py.txt`、`wikitext2.txt`、`py.64`、`py256.json`，都从 `../../` 读。想重跑就自己准备这些文件；每个 notebook 里的单元顺序就是整条流水线。

两条规矩让这个目录还能读：一次只改一个算法旋钮，然后比较产出的词表，因为这些文件存在的意义就是互相比较；以及守住「gram 就是偏移」这条不变量，因为一个自己拷贝字节的键会让内存占用翻上好几个数量级。

## 参与

见 [CONTRIBUTING.md](CONTRIBUTING.md)。一句话：这是一个不再维护的存档，所以动大手术前先开 issue，并且每个 PR 只改一个文件的算法，方便放在一起比较。

## 路线图

**已实现**

- 带工作集上限、按「去掉末字符」缩减的基准计数器（`tok.cpp`）。
- 对半劈开缩减（`tok2.cpp`）与 `len × count` 打分（`tok4.cpp`、`unigram.cpp`）。
- CUDA 版：设备哈希表加设备堆（`unigram.cu`）。
- 两阶段 `count` / `reduce` 分片格式，但无法编译（`tok5.cpp`）。
- NumPy 原型、HF BPE 基线、gram 计数 → `tokenizers` JSON 转换（notebook）。

**计划中——尚未开始**

- 让 `tok5.cpp` 能编译，跑通分片的往返，使两阶段的想法真的可用。
- 一份构建文件、一个入口，把四个 CPU 变体在同一份语料上并排比较。
- 提交一份小语料加一个冒烟测试，让 checkout 下来不依赖外部 `../../` 数据也能验证。

更长远的兴趣，是这些笔记一直绕着转的那个问题：在硬性上限下直接从 count-min 堆里吐出的词表，能不能顶替训练到收敛的分词器；以及在同等词表规模下，「丢掉还是砍半」这个近似到底损失多少质量。这里还没有答案——这些文件提供的只是把它量出来的零件。

## 许可证

[AGPL-3.0-only](LICENSE) © ZisIsNotZis。
