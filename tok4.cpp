#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <sys/mman.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>
#include <future>
#include <algorithm>
#include <thread>

static const char* g_data = nullptr;

struct NgramKey {
    uint32_t idx; // offset into g_data
    uint8_t l;    // length of gram
    bool operator==(const NgramKey &o) const {
        return l == o.l && memcmp(g_data + idx, g_data + o.idx, l) == 0;
    }
};

struct NgramHash {
    uint32_t operator()(const NgramKey &k) const {
        // FNV-1a hash for 5 bytes (4 bytes for idx and 1 for l) is not sufficient,
        // so we hash the gram bytes themselves (length k.l) for better distribution.
        uint32_t hash = 2166136261u;
        const unsigned char* ptr = reinterpret_cast<const unsigned char*>(g_data + k.idx);
        for (uint8_t i = 0; i < k.l; i++) {
            hash ^= ptr[i];
            hash *= 16777619u;
        }
        return hash;
    }
};

using NgramMap = std::unordered_map<NgramKey, uint32_t, NgramHash>;

struct HeapEntry {
    uint32_t score; // = length * count (score never exceeds 32 bits)
    NgramKey key;
};

struct HeapComp {
    bool operator()(const HeapEntry &a, const HeapEntry &b) {
        return a.score > b.score; // min-heap: lower score has higher priority
    }
};

static void reduce(NgramMap &counts, uint32_t vocab) {
    std::vector<HeapEntry> heap;
    heap.reserve(counts.size());
    for (auto &p : counts)
        heap.push_back({ p.first.l * p.second, p.first });
    std::make_heap(heap.begin(), heap.end(), HeapComp());
    while (counts.size() > vocab) {
        std::pop_heap(heap.begin(), heap.end(), HeapComp());
        HeapEntry e = heap.back();
        heap.pop_back();
        auto it = counts.find(e.key);
        if (it == counts.end() || e.score != e.key.l * it->second)
            continue;
        uint32_t cnt = it->second;
        counts.erase(it);
        if (e.key.l > 2) {
            NgramKey nk = { e.key.idx, static_cast<uint8_t>(e.key.l - 1) };
            uint32_t newCount = cnt;
            auto it2 = counts.find(nk);
            if (it2 != counts.end())
                newCount += it2->second;
            counts[nk] = newCount;
            HeapEntry newEntry = { nk.l * newCount, nk };
            heap.push_back(newEntry);
            std::push_heap(heap.begin(), heap.end(), HeapComp());
        }
    }
}

static void count_ngrams(const char *data, uint32_t len, uint8_t gram, uint32_t buf, uint32_t vocab, NgramMap &m) {
    for (uint32_t i = 0; i <= len - gram; i++) {
        NgramKey k = { i, gram };
        m[k] += 1;
        if (m.size() >= buf)
            reduce(m, vocab);
    }
}

int main(int argc, char **argv){
    int fd = open(argv[1], O_RDONLY);
    off_t fsize = lseek(fd, 0, SEEK_END);
    g_data = static_cast<const char*>(mmap(nullptr, fsize, PROT_READ, MAP_SHARED, fd, 0));
    close(fd);
    // parameters
    uint8_t gram = std::atoi(argv[2]);
    const uint32_t vocab = std::atoi(argv[3]);
    uint32_t buf = vocab * 100;
    uint32_t nthreads = 12;
    uint32_t chunk = fsize / nthreads;
    std::vector<std::future<NgramMap>> futures;
    for (uint32_t i = 0; i < nthreads; i++){
        uint32_t start = i * chunk;
        uint32_t end = (i == nthreads - 1) ? fsize : (i + 1) * chunk;
        uint32_t len = end - start;
        futures.push_back(std::async(std::launch::async, [=]() {
            NgramMap local;
            local.reserve(1024);
            count_ngrams(g_data + start, len, gram, buf, vocab, local);
            return local;
        }));
    }
    NgramMap global;
    for (auto &f : futures) {
        NgramMap part = f.get();
        for (auto &p : part)
            global[p.first] += p.second;
    }
    reduce(global, vocab);
    std::cout << "{";
    bool first = true;
    for (auto &p : global) {
        if (!first) std::cout << ",";
        first = false;
        std::cout << "\"";
        for (uint8_t i = 0; i < p.first.l; i++){
            char c = g_data[p.first.idx + i];
            if (c == '\"' || c == '\\')
                std::cout << '\\' << c;
            else if ( c == '\n' )
		std::cout << "\\n";
	    else
                std::cout << c;
        }
        std::cout << "\":" << p.second;
    }
    std::cout << "}";
    munmap(const_cast<char*>(g_data), fsize);
    return 0;
}

