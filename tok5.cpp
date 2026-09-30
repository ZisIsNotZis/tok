#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <sys/mman.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>
#include <algorithm>
static const char* DAT = nullptr;
static uint32_t LEN = 0;
struct Gram {
    uint32_t i;
    uint8_t l;
    uint64_t hash;
    inline Gram(uint32_t i, uint8_t l) : i(i), l(l), hash(0) {
        for(l=0; l<this->l; l++){
            hash ^= ((uint64_t)DAT[i+l]) << l * 6 % 27;
        }
    }
    inline bool pop() {
        if (l==2) return true;
        l--;
        hash ^= ((uint64_t)DAT[i+l]) << l * 6 % 27;
        return false;
    }
    inline bool operator==(const Gram &g) const {
        return l == g.l && !std::memcmp(DAT + i, DAT + g.i, l);
    }
};
struct GramHash {
    inline uint64_t operator()(const Gram &g) const {
        return g.hash;
    }
};
using GramMap = std::unordered_map<Gram, uint32_t, GramHash>;
static void reduce(GramMap &m, uint32_t vocab) {
    struct Heap {
        uint32_t score;
        Gram g;
    };
    struct HeapComp {
        inline bool operator()(const Heap &a, const Heap &b) {
            return a.score > b.score;
        }
    };
    std::vector<Heap> heap;
    heap.reserve(m.size());
    for (auto it : m)
        heap.push_back({it.first.l*it.second, it.first});
    std::make_heap(heap.begin(), heap.end(), HeapComp());
    while (m.size() > vocab) {
        std::pop_heap(heap.begin(), heap.end(), HeapComp());
        auto &h = heap.back();
	auto it = m.find(h.g);
        uint32_t c;
        if (it == m.end() || h.score != h.g.l * (c=it->second)) {
            heap.pop_back();
            continue;
        }
        m.erase(it);
        if (h.g.pop()) {
            heap.pop_back();
            continue;
        }
        if ((it=m.find(h.g)) == m.end()) {
            m[h.g] = c;
        } else {
            c += it->second;
            it->second = c;
        }
        h.score = h.g.l * c;
        std::push_heap(heap.begin(), heap.end(), HeapComp());
    }
}
static GramMap count(uint32_t vocab, uint8_t gram){
    GramMap m;
    for(uint32_t i=0; i<LEN-gram+1; i++){
	m[Gram(i, gram)] += 1;
        if(m.size() > vocab*10){
            reduce(m, vocab);
        }
    }
    reduce(m, vocab);
    return m;
}
static void output(const GramMap &m) {
    for (auto g : m) {
	std::fwrite(&g.first.l, sizeof(uint8_t), 1, stdout);
        std::fwrite(&g.second, sizeof(uint32_t), 1, stdout);
        std::fwrite(DAT + g.first.i, sizeof(char), g.first.l, stdout);
    }
    reduce(m, vocab);
    return m;
}
static void count_mode(char** argv) {
    uint32_t vocab = std::atoi(argv[0]);
    uint8_t gram = std::atoi(argv[1]);
    const char* file = argv[2];
    uint32_t from = std::atoi(argv[3]);
    uint32_t to = std::atoi(argv[4]);
    LEN = to - from;
    int fd = open(file, O_RDONLY);
    DAT = static_cast<const char*>(mmap(nullptr, to, PROT_READ, MAP_SHARED, fd, 0)) + from;
    close(fd);
    output(count(vocab, gram));
    munmap(const_cast<char*>(DAT), LEN);
}
static void reduce_mode(int argc, char** argv) {
    uint32_t vocab = std::atoi(argv[0]);
    uint8_t gram = std::atoi(argv[1]);
    GramMap m;
    static std::vector<char> dat;
    for (int i = 2; i < argc; i++) {
        const char* file = argv[i];
        int fd = open(file, O_RDONLY);
        off_t fsize = lseek(fd, 0, SEEK_END);
        char* file_data = static_cast<char*>(mmap(nullptr, fsize, PROT_READ, MAP_SHARED, fd, 0));
        close(fd);
        size_t offset = 0;
        while(offset < static_cast<size_t>(fsize)) {
            uint8_t l = *reinterpret_cast<uint8_t*>(file_data + offset);
            offset += sizeof(uint8_t);
            uint32_t count = *reinterpret_cast<uint32_t*>(file_data + offset);
            offset += sizeof(uint32_t);
            uint32_t beg = dat.size();
            dat.insert(dat.end(), file_data + offset, file_data + offset + std::min(l, gram));
            DAT = dat.data();
            Gram key(beg, std::min(l, gram));
            auto it = m.find(key);
            if (it == m.end()) {
                m[key] = count;
            } else {
                it->second += count;
                dat.resize(beg);
            }
            offset += l;
        }
        munmap(file_data, fsize);
    }
    reduce(m, vocab);
    output(m);
}
int main(int argc, char **argv) {
    if(argc > 1) {
        std::string cmd = argv[1];
        if(cmd == "count") {
            count_mode(argv + 2);
        } else if(cmd == "reduce") {
            reduce_mode(argc - 2, argv + 2);
        }
	return 0;
    }
    std::cerr << "count <vocab> <gram> <file> <from> <to>|reduce <vocab> <gram> <countfile>...\n";
    return 1;
}
