#include <iostream>
#include <unordered_map>
#include <vector>
#include <thread>
#include <mutex>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <queue>
#include <cstring>
#include <cstdlib>
#include <string_view>
#include <string>

struct Gram {
    size_t i;
    unsigned char len;
    static const char* base;
    Gram(size_t i, unsigned char len): i(i), len(len) {}
    const char* data() const { return base + i; }
    // 131-rolling hash computed on the fly.
    uint64_t hash() const {
        uint64_t h = 0;
        for (size_t j = 0; j < len; j++) h = h * 131 + (unsigned char)base[i + j];
        return h;
    }
    bool operator==(const Gram &o) const {
        return len == o.len && std::memcmp(data(), o.data(), len) == 0;
    }
};
const char* Gram::base = nullptr;

namespace std {
template<> struct hash<Gram> {
    size_t operator()(const Gram &g) const { return g.hash(); }
};
}

// Split a gram into two halves (using integer division) and return each half
// only if its length is at least 2.
inline std::vector<Gram> split_half(const Gram &g) {
    std::vector<Gram> halves;
    if(g.len < 2) return halves;
    unsigned char left = g.len / 2, right = g.len - left;
    if(left >= 2) halves.push_back(Gram(g.i, left));
    if(right >= 2) halves.push_back(Gram(g.i + left, right));
    return halves;
}

struct HeapEntry { unsigned count; Gram gram; };
struct HeapEntryComparator {
    bool operator()(const HeapEntry &a, const HeapEntry &b) const {
        return a.count != b.count ? a.count > b.count : a.gram.len < b.gram.len;
    }
};

// Simple string replace (used for JSON output escaping).
std::string replace(std::string_view s, std::string_view o, std::string_view n) {
    std::string r(s);
    size_t pos = 0;
    while ((pos = r.find(o, pos)) != std::string::npos) {
        r.replace(pos, o.size(), n);
        pos += n.size();
    }
    return r;
}

// Reduce the gram map until its size is <= target_size by splitting each gram.
void grams_reduce(std::unordered_map<Gram, unsigned> &cnt, unsigned target_size) {
    std::priority_queue<HeapEntry, std::vector<HeapEntry>, HeapEntryComparator> heap;
    for (auto &p : cnt) heap.push({p.second, p.first});
    while (cnt.size() > target_size) {
        HeapEntry cur = heap.top(); heap.pop();
        auto it = cnt.find(cur.gram);
        if (it == cnt.end() || it->second != cur.count) continue;
        cnt.erase(it);
        for (auto &h : split_half(cur.gram)) {
            unsigned newCount = cur.count;
            auto it2 = cnt.find(h);
            if (it2 != cnt.end()) newCount += it2->second;
            cnt[h] = newCount;
            heap.push({newCount, h});
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 5) return 1;
    int fd = open(argv[1], O_RDONLY);
    off_t filesize = lseek(fd, 0, SEEK_END);
    char* data = (char*)mmap(nullptr, filesize, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    Gram::base = data;
    
    unsigned ngram_size  = std::atoi(argv[2]);
    unsigned target_size = std::atoi(argv[3]);
    unsigned max_size    = std::atoi(argv[4]);
    size_t total = filesize - ngram_size + 1;
    
    std::unordered_map<Gram, unsigned> cnt;
    cnt.reserve(max_size);
    
    const int num_threads = 12;
    std::vector<std::thread> threads;
    std::mutex m;
    size_t chunk = total / num_threads;
    for (int t = 0; t < num_threads; t++) {
        size_t start = t * chunk;
        size_t end = (t == num_threads - 1) ? total : (t + 1) * chunk;
        threads.emplace_back([=, &cnt, &m]() {
            std::unordered_map<Gram, unsigned> local;
            for (size_t i = start; i < end; i++) {
                local[Gram(i, ngram_size)]++;
                if (local.size() >= max_size)
                    grams_reduce(local, target_size);
            }
            std::lock_guard<std::mutex> lock(m);
            for (auto &p : local) cnt[p.first] += p.second;
        });
    }
    for (auto &th : threads) th.join();
    grams_reduce(cnt, target_size);
    
    std::cout << '{';
    for (auto &p : cnt)
        std::cout << '"' 
                  << replace(replace(std::string_view(data + p.first.i, p.first.len), "\n", "\\n"), "\"", "\\\"")
                  << "\":" << p.second << ',';
    std::cout << '}';
    
    munmap(data, filesize);
    return 0;
}

