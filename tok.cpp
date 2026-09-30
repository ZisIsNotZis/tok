#include <iostream>
#include <unordered_map>
#include <vector>
#include <thread>
#include <mutex>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <algorithm>
#include <queue>
#include <cstring>
#include <cstdlib>

struct Gram {
    size_t i;
    unsigned char len;
    uint64_t hash;
    static const char* base;

    // 131-rolling hash: h = a[0]*131^(len-1) + ... + a[len-1]
    Gram(size_t i, unsigned char len) : i(i), len(len) {
        uint64_t h = 0;
        for (size_t j = 0; j < len; j++) {
            h = h * 131 + (unsigned char)base[i + j];
        }
        hash = h;
    }
    Gram(size_t i, unsigned char len, uint64_t hash) : i(i), len(len), hash(hash) {}

    const char* data() const { return base + i; }
    bool operator==(const Gram &other) const {
        return len == other.len && std::memcmp(data(), other.data(), len) == 0;
    }
};

const char* Gram::base = nullptr;

namespace std {
    template <>
    struct hash<Gram> {
        size_t operator()(const Gram &g) const {
            return g.hash;
        }
    };
}

// Drop the last character from the gram (using rolling hash property).
inline Gram drop_last(const Gram &g) {
    uint64_t new_hash = (g.hash - (unsigned char)g.base[g.i + g.len - 1]) / 131;
    return Gram(g.i, g.len - 1, new_hash);
}

// Heap entry for reduction.
struct HeapEntry {
    unsigned count;
    Gram gram;
};

// Comparator: first by count (lower counts come first), then by gram length (longer grams get pruned first).
struct HeapEntryComparator {
    bool operator()(const HeapEntry &a, const HeapEntry &b) const {
        if (a.count != b.count)
            return a.count > b.count; // lower count has higher priority
        return a.gram.len < b.gram.len; // tie-breaker: larger len first
    }
};

std::string replace(std::string_view s, std::string_view old_sub, std::string_view new_sub) {
    std::string result(s);
    size_t pos = 0;
    while ((pos = result.find(old_sub, pos)) != std::string::npos) {
        result.replace(pos, old_sub.length(), new_sub);
        pos += new_sub.length();
    }
    return result;
}

// Reduce the map of grams until its size is <= target_size.
// The reduction uses a heap (with versioning) and "drops" the last character when pruning.
void grams_reduce(std::unordered_map<Gram, unsigned> &cnt, unsigned target_size) {
    std::priority_queue<HeapEntry, std::vector<HeapEntry>, HeapEntryComparator> heap;
    for (auto &p : cnt) {
        heap.push(HeapEntry{ p.second, p.first });
    }
    while (cnt.size() > target_size) {
        HeapEntry cur = heap.top();
        heap.pop();
        auto it = cnt.find(cur.gram);
        if (it == cnt.end() || it->second != cur.count)
            continue;
        cnt.erase(it);
        if (cur.gram.len > 2) {
            Gram shorter = drop_last(cur.gram);
            unsigned newCount = cur.count;
            auto it2 = cnt.find(shorter);
            if (it2 != cnt.end())
                newCount += it2->second;
            cnt[shorter] = newCount;
            heap.push({ newCount, shorter });
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
    std::mutex cnt_mutex;
    size_t chunk = total / num_threads;

    for (int t = 0; t < num_threads; t++) {
        size_t start = t * chunk;
        size_t end = (t == num_threads - 1) ? total : (t + 1) * chunk;
        threads.emplace_back([=, &cnt, &cnt_mutex]() {
	    std::cerr << t << " count\n";
	    std::unordered_map<Gram, unsigned> local;
            for (size_t i = start; i < end; i++) {
                Gram g(i, ngram_size);
                local[g]++;
                if (local.size() >= max_size) {
	    	    std::cerr << t << " reduce\n";
                    grams_reduce(local, target_size);
	    	    std::cerr << t << " count\n";
		}
            }
	    std::cerr << t << " sync\n";
            std::lock_guard<std::mutex> lock(cnt_mutex);
            for (auto &p : local) {
                cnt[p.first] += p.second;
            }
	    std::cerr << t << " end\n";
        });
    }
    for (auto &th : threads)
        th.join();

    grams_reduce(cnt, target_size);

    // Output each gram: its data (the gram substring) immediately followed by 0xff.
    std::cout << '{';
    for (auto &p : cnt) {
        std::cout << '"' << replace(replace(std::string_view(data+p.first.i, p.first.len),"\n","\\n"),"\"","\\\"") << "\":" << p.second << ',';
    }
    std::cout << '}';

    munmap(data, filesize);
    return 0;
}

