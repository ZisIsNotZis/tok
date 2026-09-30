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
#include <thread>
struct Key{
	uint64_t hash;
	uint8_t sz;
	inline bool operator==(Key k)const{
		return hash == k.hash && sz == k.sz;//if hash and sz equal, gram content probably equal
    }
};
template<>struct std::hash<Key>{
    inline uint64_t operator()(Key k)const{
    	return k.hash;
    }
};
struct Val{
	uint32_t pos;
	uint32_t cnt;
};
inline uint64_t rotl64(uint64_t dat, uint16_t i){
	i = i*6%64;
	return (dat<<i)|(dat>>64-i);
}
inline void reduce(uint8_t *dat, std::unordered_map<Key,Val> &map, uint32_t vocab){
	struct Keyscore{
		Key key;
		uint32_t score;
		inline bool operator<(Keyscore k)const{
			return score < k.score;
		}
    };
	std::vector<Keyscore> heap;
	heap.reserve(map.size());
	for(auto it: map)
        heap.push_back({it.first, it.first.sz*it.second.cnt});
    std::make_heap(heap.begin(), heap.end());
	while(map.size() > vocab){
		std::pop_heap(heap.begin(), heap.end());
		auto &back = heap.back();//get min
		auto it = map.find(back.key);
		if(it == map.end() || back.key.sz*it->second.cnt != back.score){//entry outdated, ignore
			heap.pop_back();
			continue;
		}
		Val val = it->second;
		map.erase(it);
		if(back.key.sz == 2){//gram too short, discard
        	heap.pop_back();
        	continue;
        }
        back.key.sz--;
        back.key.hash ^= rotl64(dat[val.pos+back.key.sz], back.key.sz);
        if((it=map.find(back.key)) == map.end())
        	map[back.key] = val;
        else{
        	val.cnt += it->second.cnt;
        	it->second.cnt = val.cnt;
        }
        back.score = back.key.sz*val.cnt;
        std::push_heap(heap.begin(), heap.end());
	}
}
inline void reduce2(uint8_t *dat, std::unordered_map<Key,Val> &map, std::unordered_map<Key,Val> &map2, uint32_t vocab, uint32_t vocabMax){
	for(auto it: map2){
		auto it2 = map.find(it.first);
		if(it2 == map.end()){
			map[it.first] = it.second;
			if(map.size() > vocabMax)
				reduce(dat, map, vocab);
		}else
			it2->second.cnt += it.second.cnt;
	}
	reduce(dat, map, vocab);
}
inline void count(uint8_t *dat, std::unordered_map<Key,Val>& map, uint32_t pos, uint32_t sz, uint8_t gram, uint32_t vocab, uint32_t vocabMax){
	uint64_t hash = 0;
	for(uint8_t i=0; i<gram; i++)
		hash ^= rotl64(dat[pos+i], i);
	map[{hash, gram}] = {pos, 1};
	for(uint32_t i=0; i<sz; i++){
		hash = rotl64(hash ^ rotl64(dat[pos+i], 0), 31) ^ rotl64(dat[pos+gram+i], gram-1); //31*6=-1*6 (mod 64)
		auto it = map.find({hash, gram});
		if(it == map.end()){
			map[{hash, gram}] = {pos+i+1, 1};
			if(map.size() > vocabMax)
				reduce(dat, map, vocab);
		}else
			it->second.cnt += 1;
	}
	reduce(dat, map, vocab);
}
int main(int argc, char **argv) {
	if(argc!=5) {
		fprintf(stderr, "<gram> <vocab> <vocabMax> <in>\n");
		return 1;
	}
	uint8_t gram = (uint8_t)atoi(argv[1]);
	uint32_t vocab = (uint32_t)atoi(argv[2]);
	uint32_t vocabMax = (uint32_t)atoi(argv[3]);
	int fd = open(argv[4], O_RDONLY);
	uint32_t sz = lseek(fd, 0, SEEK_END);
	uint8_t *dat = (uint8_t*)mmap(nullptr, sz, PROT_READ, MAP_SHARED, fd, 0);
	close(fd);
	std::vector<std::unordered_map<Key,Val>> maps(12);
	std::vector<std::thread> threads;
	for(int i=0; i<12; i++)
		threads.push_back(std::thread(count, dat, std::ref(maps[i]), (sz-gram+1)/12*i, (sz-gram+1)/12, gram, vocab, vocabMax));
	for(auto& it: threads)
		it.join();
	while(maps.size()>1){
		threads.clear();
		for(int i=0; i<maps.size()/2; i++)
			threads.push_back(std::thread(reduce2, dat, std::ref(maps[i]), std::ref(maps[(maps.size()+1)/2+i]), vocab, vocabMax));
		for(auto& it: threads)
			it.join();
		maps.resize((maps.size()+1)/2);
	}
	for(auto& it: maps[0]){
		fwrite(&it.first.sz, 1, 1, stdout);
		fwrite(dat+it.second.pos, 1, it.first.sz, stdout);
	}
	return 0;
}
