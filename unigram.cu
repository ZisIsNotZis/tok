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
	__device__ inline bool operator==(Key k)const{
		return hash == k.hash && sz == k.sz;//if hash and sz equal, gram content probably equal
    }
};
struct Val{
	uint32_t pos;
	uint32_t cnt;
};
template<class V>struct Map{
	Key *key;
	V *val;
	uint32_t cap,sz;
	__device__ inline void init(){
		for(uint32_t i=0; i<cap; i++)
			key[i].sz = 0;
	}
	__device__ inline Key& operator()(Key k){
		uint32_t tomb = 0xFFFFFFFF;
		for(uint32_t i=k.hash&(cap-1);; i=(i+1)&(cap-1))
			if(key[i].sz == 0)
				return key[tomb==0xFFFFFFFF ? i : tomb];
			else if(key[i].sz == 1){
				if(tomb == 0xFFFFFFFF)
					tomb = i;
			}else if(key[i] == k)
				return key[i];
	}
	__device__ inline V& operator[](Key& k){
		return val[&k-key];
	}
};
struct Keyscore{
	Key key;
	uint32_t score;
	__device__ inline bool operator<(Keyscore k)const{
		return score < k.score;
	}
};
template<class T>struct Heap{
	T *t;
	uint32_t sz;
	__device__ inline void push(T tmp){
		uint32_t i = sz++;
        t[i] = tmp;
        while(i){
            uint32_t p = (i - 1) >> 1;
            if (!(t[i] < t[p])) break;
            T swapTmp = t[p];
            t[p] = t[i];
            t[i] = swapTmp;
            i = p;
        }
	}
	__device__ inline T pop(){
		T out = t[0];
        t[0] = t[--sz];
        uint32_t i = 0;
        while(true){
            uint32_t l = 2*i + 1;
            uint32_t r = l + 1;
            uint32_t s = i;
            if (l < sz && t[l] < t[s]) s = l;
            if (r < sz && t[r] < t[s]) s = r;
            if (s == i) break;
            T swapTmp = t[i];
            t[i] = t[s];
            t[s] = swapTmp;
            i = s;
        }
        return out;
	}
};
__device__ inline uint64_t rotl64(uint64_t dat, uint16_t i){
	i = i*6&63;
	return (dat<<i)|(dat>>64-i);
}
__device__ inline void reduce(uint8_t *dat, Map<Val>& map, Keyscore *keyscore, uint32_t vocab){
	Heap<Keyscore> heap{keyscore,0};
	for(uint32_t i=0; i<map.cap; i++){
		if(map.key[i].sz <= 1) continue;
        heap.push({map.key[i], map.key[i].sz*map.val[i].cnt});
	}
	while(map.sz > vocab){
		Keyscore back = heap.pop();
		Val val;
		{
			Key& it = map(back.key);
			if(it.sz <= 1) continue;
			val = map[it];
			if(back.key.sz*val.cnt != back.score) continue;
			it.sz = 1;
			map.sz--;
		}
		if(back.key.sz == 2) continue;
        back.key.sz--;
        back.key.hash ^= rotl64(dat[val.pos+back.key.sz], back.key.sz);
        {
        	Key& it = map(back.key);
			if(it.sz <= 1){
				it = back.key;
				map[it] = val;
				map.sz++;
			}else
				map[it].cnt = val.cnt = map[it].cnt + val.cnt;
		}
        back.score = back.key.sz*val.cnt;
        heap.push(back);
	}
}
__global__ void reduceKer(uint8_t *dat, Key *key, Val *val, Keyscore *keyscore, uint32_t sz, uint32_t vocab, uint32_t vocabMax, uint32_t vocabCap){
	uint32_t th = blockIdx.x*blockDim.x+threadIdx.x;
	Map<Val> map{key+(th*2+0)*sz, val+(th*2+0)*sz, vocabCap, vocab};
	Map<Val> map2{key+(th*2+1)*sz, val+(th*2+1)*sz, vocabCap};
	for(uint32_t i=0; i<map2.cap; i++){
		if(map2.key[i].sz <= 1) continue;
		Key& it = map(map2.key[i]);
		if(it.sz <= 1){
			it = map2.key[i];
			map[it] = map2.val[i];
			map.sz++;
			if(map.sz > vocabMax)
				reduce(dat, map, keyscore, vocab);
		}else
			map[it].cnt += map2.val[i].cnt;
	}
	reduce(dat, map, keyscore, vocab);
}
__global__ void countKer(uint8_t *dat, Key *key, Val *val, Keyscore *keyscore, uint32_t sz, uint8_t gram, uint32_t vocab, uint32_t vocabMax, uint32_t vocabCap){
	uint32_t th = blockIdx.x*blockDim.x+threadIdx.x;
	uint32_t pos = th*sz;
	printf("%d %d %d\n", th, pos, sz);
	uint8_t tmps[16], tmp;
	uint64_t hash = 0;
	for(uint8_t i=0; i<gram; i++){
		tmps[i] = tmp = dat[pos+i];
		hash ^= rotl64(tmp, i);
	}
	Map<Val> map{key+th*vocabCap, val+th*vocabCap, vocabCap};
	map.init();
	{
		Key& it = map({hash, gram});
		it = {hash, gram};
		map[it] = {pos, 1};
		map.sz = 1;
	}
	for(uint32_t i=0; i<sz; i++){
		hash ^= rotl64(tmps[i%gram], 0);
		tmps[i%gram] = tmp = dat[pos+gram+i];
		hash = rotl64(hash, 31) ^ rotl64(tmp, gram-1); //31*6=-1*6 (mod 64)
		{
			Key& it = map({hash, gram});
			if(it.sz <= 1){
				it = {hash, gram};
				map[it] = {pos+i+1, 1};
				map.sz += 1;
				if(map.sz > vocabMax)
					reduce(dat, map, keyscore, vocab);
			}else
				map[it].cnt += 1;
		}
	}
	reduce(dat, map, keyscore, vocab);
}
int main(int argc, char **argv) {
	if(argc!=8) {
		fprintf(stderr, "<gram> <vocab> <vocabMax> <vocabCap> <block> <thread> <in>\n");
		return 1;
	}
	uint8_t gram = (uint8_t)atoi(argv[1]);
	uint32_t vocab = (uint32_t)atoi(argv[2]);
	uint32_t vocabMax = (uint32_t)atoi(argv[3]);
	uint32_t vocabCap = (uint32_t)atoi(argv[4]);
	uint32_t block = (uint32_t)atoi(argv[5]);
	uint32_t thread = (uint32_t)atoi(argv[6]);
	uint32_t th = block*thread;
	int fd = open(argv[7], O_RDONLY);
	uint32_t sz = lseek(fd, 0, SEEK_END);
	uint8_t *d_dat, *dat = (uint8_t*)mmap(nullptr, sz, PROT_READ, MAP_SHARED, fd, 0);
	Key *d_key, *key;
	Val *d_val, *val;
	Keyscore *d_keyscore;
	close(fd);
	cudaMalloc(&d_dat, sz);
    cudaMalloc(&d_key, th*vocabCap*sizeof(Key));
    cudaMalloc(&d_val, th*vocabCap*sizeof(Val));
    cudaMalloc(&d_keyscore, th*vocabCap*sizeof(Keyscore));
    cudaMemcpy(d_dat, dat, sz, cudaMemcpyHostToDevice);
    countKer<<<block,thread>>>(d_dat, d_key, d_val, d_keyscore, (sz-gram+1)/th, gram, vocab, vocabMax, vocabCap);
    cudaDeviceSynchronize();
    for(int i=1; i<th; i*=2){
    	countKer<<<th,1>>>(d_dat, d_key, d_val, d_keyscore, (sz-gram+1)/th*i, gram, vocab, vocabMax, vocabCap);
	    cudaDeviceSynchronize();
    }
    key = (Key*)malloc(vocabCap*sizeof(Key));
    val = (Val*)malloc(vocabCap*sizeof(Val));
    cudaMemcpy(key, d_key, vocabCap*sizeof(Key), cudaMemcpyDeviceToHost);
    cudaMemcpy(val, d_val, vocabCap*sizeof(Val), cudaMemcpyDeviceToHost);
    for(uint32_t i=0; i<vocabCap; i++){
		if(key[i].sz <= 1) continue;
		fwrite(&key[i].sz, 1, 1, stdout);
		fwrite(dat+val[i].pos, 1, key[i].sz, stdout);
	}
	return 0;
}