#pragma once

#include <thrust/functional.h>
#include <thrust/pair.h>
#include <ostream>

namespace {
const uint8_t max_warps_per_block  = 32;
const uint8_t num_threads_per_warp = 32;
const uint32_t threads_per_block   = 256;

struct get_num_units {
  const uint32_t *qt_len;
  const uint32_t size;
  get_num_units(const uint32_t *_qt_len, const uint32_t _size) : qt_len(_qt_len), size(_size) {}

  __device__ uint32_t operator()(uint32_t qid)
  {
    uint32_t len = qt_len[qid];
    uint32_t sz  = (len - 1) / size + 1;
    // uint32_t tid = threadIdx.x + blockDim.x*blockIdx.x;
    // printf("tid=%d qid=%d len=%d sz=%d\n",tid,qid,len,sz);
    return sz;
  }
};

struct gen_offset_length {
  const uint32_t *qt_length;
  const uint32_t N;

  gen_offset_length(const uint32_t _N, const uint32_t *_qt_length) : N(_N), qt_length(_qt_length) {}

  __device__ thrust::tuple<uint32_t, uint32_t> operator()(thrust::tuple<uint32_t, uint32_t> v)
  {
    uint32_t qid = thrust::get<0>(v);
    uint32_t bid = thrust::get<1>(v);
    uint32_t num = qt_length[qid];
    uint32_t len = ((bid + 1) * N > num) ? (num - bid * N) : N;
    uint32_t off = bid * N;
    // uint32_t tid = threadIdx.x + blockDim.x*blockIdx.x;
    // f("tid=%d qid=%d bid=%d num=%d len=%d off=%d\n",tid,qid,bid,num,len,off);
    return thrust::make_tuple(off, len);
  }
};

struct pq_remove_zero {
  __device__ bool operator()(thrust::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> v)
  {
    return thrust::get<4>(v) == 0;
  }
};

}  // namespace
