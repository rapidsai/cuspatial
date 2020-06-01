#include <thrust/tuple.h>

#include <cstdint>

#pragma once

namespace cuspatial {
namespace detail {

template<typename T>
using haus = thrust::tuple<thrust::tuple<int32_t, int32_t>, int64_t, int64_t, int64_t, T, T, T, T>;

// haus key
template<typename T> __device__ auto haus_key(haus<T> value) { return thrust::get<0>(value); }
template<typename T> __device__ int32_t haus_dst(haus<T> value) { return thrust::get<1>(value); }
template<typename T> __device__ auto haus_col_l(haus<T> value) { return thrust::get<2>(value); }
template<typename T> __device__ auto haus_col_r(haus<T> value) { return thrust::get<3>(value); }

template<typename T> __device__ auto haus_min_l(haus<T> value) { return thrust::get<4>(value); }
template<typename T> __device__ auto haus_min_r(haus<T> value) { return thrust::get<5>(value); }
template<typename T> __device__ auto haus_max(haus<T> value) { return thrust::get<6>(value); }

template<typename T> __device__ auto haus_res(haus<T> value) { return thrust::get<7>(value); }

template<typename T>
struct haus_key_compare
{
    bool __device__ operator()(haus<T> a, haus<T> b)
    {
        return haus_key(a) == haus_key(b);
    }
};

template<typename T>
struct haus_reduce
{
    haus<T> __device__ operator()(haus<T> const& lhs, haus<T> const& rhs)
    {
        T new_max = std::max(haus_max(lhs), haus_max(rhs));
        T new_min_l = haus_min_l(lhs);
        T new_min_r = haus_min_r(rhs);

        // if same on left, both merge left.

        auto const open_l = haus_col_l(lhs) == haus_col_r(lhs);
        auto const open_r = haus_col_l(rhs) == haus_col_r(rhs);
        auto const open_m = haus_col_r(lhs) == haus_col_l(rhs);

        auto const inner_l = haus_min_r(lhs);
        auto const inner_r = haus_min_l(rhs);

        if (open_m and not open_l and not open_r) // correct
        {
            new_max = std::max(new_max, std::min(inner_l, inner_r)); // correct
        }
        else
        {
            if (open_l) {
                new_min_l = std::min(new_min_l, inner_l); // correct
            } else if (open_m) {
                new_min_r = std::min(new_min_r, inner_l); // correct
            } else {
                new_max = std::max(new_max, inner_l); // correct
            }
            
            if (open_r) {
                new_min_r = std::min(new_min_r, inner_r); // correct
            } else if (open_m) {
                new_min_l = std::min(new_min_l, inner_r); // correct
            } else {
                new_max = std::max(new_max, inner_r); // correct
            }
        }

        auto next_open = open_m and open_l and open_r;

        auto const x = next_open ? std::min(new_min_l, new_min_r) : std::max(new_min_l, new_min_r);

        return haus<T>{
            haus_key(lhs),
            haus_dst(rhs),
            haus_col_l(lhs),
            haus_col_r(rhs),
            new_min_l,
            new_min_r,
            new_max,
            std::max(new_max, x)
        };
    }
};
    
} // namespace detail

} // namespace cuspatial
