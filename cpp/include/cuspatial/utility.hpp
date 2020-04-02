/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cusparse.h"

namespace cuspatial {
namespace detail {

} // detail namespace

// Macro block for printing cudf::columns

#define ALLOW_PRINT 0
#if ALLOW_PRINT

template<typename T>
void tPrint(thrust::detail::normal_iterator<T> start, thrust::detail::normal_iterator<T> stop, const char* name="None") {
    std::cout << name << " size: " << stop-start << " ";
    thrust::copy(start, stop, std::ostream_iterator<float>(std::cout, " "));
    std::cout << "\n";
}

template<typename T>
void tPrint(rmm::device_vector<T> vec, const char* name="None") {
  tPrint(vec.begin(), vec.end(), name);
}

void tPrint(cudf::mutable_column_view col, const char* name="None") {
  rmm::device_vector<float> vec = rmm::device_vector<float>(col.data<float>(), col.data<float>()+col.size());
  tPrint(vec.begin(), vec.end(), name);
}

void tPrint(cudf::column_view col, const char* name="None") {
  rmm::device_vector<float> vec = rmm::device_vector<float>(col.data<float>(), col.data<float>()+col.size());
  tPrint(vec.begin(), vec.end(), name);
}
#define TPRINT(vec, name) (tPrint( vec, name))

#else
#define TPRINT(vec, name) {}
#endif

} // cuspatial namespace
