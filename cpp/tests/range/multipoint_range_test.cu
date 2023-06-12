/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cuspatial_test/base_fixture.hpp>
#include <cuspatial_test/vector_equality.hpp>
#include <cuspatial_test/vector_factories.cuh>

#include <cuspatial/geometry/segment.cuh>
#include <cuspatial/geometry/vec_2d.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <thrust/sequence.h>

#include <initializer_list>

using namespace cuspatial;
using namespace cuspatial::test;

template <typename T>
class MultipointRangeTest : public BaseFixture {
 public:
  struct copy_leading_point_per_multipoint {
    template<typename MultiPointRef>
    vec_2d<T> __device__ operator()(MultiPointRef multipoint)
    {
      return multipoint.size() > 0 ? multipoint[0] : vec_2d<T>{-1, -1};
    }
  };

  template<typename MultiPointRange>
  struct point_idx_to_geometry_idx {

    MultiPointRange rng;

    std::size_t __device__ operator()(std::size_t pidx)
    {
        return rng.geometry_idx_from_point_idx(pidx);
    }
  };

  void SetUp() { make_test_multipoints(); }
  auto range() { return test_multipoints.range(); }

  virtual void make_test_multipoints() = 0;

  void run_test()
  {
    test_num_multipoints();

    test_num_points();

    test_size();

    test_multipoint_it();

    test_begin();

    test_end();

    test_point_it();

    test_offsets_it();

    test_geometry_idx_from_point_idx();

    test_subscript_operator();

    test_point();

    test_is_single_point_range();
  }

  virtual void test_num_multipoints() = 0;

  virtual void test_num_points() = 0;

  virtual void test_size() = 0;

  virtual void test_multipoint_it() = 0;

  virtual void test_begin() = 0;

  virtual void test_end() = 0;

  virtual void test_point_it() = 0;

  virtual void test_offsets_it() = 0;

  virtual void test_geometry_idx_from_point_idx() = 0;

  virtual void test_subscript_operator() = 0;

  virtual void test_point() = 0;

  virtual void test_is_single_point_range() = 0;

 protected:
  rmm::device_uvector<vec_2d<T>> copy_leading_points()
  {
    auto rng = this->range();
    rmm::device_uvector<vec_2d<T>> leading_points(rng.num_multipoints(), this->stream());
    thrust::transform(rmm::exec_policy(this->stream()),
                      rng.multipoint_begin(),
                      rng.multipoin_end(),
                      leading_points.begin(),
                      copy_leading_point_per_multipoint{});

    return leading_points;
  }

  rmm::device_uvector<vec_2d<T>> copy_all_points()
  {
    auto rng = this->range();
    rmm::device_uvector<vec_2d<T>> points(rng.num_points(), this->stream());
    thrust::copy(
      rmm::exec_policy(this->stream()), rng.point_begin(), rng.point_end(), points.begin());
    return points;
  };

  rmm::device_uvector<vec_2d<T>> copy_offsets()
  {
    auto rng = this->range();
    rmm::device_uvector<std::size_t> offsets(rng.num_multipoints()+1, this->stream());
    thrust::copy(
      rmm::exec_policy(this->stream()), rng.offset_begin(), rng.offset_end(), offsets.begin());
    return offsets;
  };

  rmm::device_uvector<std::size_t> copy_geoemtry_idx()
  {
    auto rng = this->range();
    rmm::device_uvector<std::size_t> idx(rng.num_points(), this->stream());

    thrust::tabulate(
        rmm::exec_policy(this->stream()),
        idx.begin(),
        idx.end(),
        point_idx_to_geometry_idx{rng}
    );
    return idx;
  }

  rmm::device_uvector<vec_2d<T>> copy_ith_multipoint(std::size_t i)
  {
    rmm::device_scalar<std::size_t> num_points(this->stream());

  }

  multipoint_array<T, std::size_t> test_multipoints;
};

template <typename T>
class EmptyMultiPointRangeTest : public MultipointRangeTest<T> {
 public:
  void make_test_multipoints() { this->test_multipoints = make_multipoint_array<T>({}); }

  void test_num_multipoints() { EXPECT_EQ(this->range().num_multipoints(), 0); }

  void test_num_points() { EXPECT_EQ(this->range().num_points(), 0); }

  void test_size() { EXPECT_EQ(this->range().size(), 0); }

  void test_multipoint_it()
  {
    auto leading_points = this->copy_leading_points();
    auto expected       = rmm::device_uvector<vec_2d<T>>(0, this->stream());
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(leading_points, expected);
  }

  void test_begin() { EXPECT_EQ(this->range().begin(), this->range().multipoint_begin()); }

  void test_end() { EXPECT_EQ(this->range().end(), this->range().multipoint_end()); }

  void test_point_it()
  {
    auto points   = this->copy_all_points();
    auto expected = rmm::device_uvector<vec_2d<T>>(0, this->stream());
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(points, expected);
  }

  void test_offsets_it() {
    auto offsets = this->copy_all_points();
    auto expected = make_device_vector<std::size_t>({0});
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(offsets, expected);
  }

  void test_geometry_idx_from_point_idx()
  {
    auto geometry_indices = rmm::device_uvector<std::size_t>(0, this->stream());
    auto expected = rmm::device_uvector<std::size_t>(0, this->stream()) ;

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(geometry_indices, expected);
  }

  void test_subscript_operator()
  {


  }

};

template <typename T>
class LengthOneMultiPointRangeTest : public MultipointRangeTest<T> {
 public:
  void make_test_multipoints()
  {
    this->test_multipoints = make_multipoint_array<T>({{{1.0, 1.0}}});
  }
};

template <typename T>
class LengthFiveMultiPointRangeTest : public MultipointRangeTest<T> {
 public:
  void make_test_multipoints()
  {
    this->test_multipoints = make_multipoint_array<T>({{{0.0, 0.0}, {1.0, 1.0}},
                                                       {{10.0, 10.0}},
                                                       {{20.0, 21.0}, {22.0, 23.0}},
                                                       {{30.0, 31.0}, {32.0, 33.0}, {34.0, 35.0}},
                                                       {{}}});
  }
};

template <typename T>
class LengthOneThousandRangeTest : public MultipointRangeTest<T> {
 public:
  void make_test_multipoints()
  {
    std::size_t constexpr num_multipoints          = 1000;
    std::size_t constexpr num_point_per_multipoint = 3;

    rmm::device_uvector<std::size_t> geometry_offsets(num_multipoints + 1, this->stream());
    rmm::device_uvector<vec_2d<T>> coordinates(num_multipoints * num_point_per_multipoint,
                                               this->stream());

    thrust::sequence(
      rmm::exec_policy(this->stream()), geometry_offsets.begin(), geometry_offsets.end(), 0, 3);

    thrust::generate_n(rmm::exec_policy(this->stream()),
                       geometry_offsets.begin(),
                       geometry_offsets.end(),
                       [] __device__() {
                         return vec_2d<T>{0.0, 10.0};
                       });

    this->test_multipoints =
      make_multipoint_array<T>(std::move(geometry_offsets), std::move(coordinates));
  }
};
