/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <cuspatial_test/base_fixture.hpp>
#include <cuspatial_test/vector_equality.hpp>
#include <cuspatial_test/vector_factories.cuh>

#include <cuspatial/detail/functors.cuh>
#include <cuspatial/error.hpp>
#include <cuspatial/geometry/segment.cuh>
#include <cuspatial/geometry/vec_2d.hpp>
#include <cuspatial/iterator_factory.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/sequence.h>

#include <initializer_list>

using namespace cuspatial;
using namespace cuspatial::test;

template <typename MultiPointRange, typename OutputIt>
CUSPATIAL_KERNEL void array_access_tester(MultiPointRange multipoints,
                                          std::size_t i,
                                          OutputIt output_points)
{
  using T = typename MultiPointRange::element_t;
  thrust::copy(thrust::seq, multipoints[i].begin(), multipoints[i].end(), output_points);
}

template <typename MultiPointRange, typename OutputIt>
CUSPATIAL_KERNEL void point_accessor_tester(MultiPointRange multipoints,
                                            std::size_t i,
                                            OutputIt point)
{
  using T  = typename MultiPointRange::element_t;
  point[0] = multipoints.point(i);
}

template <typename T>
class MultipointRangeTest : public BaseFixture {
 public:
  struct copy_leading_point_per_multipoint {
    template <typename MultiPointRef>
    vec_2d<T> __device__ operator()(MultiPointRef multipoint)
    {
      return multipoint.size() > 0 ? multipoint[0] : vec_2d<T>{-1, -1};
    }
  };

  template <typename MultiPointRange>
  struct point_idx_to_geometry_idx {
    MultiPointRange rng;

    point_idx_to_geometry_idx(MultiPointRange r) : rng(r) {}

    std::size_t __device__ operator()(std::size_t pidx)
    {
      return rng.geometry_idx_from_point_idx(pidx);
    }
  };

  void SetUp() { make_test_multipoints(); }
  auto range() { return test_multipoints->range(); }

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

    test_point_accessor();

    test_is_single_point_range();
  }

  virtual void test_num_multipoints() = 0;

  virtual void test_num_points() = 0;

  void test_size() { EXPECT_EQ(this->range().size(), this->range().num_multipoints()); }

  virtual void test_multipoint_it() = 0;

  void test_begin() { EXPECT_EQ(this->range().begin(), this->range().multipoint_begin()); }

  void test_end() { EXPECT_EQ(this->range().end(), this->range().multipoint_end()); }

  virtual void test_point_it() = 0;

  virtual void test_offsets_it() = 0;

  virtual void test_geometry_idx_from_point_idx() = 0;

  virtual void test_subscript_operator() = 0;

  virtual void test_point_accessor() = 0;

  virtual void test_is_single_point_range() = 0;

 protected:
  rmm::device_uvector<vec_2d<T>> copy_leading_points()
  {
    auto rng = this->range();
    rmm::device_uvector<vec_2d<T>> leading_points(rng.num_multipoints(), this->stream());
    thrust::transform(rmm::exec_policy(this->stream()),
                      rng.multipoint_begin(),
                      rng.multipoint_end(),
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

  rmm::device_uvector<std::size_t> copy_offsets()
  {
    auto rng = this->range();
    rmm::device_uvector<std::size_t> offsets(rng.num_multipoints() + 1, this->stream());
    thrust::copy(
      rmm::exec_policy(this->stream()), rng.offsets_begin(), rng.offsets_end(), offsets.begin());
    return offsets;
  };

  rmm::device_uvector<std::size_t> copy_geometry_idx()
  {
    auto rng = this->range();
    rmm::device_uvector<std::size_t> idx(rng.num_points(), this->stream());

    thrust::tabulate(
      rmm::exec_policy(this->stream()), idx.begin(), idx.end(), point_idx_to_geometry_idx{rng});
    return idx;
  }

  rmm::device_scalar<vec_2d<T>> copy_ith_point(std::size_t i)
  {
    auto rng = this->range();

    rmm::device_scalar<vec_2d<T>> point(this->stream());
    point_accessor_tester<<<1, 1, 0, this->stream()>>>(rng, i, point.data());
    CUSPATIAL_CHECK_CUDA(this->stream());

    return point;
  }

  rmm::device_uvector<vec_2d<T>> copy_ith_multipoint(std::size_t i)
  {
    auto rng = this->range();
    rmm::device_scalar<std::size_t> num_points(this->stream());
    auto count_iterator = make_count_iterator_from_offset_iterator(this->range().offsets_begin());
    thrust::copy_n(rmm::exec_policy(this->stream()), count_iterator, 1, num_points.data());

    rmm::device_uvector<vec_2d<T>> multipoint(num_points.value(this->stream()), this->stream());
    array_access_tester<<<1, 1, 0, this->stream()>>>(rng, i, multipoint.begin());
    CUSPATIAL_CHECK_CUDA(this->stream());

    return multipoint;
  }

  std::unique_ptr<multipoint_array<rmm::device_vector<std::size_t>, rmm::device_vector<vec_2d<T>>>>
    test_multipoints;
};

template <typename T>
class EmptyMultiPointRangeTest : public MultipointRangeTest<T> {
 public:
  void make_test_multipoints()
  {
    auto array             = make_multipoint_array<T>({});
    this->test_multipoints = std::make_unique<decltype(array)>(std::move(array));
  }

  void test_num_multipoints() { EXPECT_EQ(this->range().num_multipoints(), 0); }

  void test_num_points() { EXPECT_EQ(this->range().num_points(), 0); }

  void test_multipoint_it()
  {
    auto leading_points = this->copy_leading_points();
    auto expected       = rmm::device_uvector<vec_2d<T>>(0, this->stream());
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(leading_points, expected);
  }

  void test_point_it()
  {
    auto points   = this->copy_all_points();
    auto expected = rmm::device_uvector<vec_2d<T>>(0, this->stream());
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(points, expected);
  }

  void test_offsets_it()
  {
    auto offsets  = this->copy_offsets();
    auto expected = make_device_vector<std::size_t>({0});
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(offsets, expected);
  }

  void test_geometry_idx_from_point_idx()
  {
    auto geometry_indices = this->copy_geometry_idx();
    auto expected         = rmm::device_uvector<std::size_t>(0, this->stream());

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(geometry_indices, expected);
  }

  void test_subscript_operator()
  {
    // Range is empty, nothing to test.
    SUCCEED();
  }

  void test_point_accessor()
  {
    // Range is empty, nothing to test.
    SUCCEED();
  }

  void test_is_single_point_range()
  {
    // Range is empty, undefined behavior.
    SUCCEED();
  }
};

TYPED_TEST_CASE(EmptyMultiPointRangeTest, FloatingPointTypes);
TYPED_TEST(EmptyMultiPointRangeTest, Test) { this->run_test(); }

template <typename T>
class LengthOneMultiPointRangeTest : public MultipointRangeTest<T> {
 public:
  void make_test_multipoints()
  {
    auto array             = make_multipoint_array<T>({{{1.0, 1.0}, {10.0, 10.0}}});
    this->test_multipoints = std::make_unique<decltype(array)>(std::move(array));
  }

  void test_num_multipoints() { EXPECT_EQ(this->range().num_multipoints(), 1); }

  void test_num_points() { EXPECT_EQ(this->range().num_points(), 2); }

  void test_multipoint_it()
  {
    auto leading_points = this->copy_leading_points();
    auto expected       = make_device_vector<vec_2d<T>>({{1.0, 1.0}});
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(leading_points, expected);
  }

  void test_point_it()
  {
    auto points   = this->copy_all_points();
    auto expected = make_device_vector<vec_2d<T>>({{1.0, 1.0}, {10.0, 10.0}});
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(points, expected);
  }

  void test_offsets_it()
  {
    auto offsets  = this->copy_offsets();
    auto expected = make_device_vector<std::size_t>({0, 2});
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(offsets, expected);
  }

  void test_geometry_idx_from_point_idx()
  {
    auto geometry_indices = this->copy_geometry_idx();
    auto expected         = make_device_vector<std::size_t>({0, 0});

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(geometry_indices, expected);
  }

  void test_subscript_operator()
  {
    auto multipoint = this->copy_ith_multipoint(0);
    auto expected   = make_device_vector<vec_2d<T>>({{1.0, 1.0}, {10.0, 10.0}});

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(multipoint, expected);
  }

  void test_point_accessor()
  {
    auto point    = this->copy_ith_point(1);
    auto expected = vec_2d<T>{10.0, 10.0};
    EXPECT_EQ(point.value(this->stream()), expected);
  }

  void test_is_single_point_range() { EXPECT_FALSE(this->range().is_single_point_range()); }
};

TYPED_TEST_CASE(LengthOneMultiPointRangeTest, FloatingPointTypes);
TYPED_TEST(LengthOneMultiPointRangeTest, Test) { this->run_test(); }

template <typename T>
class LengthFiveMultiPointRangeTest : public MultipointRangeTest<T> {
 public:
  void make_test_multipoints()
  {
    auto array             = make_multipoint_array<T>({{{0.0, 0.0}, {1.0, 1.0}},
                                                       {{10.0, 10.0}},
                                                       {{20.0, 21.0}, {22.0, 23.0}},
                                                       {{30.0, 31.0}, {32.0, 33.0}, {34.0, 35.0}},
                                                       {}});
    this->test_multipoints = std::make_unique<decltype(array)>(std::move(array));
  }

  void test_num_multipoints() { EXPECT_EQ(this->range().num_multipoints(), 5); }

  void test_num_points() { EXPECT_EQ(this->range().num_points(), 8); }

  void test_multipoint_it()
  {
    auto leading_points = this->copy_leading_points();
    auto expected       = make_device_vector<vec_2d<T>>(
      {{0.0, 0.0}, {10.0, 10.0}, {20.0, 21.0}, {30.0, 31.0}, {-1, -1}});
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(leading_points, expected);
  }

  void test_point_it()
  {
    auto points   = this->copy_all_points();
    auto expected = make_device_vector<vec_2d<T>>({{0.0, 0.0},
                                                   {1.0, 1.0},
                                                   {10.0, 10.0},
                                                   {20.0, 21.0},
                                                   {22.0, 23.0},
                                                   {30.0, 31.0},
                                                   {32.0, 33.0},
                                                   {34.0, 35.0}});
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(points, expected);
  }

  void test_offsets_it()
  {
    auto offsets  = this->copy_offsets();
    auto expected = make_device_vector<std::size_t>({0, 2, 3, 5, 8, 8});
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(offsets, expected);
  }

  void test_geometry_idx_from_point_idx()
  {
    auto geometry_indices = this->copy_geometry_idx();
    auto expected         = make_device_vector<std::size_t>({0, 0, 1, 2, 2, 3, 3, 3});
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(geometry_indices, expected);
  }

  void test_subscript_operator()
  {
    auto second_multipoint = this->copy_ith_multipoint(2);
    auto expected          = make_device_vector<vec_2d<T>>({{20.0, 21.0}, {22.0, 23.0}});
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(second_multipoint, expected);
  }

  void test_point_accessor()
  {
    auto third_point = this->copy_ith_point(3);
    auto expected    = vec_2d<T>{20.0, 21.0};
    EXPECT_EQ(third_point.value(this->stream()), expected);
  }

  void test_is_single_point_range() { EXPECT_FALSE(this->range().is_single_point_range()); }
};

TYPED_TEST_CASE(LengthFiveMultiPointRangeTest, FloatingPointTypes);
TYPED_TEST(LengthFiveMultiPointRangeTest, Test) { this->run_test(); }

template <typename T>
class LengthOneThousandRangeTest : public MultipointRangeTest<T> {
 public:
  std::size_t static constexpr num_multipoints          = 1000;
  std::size_t static constexpr num_point_per_multipoint = 3;
  std::size_t static constexpr num_points = num_multipoints * num_point_per_multipoint;
  void make_test_multipoints()
  {
    rmm::device_vector<std::size_t> geometry_offsets(num_multipoints + 1);
    rmm::device_vector<vec_2d<T>> coordinates(num_points);

    thrust::sequence(rmm::exec_policy(this->stream()),
                     geometry_offsets.begin(),
                     geometry_offsets.end(),
                     0ul,
                     num_point_per_multipoint);

    thrust::tabulate(rmm::exec_policy(this->stream()),
                     coordinates.begin(),
                     coordinates.end(),
                     [] __device__(auto i) {
                       return vec_2d<T>{static_cast<T>(i), 10.0};
                     });

    auto array =
      make_multipoint_array<std::size_t, T>(std::move(geometry_offsets), std::move(coordinates));

    this->test_multipoints = std::make_unique<decltype(array)>(std::move(array));
  }

  void test_num_multipoints() { EXPECT_EQ(this->range().num_multipoints(), num_multipoints); }

  void test_num_points() { EXPECT_EQ(this->range().num_points(), num_points); }

  void test_multipoint_it()
  {
    auto leading_points = this->copy_leading_points();
    auto expect         = rmm::device_uvector<vec_2d<T>>(num_multipoints, this->stream());

    thrust::tabulate(
      rmm::exec_policy(this->stream()), expect.begin(), expect.end(), [] __device__(auto i) {
        return vec_2d<T>{static_cast<T>(i) * num_point_per_multipoint, 10.0};
      });

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(leading_points, expect);
  }

  void test_point_it()
  {
    auto all_points = this->copy_all_points();
    auto expect     = rmm::device_uvector<vec_2d<T>>(num_points, this->stream());

    thrust::tabulate(
      rmm::exec_policy(this->stream()), expect.begin(), expect.end(), [] __device__(auto i) {
        return vec_2d<T>{static_cast<T>(i), 10.0};
      });

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(all_points, expect);
  }

  void test_offsets_it()
  {
    auto offsets = this->copy_offsets();
    auto expect  = rmm::device_uvector<std::size_t>(num_multipoints + 1, this->stream());
    thrust::sequence(rmm::exec_policy(this->stream()),
                     expect.begin(),
                     expect.end(),
                     0ul,
                     num_point_per_multipoint);

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(offsets, expect);
  }

  void test_geometry_idx_from_point_idx()
  {
    auto indices = this->copy_geometry_idx();
    auto expect  = rmm::device_uvector<std::size_t>(3000, this->stream());
    thrust::tabulate(
      rmm::exec_policy(this->stream()), expect.begin(), expect.end(), [] __device__(auto i) {
        return i / num_point_per_multipoint;
      });

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(indices, expect);
  }

  void test_subscript_operator()
  {
    auto multipoint_five_hundred_thirty_third = this->copy_ith_multipoint(533);
    auto expect = make_device_vector<vec_2d<T>>({{533 * num_point_per_multipoint, 10.0},
                                                 {533 * num_point_per_multipoint + 1, 10.0},
                                                 {533 * num_point_per_multipoint + 2, 10.0}});

    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(multipoint_five_hundred_thirty_third, expect);
  }

  void test_point_accessor()
  {
    auto point_seventeen_hundred_seventy_six = this->copy_ith_point(1776);
    auto expect                              = vec_2d<T>{1776, 10.0};

    EXPECT_EQ(point_seventeen_hundred_seventy_six.value(this->stream()), expect);
  }

  void test_is_single_point_range() { EXPECT_FALSE(this->range().is_single_point_range()); }
};

TYPED_TEST_CASE(LengthOneThousandRangeTest, FloatingPointTypes);
TYPED_TEST(LengthOneThousandRangeTest, Test) { this->run_test(); }
