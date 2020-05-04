/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <gtest/gtest.h>
#include <cuspatial/point_in_polygon.hpp>
#include <string>
#include <vector>
#include "pip_util.h"

#include <tests/utilities/legacy/cudf_test_fixtures.h>
#include <tests/utilities/legacy/column_wrapper.cuh>
#include <tests/utilities/legacy/cudf_test_utils.cuh>

struct PIPToy : public GdfTest {
  int point_len = 3;

  double *x = new double[point_len]{0, -8, 6};
  double *y = new double[point_len]{0, -8, 6};

  cuspatial::polygons<double> h_polygon;

  int set_initialize()
  {
    h_polygon.num_group        = 1;
    h_polygon.num_feature      = 2;
    h_polygon.num_ring         = 2;
    h_polygon.num_vertex       = 10;
    h_polygon.feature_position = new uint32_t[h_polygon.num_feature]{1, 2};
    h_polygon.ring_position    = new uint32_t[h_polygon.num_ring]{5, 10};
    h_polygon.x = new double[h_polygon.num_vertex]{-10, 5, 5, -10, -10, 0, 10, 10, 0, 0};
    h_polygon.y = new double[h_polygon.num_vertex]{-10, -10, 5, 5, -10, 0, 0, 10, 10, 0};

    return 1;
  }

  std::vector<uint32_t> exec_gpu_pip()
  {
    // std::vector g_pos_v(h_polygon.group_position,h_polygon.group_position+h_polygon.num_group);
    std::vector<uint32_t> f_pos_v(h_polygon.feature_position,
                                  h_polygon.feature_position + h_polygon.num_feature);
    std::vector<uint32_t> r_pos_v(h_polygon.ring_position,
                                  h_polygon.ring_position + h_polygon.num_ring);
    std::vector<double> ply_x_v(h_polygon.x, h_polygon.x + h_polygon.num_vertex);
    std::vector<double> ply_y_v(h_polygon.y, h_polygon.y + h_polygon.num_vertex);
    std::vector<double> pnt_x_v(x, x + this->point_len);
    std::vector<double> pnt_y_v(y, y + this->point_len);

    cudf::test::column_wrapper<uint32_t> polygon_fpos_wrapp{f_pos_v};
    cudf::test::column_wrapper<uint32_t> polygon_rpos_wrapp{r_pos_v};
    cudf::test::column_wrapper<double> polygon_x_wrapp{ply_x_v};
    cudf::test::column_wrapper<double> polygon_y_wrapp{ply_y_v};
    cudf::test::column_wrapper<double> point_x_wrapp{pnt_x_v};
    cudf::test::column_wrapper<double> point_y_wrapp{pnt_y_v};

    gdf_column res_bm1 = cuspatial::point_in_polygon_bitmap(*(point_x_wrapp.get()),
                                                            *(point_y_wrapp.get()),
                                                            *(polygon_fpos_wrapp.get()),
                                                            *(polygon_rpos_wrapp.get()),
                                                            *(polygon_x_wrapp.get()),
                                                            *(polygon_y_wrapp.get()));

    std::vector<uint32_t> gpu_pip_res(this->point_len);
    EXPECT_EQ(cudaMemcpy(gpu_pip_res.data(),
                         res_bm1.data,
                         this->point_len * sizeof(uint32_t),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    return gpu_pip_res;
  }

  void set_finalize()
  {
    delete[] h_polygon.group_length;
    delete[] h_polygon.feature_length;
    delete[] h_polygon.ring_length;
    if (!h_polygon.is_inplace) {
      delete[] h_polygon.group_position;
      delete[] h_polygon.feature_position;
      delete[] h_polygon.ring_position;
    }
    delete[] h_polygon.x;
    delete[] h_polygon.y;

    delete[] x;
    delete[] y;
  }
};

TEST_F(PIPToy, piptest)
{
  ASSERT_GE(this->set_initialize(), 0);

  std::vector<uint32_t> cpu_pip_res =
    cpu_pip_loop(this->point_len, this->x, this->y, this->h_polygon);

  std::vector<uint32_t> gpu_pip_res = this->exec_gpu_pip();
  EXPECT_THAT(gpu_pip_res, testing::Eq(cpu_pip_res));

  this->set_finalize();
}
