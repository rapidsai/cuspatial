#include <cuproj/projection.cuh>

#include <cuspatial/geometry/vec_2d.hpp>

namespace cuprojshim {

template <typename T> using coordinate = cuspatial::vec_2d<T>;

void transform(cuproj::projection<coordinate<double>> const &proj,
               coordinate<double> *xy_in, coordinate<double> *xy_out,
               std::size_t n, cuproj::direction dir) {
  proj.transform(xy_in, xy_in + n, xy_out, dir);
}

} // namespace cuprojshim
