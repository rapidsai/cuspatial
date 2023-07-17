#include <cuproj/projection.cuh>
#include <cuproj/vec_2d.cuh>

namespace cuprojshim {

void transform(cuproj::projection<cuproj::vec_2d<double>> const &proj,
               cuproj::vec_2d<double> *xy_in, cuproj::vec_2d<double> *xy_out,
               std::size_t n, cuproj::direction dir) {
  proj.transform(xy_in, xy_in + n, xy_out, dir);
}

} // namespace cuprojshim
