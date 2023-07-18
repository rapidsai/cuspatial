#include <cuproj/projection_factories.hpp>
#include <cuprojshim.hpp>

namespace cuprojshim {

std::unique_ptr<cuproj::projection<cuproj::vec_2d<double>>>
make_projection(std::string const &src_epsg, std::string const &dst_epsg) {
  return cuproj::make_projection<cuproj::vec_2d<double>>(src_epsg, dst_epsg);
}

void transform(cuproj::projection<cuproj::vec_2d<double>> const &proj,
               cuproj::vec_2d<double> *xy_in, cuproj::vec_2d<double> *xy_out,
               std::size_t n, cuproj::direction dir) {
  proj.transform(xy_in, xy_in + n, xy_out, dir);
}

} // namespace cuprojshim
