#include <cuproj/projection.cuh>
#include <cuproj/vec_2d.hpp>

namespace cuprojshim {

std::unique_ptr<cuproj::projection<cuproj::vec_2d<double>>>
make_projection(std::string const &src_epsg, std::string const &dst_epsg);

void transform(cuproj::projection<cuproj::vec_2d<double>> const &proj,
               cuproj::vec_2d<double> *xy_in, cuproj::vec_2d<double> *xy_out,
               std::size_t n, cuproj::direction dir);

} // namespace cuprojshim
