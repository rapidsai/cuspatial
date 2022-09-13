# Creating a Development Environment

cuSpatial follows the RAPIDS release schedule, so developers are encouraged to develop 
using the latest development branch of RAPIDS libraries that cuspatial depends on. Other
cuspatial dependencies can be found in `conda/environments/`.

Maintaining a local development environment can be an arduous task, especially after each
RAPIDS release. Most cuspatial developers today use
[rapids-compose](https://github.com/trxcllnt/rapids-compose) to setup their development environment.
It contains helpful scripts to build a RAPIDS development container image with the required
dependencies and RAPIDS libraries automatically fetched and correctly versioned. It also provides
script commands for simple building and testing of all RAPIDS libraries, including cuSpatial.
`rapids-compose` is the recommended way to set up your environment to develop for cuspatial.

For developers who would like to build from conda or from source, see
[README.md](https://github.com/rapidsai/cuspatial/blob/main/README.md).
