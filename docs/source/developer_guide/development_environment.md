# Creating a Development Environment

Since cuspatial takes the sample pace of release as the rest of RAPIDS eco-system,
developers are encouraged to develop cuspatial with the latest development branch of RAPIDS library that cuspatial depends on.
Other cuspatial dependencies can be found in `conda/environments/`.

Maintaining the development environment can be an arduous task for developers,
especially after each rapids releases.
Most cuspatial developers today uses [rapids-compose](https://github.com/trxcllnt/rapids-compose) to setup their development environment.
It contains helpful scripts to automatcially fetch other libraries in rapids community and builds a container image,
which also builds from source for each RAPIDS repository.
`rapids-compose` is the recommended way to setup environment to develop for cuspatial.

For developers who would like to build from conda or from source,
see [README.md](https://github.com/rapidsai/cuspatial/blob/main/README.md).
