# Unit Testing in libcuspatial

Unit tests in libcuspatial are written using
[Google Test](https://github.com/google/googletest/blob/master/docs/primer.md).

## Best Practices: What Should We Test?

In general we should test to make sure all code paths are covered. This is not always easy or
possible. But generally this means we test all supported combinations of algorithms and data types,
and the main iterator and container types supported by algorithms.  Here are some other guidelines.

 * Test public APIs. Try to ensure that public API tests result in 100% coverage of libcuspatial
   code (including internal details and utilities).

 * Test exceptional cases. For example, anything that causes the function to `throw`.

 * Test boundary cases. For example points that fall exactly on lines or boundaries.

 * In general empty input is not an error in libcuspatial. Typically empty input results in empty
   output. Tests should verify this.

 * Most algorithms should have one or more tests exercising inputs with a large enough number of
   rows to require launching multiple thread blocks, especially when values are ultimately
   communicated between blocks (e.g. reductions). This is especially important for custom kernels
   but also applies to Thrust and CUB algorithm calls with lambdas / functors.

## Header-only and Column-based API tests

libcuspatial currently has two C++ APIs: the column-based API uses libcudf data structures as 
input and output. These tests live in `cpp/tests/` and can use libcudf features for constructing
columns and tables. The header-only API does not depend on libcudf at all and so tests of these
APIs should not include any libcudf headers. These tests currently live in `cpp/tests/experimental`.

Generally, we test algorithms and business logic in the header-only API's unit tests. 
Column-based API tests should only cover specifics of the column-based API, such as type 
handling, input validation, and exceptions that are only thrown by that API.

## Directory and File Naming

The naming of unit test directories and source files should be consistent with the feature being
tested. For example, the tests for APIs in `point_in_polygon.hpp` should live in 
`cuspatial/cpp/tests/point_in_polygon_test.cpp`. Each feature (or set of related features) should
have its own test source file named `<feature>_test.cu/cpp`. 

In the interest of improving compile time, whenever possible, test source files should be `.cpp`
files because `nvcc` is slower than `gcc` in compiling host code. Note that `thrust::device_vector`
includes device code, and so must only be used in `.cu` files. `rmm::device_uvector`,
`rmm::device_buffer` and the various `column_wrapper` types described later can be used in `.cpp`
files, and are therefore preferred in test code over `thrust::device_vector`.

Testing header-only APIs requires CUDA compilation so should be done in `.cu` files.

## Base Fixture

All libcuspatial unit tests should make use of a GTest 
["Test Fixture"](https://github.com/google/googletest/blob/master/docs/primer.md#test-fixtures-using-the-same-data-configuration-for-multiple-tests-same-data-multiple-tests).
Even if the fixture is empty, it should inherit from the base fixture `cuspatial::test::BaseFixture`
found in `cpp/tests/base_fixture.hpp`. This ensures that RMM is properly initialized and
finalized. `cuspatial::test::BaseFixture` already inherits from `testing::Test` and therefore it is
not necessary for your test fixtures to inherit from it.

Example:

    class MyTestFixture : public cuspatial::test::BaseFixture {...};

## Typed Tests

In general, libcuspatial features must work across all supported types (for cuspatial this 
typically just means `float` and `double`). In order to automate the process of running
the same tests across multiple types, we use GTest's
[Typed Tests](https://github.com/google/googletest/blob/master/docs/advanced.md#typed-tests).
Typed tests allow you to write a test once and run it across a list of types.

For example:

```c++
// Fixture must be a template
template <typename T>
class TypedTestFixture : cuspatial::test::BaseFixture {...};
using TestTypes = ::test::types<float,double>; // Notice custom cudf type list type
TYPED_TEST_SUITE(TypedTestFixture, TestTypes);
TYPED_TEST(TypedTestFixture, FirstTest){
    // Access the current type using `TypeParam`
    using T = TypeParam;
}
```

In this example, all tests using the `TypedTestFixture` fixture will run once for each type in the
list defined in `TestTypes` (`float, double`).

## Utilities

libcuspatial test utilities include `cuspatial::test::expect_vector_equivalent()` in
`cpp/tests/utility/vector_equality()`. This function compares two containers using Google Test's 
approximate matching for floating-point values. It can handle vectors of `cuspatial::vec_2d<T>`,
where `T` is `float` or `double`. It automatically copies data in device containers to host 
containers before comparing, so you can pass it one host and one device vector, for example. 

Example:

```c++
 auto h_expected = std::vector<cuspatial::vec_2d<float>>{...}; // expected values

 auto d_actual = rmm::device_vector<cuspatial::vec_2d<float>>{...}; // actual computed values

 cuspatial::test::expect_vector_equivalent(h_expected, d_actual);
```

Before creating your own test utilities, look to see if one already exists that does
what you need. If not, consider adding a new utility to do what you need. However, make sure that
the utility is generic enough to be useful for other tests and is not overly tailored to your
specific testing need.
