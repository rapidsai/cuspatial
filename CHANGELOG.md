# cuSpatial 23.02.00 (Date TBD)

Please see https://github.com/rapidsai/cuspatial/releases/tag/v23.02.00a for the latest changes to this development branch.

# cuSpatial 22.12.00 (8 Dec 2022)

## 🚨 Breaking Changes

- Update quadtree spatial join docstrings ([#797](https://github.com/rapidsai/cuspatial/pull/797)) [@trxcllnt](https://github.com/trxcllnt)
- Rename lonlat_to_cartesian to sinusoidal_projection ([#794](https://github.com/rapidsai/cuspatial/pull/794)) [@harrism](https://github.com/harrism)
- Consolidate bounding box code ([#793](https://github.com/rapidsai/cuspatial/pull/793)) [@harrism](https://github.com/harrism)
- Globally replace &quot;polyline&quot; with &quot;linestring&quot; ([#788](https://github.com/rapidsai/cuspatial/pull/788)) [@harrism](https://github.com/harrism)
- Refactor of `pairwise_linestring_distance` to use `multilinestring_range`, adds support to multilinestring distance ([#755](https://github.com/rapidsai/cuspatial/pull/755)) [@isVoid](https://github.com/isVoid)
- Introduce `multilinestring_range` structure, simplifies point-linestring distance API ([#747](https://github.com/rapidsai/cuspatial/pull/747)) [@isVoid](https://github.com/isVoid)
- Add python bindings for (multi)point-(multi)point distance ([#734](https://github.com/rapidsai/cuspatial/pull/734)) [@isVoid](https://github.com/isVoid)
- Introduce `multipoint_range` interface; Refactors `point_distance` API to support multipoint to multipoint distance. ([#731](https://github.com/rapidsai/cuspatial/pull/731)) [@isVoid](https://github.com/isVoid)

## 🐛 Bug Fixes

- Add Floating Point Equality Tests to Build List ([#812](https://github.com/rapidsai/cuspatial/pull/812)) [@isVoid](https://github.com/isVoid)
- Fix a Bug in Segment Intersection Primitive ([#808](https://github.com/rapidsai/cuspatial/pull/808)) [@isVoid](https://github.com/isVoid)
- Don&#39;t use CMake 3.25.0 as it has a FindCUDAToolkit show stopping bug ([#805](https://github.com/rapidsai/cuspatial/pull/805)) [@robertmaynard](https://github.com/robertmaynard)
- Fix style checks. ([#791](https://github.com/rapidsai/cuspatial/pull/791)) [@bdice](https://github.com/bdice)
- Force using old fmt in nvbench. ([#783](https://github.com/rapidsai/cuspatial/pull/783)) [@isVoid](https://github.com/isVoid)
- Fix issue with `.loc` returning values out of order. ([#782](https://github.com/rapidsai/cuspatial/pull/782)) [@thomcom](https://github.com/thomcom)
- address issue with vector equivalent utility ([#777](https://github.com/rapidsai/cuspatial/pull/777)) [@isVoid](https://github.com/isVoid)

## 📖 Documentation

- Add symlink to `users.ipynb` for notebooks CI ([#790](https://github.com/rapidsai/cuspatial/pull/790)) [@thomcom](https://github.com/thomcom)
- Fix failed automerge (branch 22.12 merge 22.10) ([#740](https://github.com/rapidsai/cuspatial/pull/740)) [@harrism](https://github.com/harrism)

## 🚀 New Features

- Add `find_duplicate_points` Internal API ([#815](https://github.com/rapidsai/cuspatial/pull/815)) [@isVoid](https://github.com/isVoid)
- Add Internal Function `intersection_count_upper_bound` ([#795](https://github.com/rapidsai/cuspatial/pull/795)) [@isVoid](https://github.com/isVoid)
- Support `Multilinestring` in column API and python Bindings for `pairwise_linestring_distance` ([#786](https://github.com/rapidsai/cuspatial/pull/786)) [@isVoid](https://github.com/isVoid)
- Introduce Segment Intersection Primitive ([#778](https://github.com/rapidsai/cuspatial/pull/778)) [@isVoid](https://github.com/isVoid)
- Introduce ULP Based Floating Point Equality Test to Device Function ([#773](https://github.com/rapidsai/cuspatial/pull/773)) [@isVoid](https://github.com/isVoid)
- Augment Cuspatial Test Utility to Allow User Specified Abs Error ([#752](https://github.com/rapidsai/cuspatial/pull/752)) [@isVoid](https://github.com/isVoid)
- Create `pairwise_point_in_polygon` to be used by pairwise `GeoSeries` ([#750](https://github.com/rapidsai/cuspatial/pull/750)) [@thomcom](https://github.com/thomcom)
- Replacing markdown issue templates with yml forms ([#748](https://github.com/rapidsai/cuspatial/pull/748)) [@jarmak-nv](https://github.com/jarmak-nv)
- Introduce `multilinestring_range` structure, simplifies point-linestring distance API ([#747](https://github.com/rapidsai/cuspatial/pull/747)) [@isVoid](https://github.com/isVoid)
- Add python bindings for (multi)point-(multi)point distance ([#734](https://github.com/rapidsai/cuspatial/pull/734)) [@isVoid](https://github.com/isVoid)
- Introduce `multipoint_range` interface; Refactors `point_distance` API to support multipoint to multipoint distance. ([#731](https://github.com/rapidsai/cuspatial/pull/731)) [@isVoid](https://github.com/isVoid)

## 🛠️ Improvements

- Set max version pin for `gdal` ([#806](https://github.com/rapidsai/cuspatial/pull/806)) [@ajschmidt8](https://github.com/ajschmidt8)
- Update Dependency List with `dependencies.yaml` ([#803](https://github.com/rapidsai/cuspatial/pull/803)) [@isVoid](https://github.com/isVoid)
- Labeler: Change cpp label to libcuspatial ([#800](https://github.com/rapidsai/cuspatial/pull/800)) [@jarmak-nv](https://github.com/jarmak-nv)
- Update pre-commit configuration and CI ([#799](https://github.com/rapidsai/cuspatial/pull/799)) [@bdice](https://github.com/bdice)
- Update quadtree spatial join docstrings ([#797](https://github.com/rapidsai/cuspatial/pull/797)) [@trxcllnt](https://github.com/trxcllnt)
- Rename lonlat_to_cartesian to sinusoidal_projection ([#794](https://github.com/rapidsai/cuspatial/pull/794)) [@harrism](https://github.com/harrism)
- Consolidate bounding box code ([#793](https://github.com/rapidsai/cuspatial/pull/793)) [@harrism](https://github.com/harrism)
- Globally replace &quot;polyline&quot; with &quot;linestring&quot; ([#788](https://github.com/rapidsai/cuspatial/pull/788)) [@harrism](https://github.com/harrism)
- Fix conda channel order ([#787](https://github.com/rapidsai/cuspatial/pull/787)) [@harrism](https://github.com/harrism)
- Delete deprecated board GitHub Actions ([#779](https://github.com/rapidsai/cuspatial/pull/779)) [@jarmak-nv](https://github.com/jarmak-nv)
- Better slicing via `union_offsets` ([#776](https://github.com/rapidsai/cuspatial/pull/776)) [@thomcom](https://github.com/thomcom)
- Header-only refactoring of trajectory_distances_and_speeds ([#769](https://github.com/rapidsai/cuspatial/pull/769)) [@harrism](https://github.com/harrism)
- Allow None rows in `GeoSeries` and an `align` method to match `GeoPandas` ([#760](https://github.com/rapidsai/cuspatial/pull/760)) [@thomcom](https://github.com/thomcom)
- Refactor of `pairwise_linestring_distance` to use `multilinestring_range`, adds support to multilinestring distance ([#755](https://github.com/rapidsai/cuspatial/pull/755)) [@isVoid](https://github.com/isVoid)
- Remove stale labeler ([#751](https://github.com/rapidsai/cuspatial/pull/751)) [@raydouglass](https://github.com/raydouglass)
- Create `GeoSeries.contains_properly` method using point_in_polygon. ([#749](https://github.com/rapidsai/cuspatial/pull/749)) [@thomcom](https://github.com/thomcom)
- Header-only trajectory_bounding_boxes ([#741](https://github.com/rapidsai/cuspatial/pull/741)) [@harrism](https://github.com/harrism)
- Update Conda Recipe and `README.md` ([#730](https://github.com/rapidsai/cuspatial/pull/730)) [@isVoid](https://github.com/isVoid)
- Remove `cudf/cuda.cuh` Dependency ([#729](https://github.com/rapidsai/cuspatial/pull/729)) [@isVoid](https://github.com/isVoid)

# cuSpatial 22.10.00 (12 Oct 2022)

## 🚨 Breaking Changes

- Remove lonlat_2d and cartesian_2d types ([#662](https://github.com/rapidsai/cuspatial/pull/662)) [@harrism](https://github.com/harrism)
- Add Multi-Geometry support to `point_linestring_distance` and build python bindings ([#660](https://github.com/rapidsai/cuspatial/pull/660)) [@isVoid](https://github.com/isVoid)
- Decouple `interpolate` functions from trajectory ([#645](https://github.com/rapidsai/cuspatial/pull/645)) [@isVoid](https://github.com/isVoid)

## 🐛 Bug Fixes

- Fix error in users.ipynb ([#726](https://github.com/rapidsai/cuspatial/pull/726)) [@thomcom](https://github.com/thomcom)
- `unwrap_pyoptcol` is missing `except` keyword that causes exceptions ignored, fixes function bug ([#719](https://github.com/rapidsai/cuspatial/pull/719)) [@isVoid](https://github.com/isVoid)
- Fix all unexecutable code examples I can find. ([#693](https://github.com/rapidsai/cuspatial/pull/693)) [@thomcom](https://github.com/thomcom)
- Add Union-style indexing to `.points`, `.multipoints`, `.lines`, and `.polygons` `GeoSeries` accessors ([#685](https://github.com/rapidsai/cuspatial/pull/685)) [@thomcom](https://github.com/thomcom)
- Rewrite `copy_type_metadata` to reuse the inherited method and handle geocolumns specifically. ([#684](https://github.com/rapidsai/cuspatial/pull/684)) [@isVoid](https://github.com/isVoid)
- Fix `GeoDataframe` slicing issue by adding `_slice_` method. ([#680](https://github.com/rapidsai/cuspatial/pull/680)) [@thomcom](https://github.com/thomcom)
- Fix and tidy coordinate_transform_test ([#671](https://github.com/rapidsai/cuspatial/pull/671)) [@harrism](https://github.com/harrism)
- `linestring_distance` Header Only API Returns Past-the-End iterator ([#670](https://github.com/rapidsai/cuspatial/pull/670)) [@isVoid](https://github.com/isVoid)

## 📖 Documentation

- Update link to c++ developer guide ([#718](https://github.com/rapidsai/cuspatial/pull/718)) [@isVoid](https://github.com/isVoid)
- Add C++ doc links to `library_design.md` and minor documentation fixes ([#700](https://github.com/rapidsai/cuspatial/pull/700)) [@isVoid](https://github.com/isVoid)
- Document that minimum required CMake version is now 3.23.1 ([#689](https://github.com/rapidsai/cuspatial/pull/689)) [@robertmaynard](https://github.com/robertmaynard)
- Publish Developer Guide ([#673](https://github.com/rapidsai/cuspatial/pull/673)) [@harrism](https://github.com/harrism)
- Add TESTING.md and BENCHMARKING.md ([#672](https://github.com/rapidsai/cuspatial/pull/672)) [@harrism](https://github.com/harrism)
- Add DOCUMENTATION.md ([#667](https://github.com/rapidsai/cuspatial/pull/667)) [@harrism](https://github.com/harrism)
- Branch 22.10 merge 22.08 ([#654](https://github.com/rapidsai/cuspatial/pull/654)) [@harrism](https://github.com/harrism)
- Add Developer Guides, replace internal.md, CONTRIBUTING.md ([#625](https://github.com/rapidsai/cuspatial/pull/625)) [@isVoid](https://github.com/isVoid)
- Add Markdown Parser to Sphinx ([#621](https://github.com/rapidsai/cuspatial/pull/621)) [@isVoid](https://github.com/isVoid)
- Simplify PR template ([#617](https://github.com/rapidsai/cuspatial/pull/617)) [@harrism](https://github.com/harrism)
- Add libcuspatial C++ developer guide. ([#606](https://github.com/rapidsai/cuspatial/pull/606)) [@harrism](https://github.com/harrism)

## 🚀 New Features

- Initialize a `GeoDataFrame` with `dict`. ([#712](https://github.com/rapidsai/cuspatial/pull/712)) [@thomcom](https://github.com/thomcom)
- Vectorized Load, refactors `type_utils.hpp` into `iterator_factory.cuh` ([#692](https://github.com/rapidsai/cuspatial/pull/692)) [@isVoid](https://github.com/isVoid)
- Accept `None` and python list in `GeoSeries` constructor ([#686](https://github.com/rapidsai/cuspatial/pull/686)) [@isVoid](https://github.com/isVoid)
- Python API for point-linestring nearest points ([#681](https://github.com/rapidsai/cuspatial/pull/681)) [@isVoid](https://github.com/isVoid)
- cuSpatial Python User Guide ([#666](https://github.com/rapidsai/cuspatial/pull/666)) [@thomcom](https://github.com/thomcom)
- Add Multi-Geometry support to `point_linestring_distance` and build python bindings ([#660](https://github.com/rapidsai/cuspatial/pull/660)) [@isVoid](https://github.com/isVoid)
- Add C++ API for `point_linestring_nearest_points` ([#658](https://github.com/rapidsai/cuspatial/pull/658)) [@isVoid](https://github.com/isVoid)
- Auto-add new Issues and PRs to cuspatial&#39;s project ([#618](https://github.com/rapidsai/cuspatial/pull/618)) [@jarmak-nv](https://github.com/jarmak-nv)
- Integrate `GeoSeries` with `read_polygon_shapefile` ([#609](https://github.com/rapidsai/cuspatial/pull/609)) [@thomcom](https://github.com/thomcom)
- Memory_usage method for GeoSeries/GeoDataFrame and `GeoDataFrame` refactor. ([#604](https://github.com/rapidsai/cuspatial/pull/604)) [@thomcom](https://github.com/thomcom)
- Add Point Linestring Distance ([#573](https://github.com/rapidsai/cuspatial/pull/573)) [@isVoid](https://github.com/isVoid)

## 🛠️ Improvements

- Update to the latest version 3 of GDAL. ([#675](https://github.com/rapidsai/cuspatial/pull/675)) [@thomcom](https://github.com/thomcom)
- Expand expect_vector_equivalent to handle std::vector of vec_2d&lt;T&gt; and move traits out of detail ([#669](https://github.com/rapidsai/cuspatial/pull/669)) [@harrism](https://github.com/harrism)
- Statically link all CUDA toolkit libraries ([#665](https://github.com/rapidsai/cuspatial/pull/665)) [@trxcllnt](https://github.com/trxcllnt)
- Remove lonlat_2d and cartesian_2d types ([#662](https://github.com/rapidsai/cuspatial/pull/662)) [@harrism](https://github.com/harrism)
- Rename Test Folders to Align with Module Names ([#661](https://github.com/rapidsai/cuspatial/pull/661)) [@isVoid](https://github.com/isVoid)
- Move `GeoSeries` `GeoDataframe` into `core` package and hide column implementation in internal `_column` package ([#657](https://github.com/rapidsai/cuspatial/pull/657)) [@isVoid](https://github.com/isVoid)
- Refactor spatial related functions under `spatial` package ([#656](https://github.com/rapidsai/cuspatial/pull/656)) [@isVoid](https://github.com/isVoid)
- Add Missing Thrust Headers for Thrust 1.17 ([#655](https://github.com/rapidsai/cuspatial/pull/655)) [@isVoid](https://github.com/isVoid)
- Decouple `interpolate` functions from trajectory ([#645](https://github.com/rapidsai/cuspatial/pull/645)) [@isVoid](https://github.com/isVoid)
- Add header only `cuspatial::quadtree_on_points` ([#639](https://github.com/rapidsai/cuspatial/pull/639)) [@trxcllnt](https://github.com/trxcllnt)
- Header-only refactoring of derive_trajectories ([#628](https://github.com/rapidsai/cuspatial/pull/628)) [@harrism](https://github.com/harrism)
- Add python benchmarks. ([#600](https://github.com/rapidsai/cuspatial/pull/600)) [@thomcom](https://github.com/thomcom)
- Fix compile error in distance benchmark ([#596](https://github.com/rapidsai/cuspatial/pull/596)) [@trxcllnt](https://github.com/trxcllnt)

# cuSpatial 22.08.00 (17 Aug 2022)

## 🚨 Breaking Changes

- Pairwise Point to Point Distance; Rename Folder `distances` to `distance` ([#558](https://github.com/rapidsai/cuspatial/pull/558)) [@isVoid](https://github.com/isVoid)

## 🐛 Bug Fixes

- Fix Broken Link in NYC Taxi Notebook ([#616](https://github.com/rapidsai/cuspatial/pull/616)) [@isVoid](https://github.com/isVoid)
- Add missing rmm includes ([#590](https://github.com/rapidsai/cuspatial/pull/590)) [@harrism](https://github.com/harrism)
- Fix failed automerge 22.06 into 22.08 ([#562](https://github.com/rapidsai/cuspatial/pull/562)) [@harrism](https://github.com/harrism)
- Bump cuspatial python version in scikit-build ([#550](https://github.com/rapidsai/cuspatial/pull/550)) [@isVoid](https://github.com/isVoid)

## 📖 Documentation

- Defer loading of `custom.js` ([#631](https://github.com/rapidsai/cuspatial/pull/631)) [@galipremsagar](https://github.com/galipremsagar)
- Use documented header template for `doxygen` ([#619](https://github.com/rapidsai/cuspatial/pull/619)) [@galipremsagar](https://github.com/galipremsagar)
- Fix issues with day &amp; night modes in python docs ([#613](https://github.com/rapidsai/cuspatial/pull/613)) [@isVoid](https://github.com/isVoid)

## 🚀 New Features

- Add NVBench and linestring distance benchmark ([#577](https://github.com/rapidsai/cuspatial/pull/577)) [@isVoid](https://github.com/isVoid)
- Pairwise Point to Point Distance; Rename Folder `distances` to `distance` ([#558](https://github.com/rapidsai/cuspatial/pull/558)) [@isVoid](https://github.com/isVoid)

## 🛠️ Improvements

- Bump `gdal` version ([#623](https://github.com/rapidsai/cuspatial/pull/623)) [@ajschmidt8](https://github.com/ajschmidt8)
- Build benchmarks in CI ([#597](https://github.com/rapidsai/cuspatial/pull/597)) [@vyasr](https://github.com/vyasr)
- Add benchmark for points_in_spatial_window ([#595](https://github.com/rapidsai/cuspatial/pull/595)) [@harrism](https://github.com/harrism)
- Update isort to version 5.10.1 ([#589](https://github.com/rapidsai/cuspatial/pull/589)) [@vyasr](https://github.com/vyasr)
- Header-only Refactor of `point_in_polygon` ([#587](https://github.com/rapidsai/cuspatial/pull/587)) [@isVoid](https://github.com/isVoid)
- Remove GeoArrow glue code replacing gpu storage with cudf.Series and host storage with pyarrow ([#585](https://github.com/rapidsai/cuspatial/pull/585)) [@thomcom](https://github.com/thomcom)
- Create `pygeoarrow` and use it for cuSpatial feature storage and i/o ([#583](https://github.com/rapidsai/cuspatial/pull/583)) [@thomcom](https://github.com/thomcom)
- Header-only refactoring of `points_in_spatial_window` ([#579](https://github.com/rapidsai/cuspatial/pull/579)) [@harrism](https://github.com/harrism)
- Update Python version support. ([#574](https://github.com/rapidsai/cuspatial/pull/574)) [@bdice](https://github.com/bdice)
- Combine `device_atomics` ([#561](https://github.com/rapidsai/cuspatial/pull/561)) [@isVoid](https://github.com/isVoid)
- Relocate Utility Files ([#560](https://github.com/rapidsai/cuspatial/pull/560)) [@isVoid](https://github.com/isVoid)
- Fuse `transform` and `copy_if` operations in `quadtree_point_in_polygon` ([#559](https://github.com/rapidsai/cuspatial/pull/559)) [@trxcllnt](https://github.com/trxcllnt)
- Remove `itstimestamp` and `types.hpp` ([#554](https://github.com/rapidsai/cuspatial/pull/554)) [@isVoid](https://github.com/isVoid)
- Change build.sh to find C++ library by default and avoid shadowing CMAKE_ARGS ([#543](https://github.com/rapidsai/cuspatial/pull/543)) [@vyasr](https://github.com/vyasr)
- Add missing Thrust includes ([#539](https://github.com/rapidsai/cuspatial/pull/539)) [@bdice](https://github.com/bdice)
- Refactor Hausdorff distance to header-only API ([#538](https://github.com/rapidsai/cuspatial/pull/538)) [@harrism](https://github.com/harrism)

# cuSpatial 22.06.00 (7 Jun 2022)

## 🐛 Bug Fixes

- Fix out of bounds access in spline interpolation ([#544](https://github.com/rapidsai/cuspatial/pull/544)) [@vyasr](https://github.com/vyasr)
- Fix `libcuspatial` recipe dependencies ([#513](https://github.com/rapidsai/cuspatial/pull/513)) [@ajschmidt8](https://github.com/ajschmidt8)

## 📖 Documentation

- Add Doxygen Documentation for `libcuspatial` ([#534](https://github.com/rapidsai/cuspatial/pull/534)) [@isVoid](https://github.com/isVoid)
- add units to haversine distance docstring ([#522](https://github.com/rapidsai/cuspatial/pull/522)) [@bandersen23](https://github.com/bandersen23)
- Merge branch-22.04 into branch-22.06 ([#518](https://github.com/rapidsai/cuspatial/pull/518)) [@harrism](https://github.com/harrism)

## 🚀 New Features

- Refactor `linestring_distance` to header only API ([#526](https://github.com/rapidsai/cuspatial/pull/526)) [@isVoid](https://github.com/isVoid)
- Python Bindings for Pairwise Linestring Distance ([#521](https://github.com/rapidsai/cuspatial/pull/521)) [@isVoid](https://github.com/isVoid)
- Refactor lonlat_to_cartesian to header-only API ([#514](https://github.com/rapidsai/cuspatial/pull/514)) [@harrism](https://github.com/harrism)
- C++ pairwise linestring distance ([#510](https://github.com/rapidsai/cuspatial/pull/510)) [@isVoid](https://github.com/isVoid)

## 🛠️ Improvements

- Fix library directory for installation ([#537](https://github.com/rapidsai/cuspatial/pull/537)) [@vyasr](https://github.com/vyasr)
- Fix conda recipes ([#532](https://github.com/rapidsai/cuspatial/pull/532)) [@Ethyling](https://github.com/Ethyling)
- Fix various issues with CMake exports ([#527](https://github.com/rapidsai/cuspatial/pull/527)) [@vyasr](https://github.com/vyasr)
- Build cuspatial with scikit-build ([#524](https://github.com/rapidsai/cuspatial/pull/524)) [@vyasr](https://github.com/vyasr)
- Update Documentation with Pydata Sphinx Theme, and more ([#523](https://github.com/rapidsai/cuspatial/pull/523)) [@isVoid](https://github.com/isVoid)
- Use conda to build python packages during GPU tests ([#517](https://github.com/rapidsai/cuspatial/pull/517)) [@Ethyling](https://github.com/Ethyling)
- Replace `CUDA_TRY` with `CUSPATIAL_CUDA_TRY` ([#516](https://github.com/rapidsai/cuspatial/pull/516)) [@isVoid](https://github.com/isVoid)
- Use rapids-cmake for builds ([#515](https://github.com/rapidsai/cuspatial/pull/515)) [@vyasr](https://github.com/vyasr)
- Update black to 22.3.0, update usage of libcudf macros, and remove direct column indexing ([#511](https://github.com/rapidsai/cuspatial/pull/511)) [@charlesbluca](https://github.com/charlesbluca)
- Enable building static libs ([#506](https://github.com/rapidsai/cuspatial/pull/506)) [@trxcllnt](https://github.com/trxcllnt)
- Add clang-format to pre-commit hook ([#505](https://github.com/rapidsai/cuspatial/pull/505)) [@isVoid](https://github.com/isVoid)
- Add libcuspatial-tests package ([#499](https://github.com/rapidsai/cuspatial/pull/499)) [@Ethyling](https://github.com/Ethyling)
- Use conda compilers ([#495](https://github.com/rapidsai/cuspatial/pull/495)) [@Ethyling](https://github.com/Ethyling)
- Build packages using mambabuild ([#486](https://github.com/rapidsai/cuspatial/pull/486)) [@Ethyling](https://github.com/Ethyling)
- Refactor haversine_distance to a header-only API ([#477](https://github.com/rapidsai/cuspatial/pull/477)) [@harrism](https://github.com/harrism)

# cuSpatial 22.04.00 (6 Apr 2022)

## 🐛 Bug Fixes

- Swap NumericalColumn.values_host for now removed to_array ([#485](https://github.com/rapidsai/cuspatial/pull/485)) [@vyasr](https://github.com/vyasr)

## 📖 Documentation

- Improve point_in_polygon documentation regarding poly_ring_offsets ([#497](https://github.com/rapidsai/cuspatial/pull/497)) [@harrism](https://github.com/harrism)
- Fix documentation of return type of quadtree_point_in_polygon ([#490](https://github.com/rapidsai/cuspatial/pull/490)) [@harrism](https://github.com/harrism)

## 🛠️ Improvements

- Temporarily disable new `ops-bot` functionality ([#501](https://github.com/rapidsai/cuspatial/pull/501)) [@ajschmidt8](https://github.com/ajschmidt8)
- Pin gtest/gmock to 1.10.0 in dev envs ([#498](https://github.com/rapidsai/cuspatial/pull/498)) [@trxcllnt](https://github.com/trxcllnt)
- Add `.github/ops-bot.yaml` config file ([#496](https://github.com/rapidsai/cuspatial/pull/496)) [@ajschmidt8](https://github.com/ajschmidt8)
- Add CMake `install` rule for tests ([#488](https://github.com/rapidsai/cuspatial/pull/488)) [@ajschmidt8](https://github.com/ajschmidt8)
- replace `ccache` with `sccache` ([#483](https://github.com/rapidsai/cuspatial/pull/483)) [@AyodeAwe](https://github.com/AyodeAwe)

# cuSpatial 22.02.00 (2 Feb 2022)

## 🐛 Bug Fixes

- Always upload cuspatial packages ([#481](https://github.com/rapidsai/cuspatial/pull/481)) [@raydouglass](https://github.com/raydouglass)
- Remove use of libcudf&#39;s CUDA_HOST_DEVICE macro ([#475](https://github.com/rapidsai/cuspatial/pull/475)) [@harrism](https://github.com/harrism)

## 🛠️ Improvements

- Prepare upload scripts for Python 3.7 removal ([#479](https://github.com/rapidsai/cuspatial/pull/479)) [@Ethyling](https://github.com/Ethyling)
- Fix `test_pip_bitmap_column_to_binary_array` test ([#472](https://github.com/rapidsai/cuspatial/pull/472)) [@Ethyling](https://github.com/Ethyling)
- Fix imports tests syntax ([#471](https://github.com/rapidsai/cuspatial/pull/471)) [@Ethyling](https://github.com/Ethyling)
- Remove `IncludeCategories` from `.clang-format` ([#470](https://github.com/rapidsai/cuspatial/pull/470)) [@codereport](https://github.com/codereport)
- Fix Forward-Merge Conflicts in #464 ([#466](https://github.com/rapidsai/cuspatial/pull/466)) [@ajschmidt8](https://github.com/ajschmidt8)

# cuSpatial 21.12.00 (9 Dec 2021)

## 🐛 Bug Fixes

- Remove use of now removed cudf Table object. ([#455](https://github.com/rapidsai/cuspatial/pull/455)) [@vyasr](https://github.com/vyasr)

## 🛠️ Improvements

- Update DEFAULT_CUDA_VER in ci/cpu/prebuild.sh ([#468](https://github.com/rapidsai/cuspatial/pull/468)) [@Ethyling](https://github.com/Ethyling)
- Fix Changelog Merge Conflicts for `branch-21.12` ([#467](https://github.com/rapidsai/cuspatial/pull/467)) [@ajschmidt8](https://github.com/ajschmidt8)
- Upgrade `clang` to `11.1.0` ([#463](https://github.com/rapidsai/cuspatial/pull/463)) [@galipremsagar](https://github.com/galipremsagar)
- NVCC warnings are errors ([#458](https://github.com/rapidsai/cuspatial/pull/458)) [@trxcllnt](https://github.com/trxcllnt)
- Update `conda` recipes for Enhanced Compatibility effort ([#457](https://github.com/rapidsai/cuspatial/pull/457)) [@ajschmidt8](https://github.com/ajschmidt8)

# cuSpatial 21.10.00 (7 Oct 2021)

## 🐛 Bug Fixes

- Stop using now removed _apply_support_method function. ([#448](https://github.com/rapidsai/cuspatial/pull/448)) [@vyasr](https://github.com/vyasr)
- Remove cudf.core ([#444](https://github.com/rapidsai/cuspatial/pull/444)) [@thomcom](https://github.com/thomcom)
- FIX Sync version update script with CMakeLists and update version in … ([#438](https://github.com/rapidsai/cuspatial/pull/438)) [@dillon-cullinan](https://github.com/dillon-cullinan)

## 🛠️ Improvements

- Fix default cuda version in prebuild.sh for arm64 ([#451](https://github.com/rapidsai/cuspatial/pull/451)) [@Ethyling](https://github.com/Ethyling)
- Skip imports tests on arm64 ([#450](https://github.com/rapidsai/cuspatial/pull/450)) [@Ethyling](https://github.com/Ethyling)
- Update Cython Table APIs to match changes in cudf. ([#449](https://github.com/rapidsai/cuspatial/pull/449)) [@vyasr](https://github.com/vyasr)
- Fix Forward-Merge Conflicts ([#445](https://github.com/rapidsai/cuspatial/pull/445)) [@ajschmidt8](https://github.com/ajschmidt8)
- Update cudf table apis ([#437](https://github.com/rapidsai/cuspatial/pull/437)) [@vyasr](https://github.com/vyasr)
- ENH Replace gpuci_conda_retry with gpuci_mamba_retry ([#432](https://github.com/rapidsai/cuspatial/pull/432)) [@dillon-cullinan](https://github.com/dillon-cullinan)
- Pin gdal to 3.3.x ([#420](https://github.com/rapidsai/cuspatial/pull/420)) [@weiji14](https://github.com/weiji14)

# cuSpatial 21.08.00 (4 Aug 2021)

## 🐛 Bug Fixes

- Fix usage of assert_columns* APIs. ([#433](https://github.com/rapidsai/cuspatial/pull/433)) [@vyasr](https://github.com/vyasr)
- Switch to using public cudf testing utilities ([#431](https://github.com/rapidsai/cuspatial/pull/431)) [@vyasr](https://github.com/vyasr)
- Update CMake, fix testing use of assert_eq, and correct metadata copying ([#430](https://github.com/rapidsai/cuspatial/pull/430)) [@vyasr](https://github.com/vyasr)
- Improve Hausdorff perf and accept larger number of inputs. ([#424](https://github.com/rapidsai/cuspatial/pull/424)) [@cwharris](https://github.com/cwharris)
- Fix a critical issue with `parallel_search` functor. ([#405](https://github.com/rapidsai/cuspatial/pull/405)) [@thomcom](https://github.com/thomcom)

## 🛠️ Improvements

- Updating Clang Version to 11.0.0 ([#426](https://github.com/rapidsai/cuspatial/pull/426)) [@codereport](https://github.com/codereport)
- Update sphinx config ([#421](https://github.com/rapidsai/cuspatial/pull/421)) [@ajschmidt8](https://github.com/ajschmidt8)
- Bump isort, enable Cython package resorting ([#419](https://github.com/rapidsai/cuspatial/pull/419)) [@charlesbluca](https://github.com/charlesbluca)
- Fix `21.08` forward-merge conflicts ([#418](https://github.com/rapidsai/cuspatial/pull/418)) [@ajschmidt8](https://github.com/ajschmidt8)
- Correct the docs example for `cuspatial.CubicSpline` ([#411](https://github.com/rapidsai/cuspatial/pull/411)) [@thomcom](https://github.com/thomcom)
- Update version to 21.08.00 in cmakelists ([#410](https://github.com/rapidsai/cuspatial/pull/410)) [@harrism](https://github.com/harrism)
- Fix merge conflicts ([#408](https://github.com/rapidsai/cuspatial/pull/408)) [@ajschmidt8](https://github.com/ajschmidt8)
- Support spaces in build.sh paths ([#385](https://github.com/rapidsai/cuspatial/pull/385)) [@jolorunyomi](https://github.com/jolorunyomi)

# cuSpatial 21.06.00 (9 Jun 2021)

## 🚀 New Features

- `from_geopandas` converts GeoPandas GeoSeries objects into cuspatial GeoArrow form. ([#300](https://github.com/rapidsai/cuspatial/pull/300)) [@thomcom](https://github.com/thomcom)

## 🛠️ Improvements

- Update environment variable used to determine `cuda_version` ([#407](https://github.com/rapidsai/cuspatial/pull/407)) [@ajschmidt8](https://github.com/ajschmidt8)
- Update `CHANGELOG.md` links for calver ([#404](https://github.com/rapidsai/cuspatial/pull/404)) [@ajschmidt8](https://github.com/ajschmidt8)
- Move rmm::device_buffer instead of copying ([#403](https://github.com/rapidsai/cuspatial/pull/403)) [@harrism](https://github.com/harrism)
- Update docs build script ([#402](https://github.com/rapidsai/cuspatial/pull/402)) [@ajschmidt8](https://github.com/ajschmidt8)
- Update cuspatial version for calver, fix CMake FindPackage cudf ([#401](https://github.com/rapidsai/cuspatial/pull/401)) [@trxcllnt](https://github.com/trxcllnt)
- Improve performance of quadtree point-to-polyline join ([#362](https://github.com/rapidsai/cuspatial/pull/362)) [@trxcllnt](https://github.com/trxcllnt)

# cuSpatial 0.19.0 (21 Apr 2021)

## 🐛 Bug Fixes

- Revert &quot;Update conda recipes pinning of repo dependencies&quot; ([#372](https://github.com//rapidsai/cuspatial/pull/372)) [@raydouglass](https://github.com/raydouglass)
- Update conda recipes pinning of repo dependencies ([#370](https://github.com//rapidsai/cuspatial/pull/370)) [@mike-wendt](https://github.com/mike-wendt)
- Rename cartesian_product_group_index_iterator_test.cpp to .cu ([#369](https://github.com//rapidsai/cuspatial/pull/369)) [@trxcllnt](https://github.com/trxcllnt)

## 🚀 New Features

- Auto-label PRs based on their content ([#337](https://github.com//rapidsai/cuspatial/pull/337)) [@jolorunyomi](https://github.com/jolorunyomi)

## 🛠️ Improvements

- Set `install_rpath` for libcuspatial ([#375](https://github.com//rapidsai/cuspatial/pull/375)) [@trxcllnt](https://github.com/trxcllnt)
- ENH Reduce cuspatial library size ([#373](https://github.com//rapidsai/cuspatial/pull/373)) [@dillon-cullinan](https://github.com/dillon-cullinan)
- Eliminate literals passed to device_uvector::set_element_async ([#367](https://github.com//rapidsai/cuspatial/pull/367)) [@harrism](https://github.com/harrism)
- Update Changelog Link ([#366](https://github.com//rapidsai/cuspatial/pull/366)) [@ajschmidt8](https://github.com/ajschmidt8)
- Rework libcuspatial CMakeLists.txt to export targets for CPM ([#365](https://github.com//rapidsai/cuspatial/pull/365)) [@trxcllnt](https://github.com/trxcllnt)
- Prepare Changelog for Automation ([#364](https://github.com//rapidsai/cuspatial/pull/364)) [@ajschmidt8](https://github.com/ajschmidt8)
- Update 0.18 changelog entry ([#363](https://github.com//rapidsai/cuspatial/pull/363)) [@ajschmidt8](https://github.com/ajschmidt8)
- ENH Build with `cmake --build` &amp; Pass ccache variables to conda recipe &amp; use Ninja in CI ([#359](https://github.com//rapidsai/cuspatial/pull/359)) [@Ethyling](https://github.com/Ethyling)
- Update make_counting_transform_iterator namespace in tests ([#353](https://github.com//rapidsai/cuspatial/pull/353)) [@trxcllnt](https://github.com/trxcllnt)
- Fix merge conflicts in #348 ([#349](https://github.com//rapidsai/cuspatial/pull/349)) [@ajschmidt8](https://github.com/ajschmidt8)
- Pin gdal to 3.2.x ([#347](https://github.com//rapidsai/cuspatial/pull/347)) [@weiji14](https://github.com/weiji14)

# cuSpatial 0.18.0 (24 Feb 2021)

## Documentation 📖

- Fix directed_hausdorff_distance space_offsets name + documentation (#332) @cwharris

## New Features 🚀

- New build process script changes &amp; gpuCI enhancements (#338) @raydouglass

## Improvements 🛠️

- Update stale GHA with exemptions &amp; new labels (#357) @mike-wendt
- Add GHA to mark issues/prs as stale/rotten (#355) @Ethyling
- Prepare Changelog for Automation (#345) @ajschmidt8
- Pin gdal to 3.1.x (#339) @weiji14
- Use simplified `rmm::exec_policy` (#331) @harrism
- Upgrade to libcu++ on GitHub (#297) @trxcllnt

# cuSpatial 0.17.0 (10 Dec 2020)

## New Features

## Improvements
- PR #310 Pin cmake policies to cmake 3.17 version
- PR #321 Improvements to gpuCI scripts
- PR #325 Convert `cudaStream_t` to `rmm::cuda_stream_view`

## Bug Fixes
- PR #320 Fix quadtree construction bug: zero out `device_uvector` before `scatter`
- PR #328 Fix point in polygon test for cudf::gather breaking change

# cuSpatial 0.16.0 (21 Oct 2020)

## New Features
- PR #290 Add Java bindings and a cuSpatial JAR package for Java users.

## Improvements
- PR #278 Update googlebenchmark version to match rmm & cudf.
- PR #286 Upgrade Thrust to latest commit.
- PR #287 Replace RMM get_default_resource with get_current_device_resource.
- PR #289 Fix cmake warnings for GoogleTest amd GoogleBenchmark external projects.
- PR #292 Update include paths to libcudf test utilities.
- PR #295 Use move from libcpp.utility.
- PR #299 Update ci/local/README.md
- PR #303 Update yml files to include CUDA 11.0 and remove old supported versions

## Bug Fixes
- PR #291 Fix mislabeled columns in Python spatial join result table.
- PR #294 Fix include of deprecated RMM header file.
- PR #296 Updates for RMM being header only.
- PR #298 Fix Python docs to render first argument of each public function.
- PR #322 Fix build issues related to libcudf split build changes
- PR #323 Add cuda to target_link_libraries


# cuSpatial 0.15.0 (26 Aug 2020)

## New Features

- PR #146 quadtree-polygon pairing for spatial filtering
- PR #149 Add quadtree-based point-in-polygon and point-to-nearest-polyline

## Improvements
- PR #237 Remove nvstrings references from CMakeLists.txt.
- PR #239 Add docs build script.
- PR #238 Fix library and include paths in CMakeLists.txt and setup.py.
- PR #240 Remove deprecated RMM header references.
- PR #243 Install dependencies via meta package.
- PR #247 Use rmm::device_uvector and cudf::UINT types for quadtree construction.
- PR #246 Hausdorff performance improvement.
- PR #253 Update conda upload versions for new supported CUDA/Python.
- PR #250 Cartesian product iterator + more Hausdorff performance improvements.
- PR #260 Replace RMM `cnmem_memory_resource` with `pool_memory_resource` in benchmark fixture.
- PR #264 Rename `quad_bbox_join` to `join_quadtree_and_bounding_boxes`.
- PR #268 Fix cudf timestamp construction.
- PR #267 Fix branch-0.15 conda dev environment dependencies

## Bug Fixes
- PR #244 Restrict gdal version.
- PR #245 Pin gdal to be compatible with cuxfilter.
- PR #242 Fix benchmark_fixture to use memory resources.
- PR #248 Fix build by updating type_id usages after upstream breaking changes.
- PR #252 Fix CI style check failures.
- PR #254 Fix issue with incorrect docker image being used in local build script.
- PR #258 Fix compiler errors from cudf's new duration types.
- PR #281 Patch Thrust to workaround `CUDA_CUB_RET_IF_FAIL` macro clearing CUDA errors


# cuSpatial 0.14.0 (03 Jun 2020)

## New Features

- PR #143 Support constructing quadtrees on point data
- PR #182 Local gpuCI build script
- PR #145 Support computing polygon and polyline bounding boxes
- PR #208 NYC Taxi Years Correlation Notebook (thanks @taureandyernv)

## Improvements

- PR #147 Update Conda/CMake configs to match other RAPIDS projects
- PR #163 Fix cudf legacy Cython imports/cimports
- PR #166 Move trajectory.hpp files to legacy
- PR #167 Align utility.hpp with libcudf style
- PR #171 Update trajectory.hpp to libcudf++
- PR #173 Move hausdorff.hpp files to legacy
- PR #172 Move coordinate_transform.hpp files to legacy
- PR #170 Update coordinate_transform.hpp to libcudf++
- PR #174 Update hausdorff.hpp to libcudf++
- PR #183 Add libcuspatial benchmark scaffolding
- PR #186 Move haversine.hpp files to legacy
- PR #194 Add .clang-format & format all files
- PR #190 Port coordinate_transform.hpp cython files
- PR #191 Move point_in_polygon.hpp files to legacy
- PR #193 Move shapefile_readers.hpp files to legacy
- PR #196 Move utilities/utilities.hpp to legacy
- PR #195 Fix PIP docs
- PR #197 Move query.hpp files to legacy
- PR #198 Port spatial_window queries to libcudf++
- PR #192 Update point_in_polygon.hpp to libcudf++
- PR #201 Update trajectory cython to libcudf++
- PR #189 Update haversine.hpp files to libcudf++
- PR #200 Update shapefile_readers.hpp to libcudf++
- PR #203 Port point_in_polygon.hpp cython files
- PR #202 Update haversine cython to libcudf++
- PR #204 Port shapefile_readers.hpp cython files
- PR #205 Port hausdorff.hpp cython to libcudf++
- PR #206 Remove legacy code.
- PR #214 Install gdal>=3.0.2 in build.sh
- PR #222 Fix potential thrust launch failure in quadtree building
- PR #221 Add python methods to api.rst, fix formatting
- PR #225 Add short git commit to conda package
- PR #228 Fix polygon and polyline docstrings

## Bug Fixes

- PR #141 Fix dangling exec_policy pointer and invalid num_ring argument.
- PR #169 Fix shapefile reader compilation with GCC 7.x / CUDA 10.2
- PR #178 Fix broken haversine tests introduced by upstream CUDF PRs.
- PR #175 Address RMM API changes by eliminating the use of the RMM_API
- PR #199 Fix coordinate transform tests
- PR #212 Rename calls to cudf::experimental namespace to cudf::
- PR #215 Replace legacy RMM calls
- PR #218 Fix benchmark build by removing test_benchmark.cpp
- PR #232 Fix conda dependencies

# cuSpatial 0.13.0 (31 Mar 2020)

## New Features

- PR #126 Create and build cuSpatial Docs
- PR #130 Add cubic spline fit and interpolation

## Improvements

- PR #128 Use RMM's `DeviceBuffer` for Python allocations
- PR #142 Disable deprecation warnings by default
- PR #138 Update Build instructions in the README

## Bug Fixes

- PR #123 Update references to error utils after libcudf changes
- PR #136 Remove build erroring for deprecation warnings


# cuSpatial 0.12.0 (04 Feb 2020)

## New Features

## Improvements

- PR #109 Update OPS codeowners group name
- PR #113 Support libcudf++

## Bug Fixes

- PR #116 Fix API issue with shapefile reader


# cuSpatial 0.11.0 (11 Dec 2019)

## New Features

- PR #86 Add Shapefile reader for polygons
- PR #92 Python bindings for shapefile reader

## Improvements

- PR #104 Remove unused CUDA conda labels

## Bug Fixes

- PR #94 Add legacy headers as cudf migrates
- PR #98 Updates to accommodate cudf refactoring
- PR #103 Update the include paths for cuda_utils


# cuSpatial 0.10.0 (16 Oct 2019)

## New Features

- PR #7 Initial code
- PR #18 Python initial unit tests and bindings
- PR #32 Python API first pass
- PR #37 Python __init__.py package design
- PR #38 Add __init__.py empties to resolve issue with PYTHONPATH
- PR #25 Add gpuCI integration

## Improvements

- PR #31 Add Github CODEOWNERS
- PR #39 Add cython headers to install, python / cmake packaging cleanup
- PR #41 Python and Cython style cleanup, pre-commit hook
- PR #44 Update all demos with Python API
- PR #45 Improve documentation in haversine and point in polygon
- PR #50 Validate that distance and speed work with all datetimes
- PR #58 Hausdorff distance returns a DataFrame, and better docs.
- PR #61 Point-in-polygon DataFrame output
- PR #59 Improve detail of point in polygon docs
- PR #64 Use YYMMDD tag in nightly build
- PR #68 Use YYMMDD tag in nightly build of cuspatial python
- PR #97 Drop `cython` from run requirements
- PR #82 Added update-version.sh
- PR #86 Add Shapefile reader for polygons

## Bug Fixes

- PR #16 `cuspatial::subset_trajectory_id()` test improvements and bug fixes
- PR #17 Update issue / PR templates
- PR #23 Fix cudf Cython imports
- PR #24 `cuspatial::derive_trajectories()` test improvements and bug fixes
- PR #33 `cuspatial::trajectory_distance_and_speed()` test improvements and bug fixes
- PR #49 Docstring for haversine and argument ordering was backwards
- PR #66 added missing header in tests
- PR #70 Require width parameterization of bitmap to binary conversion
