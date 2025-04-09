# cuspatial 25.04.00 (9 Apr 2025)

## 🐛 Bug Fixes

- Patch PROJ to remove uninstall target ([#1552](https://github.com/rapidsai/cuspatial/pull/1552)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Change download method for trajectory_clustering.ipynb&#39;s data for container testing ([#1539](https://github.com/rapidsai/cuspatial/pull/1539)) [@taureandyernv](https://github.com/taureandyernv)

## 🛠️ Improvements

- Pass dtype objects to cudf as_column calls ([#1549](https://github.com/rapidsai/cuspatial/pull/1549)) [@mroeschke](https://github.com/mroeschke)
- Replace `thrust::optional` with `cuda::std::optional` ([#1545](https://github.com/rapidsai/cuspatial/pull/1545)) [@miscco](https://github.com/miscco)
- Use conda-build instead of conda-mambabuild ([#1544](https://github.com/rapidsai/cuspatial/pull/1544)) [@bdice](https://github.com/bdice)
- Require CMake 3.30.4 ([#1543](https://github.com/rapidsai/cuspatial/pull/1543)) [@robertmaynard](https://github.com/robertmaynard)
- Add build_type input to `test.yaml` ([#1541](https://github.com/rapidsai/cuspatial/pull/1541)) [@gforsyth](https://github.com/gforsyth)
- Use new rapids-logger library ([#1538](https://github.com/rapidsai/cuspatial/pull/1538)) [@vyasr](https://github.com/vyasr)
- Use shared-workflows branch-25.04 ([#1536](https://github.com/rapidsai/cuspatial/pull/1536)) [@bdice](https://github.com/bdice)
- add telemetry ([#1535](https://github.com/rapidsai/cuspatial/pull/1535)) [@msarahan](https://github.com/msarahan)
- Forward-merge branch-25.02 to branch-25.04 ([#1532](https://github.com/rapidsai/cuspatial/pull/1532)) [@bdice](https://github.com/bdice)
- Forward-merge branch-25.02 into branch-25.04 ([#1531](https://github.com/rapidsai/cuspatial/pull/1531)) [@mroeschke](https://github.com/mroeschke)
- disallow fallback to Make in Python builds ([#1529](https://github.com/rapidsai/cuspatial/pull/1529)) [@jameslamb](https://github.com/jameslamb)
- Add `verify-codeowners` hook ([#1528](https://github.com/rapidsai/cuspatial/pull/1528)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Forward-merge branch-25.02 to branch-25.04 ([#1525](https://github.com/rapidsai/cuspatial/pull/1525)) [@bdice](https://github.com/bdice)
- Migrate to NVKS for amd64 CI runners ([#1522](https://github.com/rapidsai/cuspatial/pull/1522)) [@bdice](https://github.com/bdice)

# cuspatial 25.02.00 (13 Feb 2025)

## 🐛 Bug Fixes

- Use test_python changed files list for Python tests. ([#1527](https://github.com/rapidsai/cuspatial/pull/1527)) [@bdice](https://github.com/bdice)

## 📖 Documentation

- remove versioneer and setup.cfg references in docs, update rapids-dependency-file-generator ([#1509](https://github.com/rapidsai/cuspatial/pull/1509)) [@jameslamb](https://github.com/jameslamb)
- Fix libcuproj doc example and make tests consistent. ([#1503](https://github.com/rapidsai/cuspatial/pull/1503)) [@harrism](https://github.com/harrism)

## 🚀 New Features

- Add CUDA device-callable projection API ([#1490](https://github.com/rapidsai/cuspatial/pull/1490)) [@harrism](https://github.com/harrism)

## 🛠️ Improvements

- Use `rapids-pip-retry` in CI jobs that might need retries ([#1533](https://github.com/rapidsai/cuspatial/pull/1533)) [@gforsyth](https://github.com/gforsyth)
- Revert CUDA 12.8 shared workflow branch changes ([#1524](https://github.com/rapidsai/cuspatial/pull/1524)) [@vyasr](https://github.com/vyasr)
- Ensure Cython bindings only interact with pylibcudf objects ([#1523](https://github.com/rapidsai/cuspatial/pull/1523)) [@mroeschke](https://github.com/mroeschke)
- Build and test with CUDA 12.8.0 ([#1518](https://github.com/rapidsai/cuspatial/pull/1518)) [@bdice](https://github.com/bdice)
- Remove nyc_taxi_years_correlation.ipynb due to missing taxi geo dataset ([#1517](https://github.com/rapidsai/cuspatial/pull/1517)) [@mroeschke](https://github.com/mroeschke)
- improve parallelism of nightly wheel builds ([#1516](https://github.com/rapidsai/cuspatial/pull/1516)) [@jameslamb](https://github.com/jameslamb)
- Add shellcheck to pre-commit and fix warnings ([#1515](https://github.com/rapidsai/cuspatial/pull/1515)) [@gforsyth](https://github.com/gforsyth)
- Have Cython methods accept pylibcudf.Column instead of cudf._lib.column.Column ([#1514](https://github.com/rapidsai/cuspatial/pull/1514)) [@mroeschke](https://github.com/mroeschke)
- Use GCC 13 in CUDA 12 conda builds. ([#1513](https://github.com/rapidsai/cuspatial/pull/1513)) [@bdice](https://github.com/bdice)
- Use cuda.bindings ([#1511](https://github.com/rapidsai/cuspatial/pull/1511)) [@bdice](https://github.com/bdice)
- Check if nightlies have succeeded recently enough ([#1508](https://github.com/rapidsai/cuspatial/pull/1508)) [@vyasr](https://github.com/vyasr)
- Remove sphinx pinning ([#1507](https://github.com/rapidsai/cuspatial/pull/1507)) [@vyasr](https://github.com/vyasr)
- Replace removed cudf._lib.utils method with pylibcudf APIs ([#1505](https://github.com/rapidsai/cuspatial/pull/1505)) [@mroeschke](https://github.com/mroeschke)
- Update version references in workflow ([#1498](https://github.com/rapidsai/cuspatial/pull/1498)) [@AyodeAwe](https://github.com/AyodeAwe)
- Remove unused masked keyword in column_empty ([#1496](https://github.com/rapidsai/cuspatial/pull/1496)) [@mroeschke](https://github.com/mroeschke)
- Remove --expt-relaxed-constexpr NVCC flag ([#1495](https://github.com/rapidsai/cuspatial/pull/1495)) [@harrism](https://github.com/harrism)
- Require approval to run CI on draft PRs ([#1491](https://github.com/rapidsai/cuspatial/pull/1491)) [@bdice](https://github.com/bdice)
- Add breaking change workflow trigger ([#1481](https://github.com/rapidsai/cuspatial/pull/1481)) [@AyodeAwe](https://github.com/AyodeAwe)

# cuspatial 24.12.00 (11 Dec 2024)

## 🚨 Breaking Changes

- prefer wheel-provided libcuspatial.so in load_library(), use RTLD_LOCAL ([#1483](https://github.com/rapidsai/cuspatial/pull/1483)) [@jameslamb](https://github.com/jameslamb)

## 🐛 Bug Fixes

- fix library-loading issues in editable installs ([#1484](https://github.com/rapidsai/cuspatial/pull/1484)) [@jameslamb](https://github.com/jameslamb)
- Make polygon fixture data valid ([#1472](https://github.com/rapidsai/cuspatial/pull/1472)) [@harrism](https://github.com/harrism)

## 📖 Documentation

- Update README architecture and platform support ([#1476](https://github.com/rapidsai/cuspatial/pull/1476)) [@bdice](https://github.com/bdice)

## 🛠️ Improvements

- prefer wheel-provided libcuspatial.so in load_library(), use RTLD_LOCAL ([#1483](https://github.com/rapidsai/cuspatial/pull/1483)) [@jameslamb](https://github.com/jameslamb)
- enforce wheel size limits, README formatting in CI ([#1482](https://github.com/rapidsai/cuspatial/pull/1482)) [@jameslamb](https://github.com/jameslamb)
- Prune workflows based on changed files ([#1479](https://github.com/rapidsai/cuspatial/pull/1479)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- devcontainer: replace `VAULT_HOST` with `AWS_ROLE_ARN` ([#1478](https://github.com/rapidsai/cuspatial/pull/1478)) [@jjacobelli](https://github.com/jjacobelli)
- Updated libcudftestutil CMake linking logic for 24.12 ([#1475](https://github.com/rapidsai/cuspatial/pull/1475)) [@lamarrr](https://github.com/lamarrr)
- build wheels without build isolation ([#1473](https://github.com/rapidsai/cuspatial/pull/1473)) [@jameslamb](https://github.com/jameslamb)
- remove unused variables in build_docs script ([#1469](https://github.com/rapidsai/cuspatial/pull/1469)) [@jameslamb](https://github.com/jameslamb)
- Merge branch-24.10 into branch-24.12 ([#1463](https://github.com/rapidsai/cuspatial/pull/1463)) [@jameslamb](https://github.com/jameslamb)
- Fix performance regression in quadtree_point_in_polygon by 5x ([#1446](https://github.com/rapidsai/cuspatial/pull/1446)) [@mroeschke](https://github.com/mroeschke)

# cuspatial 24.10.00 (9 Oct 2024)

## 🚨 Breaking Changes

- Disallow GeoSeries from accepting a column in favor of `_from_column` ([#1434](https://github.com/rapidsai/cuspatial/pull/1434)) [@mroeschke](https://github.com/mroeschke)

## 🐛 Bug Fixes

- re-enable mixing libcuspatial wheels with libcudf conda packages ([#1456](https://github.com/rapidsai/cuspatial/pull/1456)) [@jameslamb](https://github.com/jameslamb)
- fix wheel tests on Rocky Linux ([#1452](https://github.com/rapidsai/cuspatial/pull/1452)) [@jameslamb](https://github.com/jameslamb)
- Add data=None to ListColumn constructor ([#1442](https://github.com/rapidsai/cuspatial/pull/1442)) [@mroeschke](https://github.com/mroeschke)
- Remove unused shuffle_order in GeoColumn ([#1439](https://github.com/rapidsai/cuspatial/pull/1439)) [@mroeschke](https://github.com/mroeschke)
- Fix _recombine_columns when either geocolumn or non-geocolumn was empty ([#1438](https://github.com/rapidsai/cuspatial/pull/1438)) [@mroeschke](https://github.com/mroeschke)
- Internal linkage kernels ([#1436](https://github.com/rapidsai/cuspatial/pull/1436)) [@robertmaynard](https://github.com/robertmaynard)
- Disallow GeoSeries from accepting a column in favor of `_from_column` ([#1434](https://github.com/rapidsai/cuspatial/pull/1434)) [@mroeschke](https://github.com/mroeschke)

## 📖 Documentation

- Recommending `miniforge` for conda install ([#1457](https://github.com/rapidsai/cuspatial/pull/1457)) [@mmccarty](https://github.com/mmccarty)

## 🛠️ Improvements

- xfail Python tests dependent on invalid Polygons ([#1467](https://github.com/rapidsai/cuspatial/pull/1467)) [@mroeschke](https://github.com/mroeschke)
- Switch `traceback` to `native` in pytests ([#1464](https://github.com/rapidsai/cuspatial/pull/1464)) [@galipremsagar](https://github.com/galipremsagar)
- Update update-version.sh to use packaging lib ([#1462](https://github.com/rapidsai/cuspatial/pull/1462)) [@AyodeAwe](https://github.com/AyodeAwe)
- Use CI workflow branch &#39;branch-24.10&#39; again ([#1459](https://github.com/rapidsai/cuspatial/pull/1459)) [@jameslamb](https://github.com/jameslamb)
- Update flake8 to 7.1.1. ([#1458](https://github.com/rapidsai/cuspatial/pull/1458)) [@bdice](https://github.com/bdice)
- Add support for Python 3.12, pin proj back to 9.3.x, require geopandas&gt;=1.0 ([#1453](https://github.com/rapidsai/cuspatial/pull/1453)) [@jameslamb](https://github.com/jameslamb)
- rearrange dependencies.yaml, fix development instructions ([#1451](https://github.com/rapidsai/cuspatial/pull/1451)) [@jameslamb](https://github.com/jameslamb)
- Distribute libcuspatial wheels ([#1450](https://github.com/rapidsai/cuspatial/pull/1450)) [@jameslamb](https://github.com/jameslamb)
- cuproj: remove pin on &#39;proj&#39; in conda packages ([#1449](https://github.com/rapidsai/cuspatial/pull/1449)) [@jameslamb](https://github.com/jameslamb)
- cuproj: depend on librmm, not rmm ([#1448](https://github.com/rapidsai/cuspatial/pull/1448)) [@jameslamb](https://github.com/jameslamb)
- wheels: dynamically load `libcudf.so` from `libcudf` wheel ([#1447](https://github.com/rapidsai/cuspatial/pull/1447)) [@jameslamb](https://github.com/jameslamb)
- Update rapidsai/pre-commit-hooks ([#1445](https://github.com/rapidsai/cuspatial/pull/1445)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Drop Python 3.9 support ([#1443](https://github.com/rapidsai/cuspatial/pull/1443)) [@jameslamb](https://github.com/jameslamb)
- Remove NumPy &lt;2 pin, remove proj pin ([#1441](https://github.com/rapidsai/cuspatial/pull/1441)) [@seberg](https://github.com/seberg)
- Update cuspatial to use pylibcudf ([#1440](https://github.com/rapidsai/cuspatial/pull/1440)) [@vyasr](https://github.com/vyasr)
- Update pre-commit hooks ([#1435](https://github.com/rapidsai/cuspatial/pull/1435)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Improve update-version.sh ([#1432](https://github.com/rapidsai/cuspatial/pull/1432)) [@bdice](https://github.com/bdice)
- Use tool.scikit-build.cmake.version, set scikit-build-core minimum-version ([#1430](https://github.com/rapidsai/cuspatial/pull/1430)) [@jameslamb](https://github.com/jameslamb)
- Pass columns instead of Series to `cudf.DataFrame` in split-combine workflow ([#1429](https://github.com/rapidsai/cuspatial/pull/1429)) [@mroeschke](https://github.com/mroeschke)
- Address cudf.DataFrame.insert API change ([#1428](https://github.com/rapidsai/cuspatial/pull/1428)) [@mroeschke](https://github.com/mroeschke)
- ensure notebook-testing script fails when notebooks fail ([#1424](https://github.com/rapidsai/cuspatial/pull/1424)) [@jameslamb](https://github.com/jameslamb)
- Forward-merge branch-24.08 into branch-24.10 ([#1410](https://github.com/rapidsai/cuspatial/pull/1410)) [@jakirkham](https://github.com/jakirkham)

# cuspatial 24.08.00 (7 Aug 2024)

## 🐛 Bug Fixes

- Unskip taxi notebook from CI ([#1422](https://github.com/rapidsai/cuspatial/pull/1422)) [@harrism](https://github.com/harrism)
- Use CuPy array in `pip_bitmap_column_to_binary_array` ([#1418](https://github.com/rapidsai/cuspatial/pull/1418)) [@isVoid](https://github.com/isVoid)
- Conda: Move `cmake` to `requirements/build` ([#1409](https://github.com/rapidsai/cuspatial/pull/1409)) [@jakirkham](https://github.com/jakirkham)
- test notebooks in &#39;docs/&#39;, make cuspatial_api_examples self-contained, skip long-running notebook, fix some docs ([#1407](https://github.com/rapidsai/cuspatial/pull/1407)) [@jameslamb](https://github.com/jameslamb)
- Fix geopandas upgrades ([#1404](https://github.com/rapidsai/cuspatial/pull/1404)) [@quasiben](https://github.com/quasiben)
- Fix multipolygon geometry iterator. ([#1402](https://github.com/rapidsai/cuspatial/pull/1402)) [@bdice](https://github.com/bdice)
- Change _GeoSeriesUtility._from_data(index=) default to None ([#1400](https://github.com/rapidsai/cuspatial/pull/1400)) [@mroeschke](https://github.com/mroeschke)
- Fix test_intersections.py post cudf refactor ([#1398](https://github.com/rapidsai/cuspatial/pull/1398)) [@mroeschke](https://github.com/mroeschke)
- Allow anonymous user in devcontainer name ([#1396](https://github.com/rapidsai/cuspatial/pull/1396)) [@bdice](https://github.com/bdice)

## 🛠️ Improvements

- Ensure _from_data accepts columns objects only ([#1415](https://github.com/rapidsai/cuspatial/pull/1415)) [@mroeschke](https://github.com/mroeschke)
- split up CUDA-suffixed dependencies in dependencies.yaml ([#1414](https://github.com/rapidsai/cuspatial/pull/1414)) [@jameslamb](https://github.com/jameslamb)
- Build and test with CUDA 12.5.1 ([#1405](https://github.com/rapidsai/cuspatial/pull/1405)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- skip CMake 3.30.0 ([#1401](https://github.com/rapidsai/cuspatial/pull/1401)) [@jameslamb](https://github.com/jameslamb)
- Use verify-alpha-spec hook ([#1399](https://github.com/rapidsai/cuspatial/pull/1399)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Adopt CI/packaging codeowners ([#1397](https://github.com/rapidsai/cuspatial/pull/1397)) [@bdice](https://github.com/bdice)
- Remove text builds of documentation ([#1394](https://github.com/rapidsai/cuspatial/pull/1394)) [@vyasr](https://github.com/vyasr)
- use rapids-build-backend ([#1393](https://github.com/rapidsai/cuspatial/pull/1393)) [@jameslamb](https://github.com/jameslamb)
- Fix forward-merge `branch-24.06` into `branch-24.08` ([#1390](https://github.com/rapidsai/cuspatial/pull/1390)) [@trxcllnt](https://github.com/trxcllnt)
- remove unnecessary &#39;setuptools&#39; dependencies ([#1389](https://github.com/rapidsai/cuspatial/pull/1389)) [@jameslamb](https://github.com/jameslamb)

# cuspatial 24.06.00 (5 Jun 2024)

## 🚨 Breaking Changes

- Replace rmm::mr::device_memory_resource* with rmm::device_async_resource_ref ([#1373](https://github.com/rapidsai/cuspatial/pull/1373)) [@harrism](https://github.com/harrism)

## 🐛 Bug Fixes

- create conda ci test env in one step ([#1387](https://github.com/rapidsai/cuspatial/pull/1387)) [@msarahan](https://github.com/msarahan)

## 🛠️ Improvements

- Fix up imports for cudf changes ([#1383](https://github.com/rapidsai/cuspatial/pull/1383)) [@vyasr](https://github.com/vyasr)
- Fix building cuspatial with CCCL main ([#1382](https://github.com/rapidsai/cuspatial/pull/1382)) [@trxcllnt](https://github.com/trxcllnt)
- Fix quadtree spatial join OOMs on large numbers of input polygons ([#1381](https://github.com/rapidsai/cuspatial/pull/1381)) [@trxcllnt](https://github.com/trxcllnt)
- Enable warnings-as-errors for cuproj tests ([#1379](https://github.com/rapidsai/cuspatial/pull/1379)) [@mroeschke](https://github.com/mroeschke)
- Always use a static gtest and gbench ([#1377](https://github.com/rapidsai/cuspatial/pull/1377)) [@trxcllnt](https://github.com/trxcllnt)
- Migrate to `{{ stdlib(&quot;c&quot;) }}` ([#1376](https://github.com/rapidsai/cuspatial/pull/1376)) [@hcho3](https://github.com/hcho3)
- add --rm and --name to devcontainer run args ([#1375](https://github.com/rapidsai/cuspatial/pull/1375)) [@trxcllnt](https://github.com/trxcllnt)
- Replace rmm::mr::device_memory_resource* with rmm::device_async_resource_ref ([#1373](https://github.com/rapidsai/cuspatial/pull/1373)) [@harrism](https://github.com/harrism)
- Enable all tests for `arm` jobs ([#1365](https://github.com/rapidsai/cuspatial/pull/1365)) [@galipremsagar](https://github.com/galipremsagar)
- Enable pytest failures on warnings on FutureWarnings (Replace deprecated `geopandas.dataset`  module) ([#1360](https://github.com/rapidsai/cuspatial/pull/1360)) [@mroeschke](https://github.com/mroeschke)
- Fix `JOIN_POINT_IN_POLYGON_LARGE_TEST_EXP` test ([#1346](https://github.com/rapidsai/cuspatial/pull/1346)) [@trxcllnt](https://github.com/trxcllnt)

# cuSpatial 24.04.00 (10 Apr 2024)

## 🐛 Bug Fixes

- Update pre-commit-hooks to v0.0.3 ([#1361](https://github.com/rapidsai/cuspatial/pull/1361)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Fix Python 3.11 test failures ([#1351](https://github.com/rapidsai/cuspatial/pull/1351)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Fix Broken CI for 24.04 ([#1338](https://github.com/rapidsai/cuspatial/pull/1338)) [@isVoid](https://github.com/isVoid)
- handle more RAPIDS version formats in update-version.sh, refactor dependencies.yaml ([#1329](https://github.com/rapidsai/cuspatial/pull/1329)) [@jameslamb](https://github.com/jameslamb)

## 📖 Documentation

- fix formatting in developer guide ([#1330](https://github.com/rapidsai/cuspatial/pull/1330)) [@jameslamb](https://github.com/jameslamb)

## 🛠️ Improvements

- Use `conda env create --yes` instead of `--force` ([#1364](https://github.com/rapidsai/cuspatial/pull/1364)) [@bdice](https://github.com/bdice)
- Add upper bound to prevent usage of NumPy 2 ([#1359](https://github.com/rapidsai/cuspatial/pull/1359)) [@bdice](https://github.com/bdice)
- Enable pytest failures on warnings from cudf (Ensure values being set are casted to the object&#39;s type) ([#1358](https://github.com/rapidsai/cuspatial/pull/1358)) [@mroeschke](https://github.com/mroeschke)
- Remove hard-coding of RAPIDS version where possible ([#1357](https://github.com/rapidsai/cuspatial/pull/1357)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Replace `cudf.core` imports with public APIs ([#1356](https://github.com/rapidsai/cuspatial/pull/1356)) [@mroeschke](https://github.com/mroeschke)
- Switch `pytest-xdist` algorithm to `worksteal` ([#1355](https://github.com/rapidsai/cuspatial/pull/1355)) [@bdice](https://github.com/bdice)
- Replace cudf column.full with cudf.as_column ([#1350](https://github.com/rapidsai/cuspatial/pull/1350)) [@mroeschke](https://github.com/mroeschke)
- Automate C++ include file grouping and ordering using clang-format ([#1349](https://github.com/rapidsai/cuspatial/pull/1349)) [@harrism](https://github.com/harrism)
- Add support for Python 3.11, require NumPy 1.23+ ([#1348](https://github.com/rapidsai/cuspatial/pull/1348)) [@jameslamb](https://github.com/jameslamb)
- write PROJ build dir export rules ([#1347](https://github.com/rapidsai/cuspatial/pull/1347)) [@trxcllnt](https://github.com/trxcllnt)
- Update devcontainers to CUDA Toolkit 12.2 ([#1344](https://github.com/rapidsai/cuspatial/pull/1344)) [@trxcllnt](https://github.com/trxcllnt)
- target branch-24.04 for GitHub Actions workflows ([#1343](https://github.com/rapidsai/cuspatial/pull/1343)) [@jameslamb](https://github.com/jameslamb)
- Update ops-bot.yaml ([#1337](https://github.com/rapidsai/cuspatial/pull/1337)) [@AyodeAwe](https://github.com/AyodeAwe)
- Remove -DNVBench_ENABLE_CUPTI=OFF. ([#1325](https://github.com/rapidsai/cuspatial/pull/1325)) [@bdice](https://github.com/bdice)
- Support CUDA 12.2 ([#1320](https://github.com/rapidsai/cuspatial/pull/1320)) [@jameslamb](https://github.com/jameslamb)

# cuSpatial 24.02.00 (12 Feb 2024)

## 🚨 Breaking Changes

- Update to CCCL 2.2.0. ([#1315](https://github.com/rapidsai/cuspatial/pull/1315)) [@bdice](https://github.com/bdice)
- Switch to scikit-build-core ([#1313](https://github.com/rapidsai/cuspatial/pull/1313)) [@vyasr](https://github.com/vyasr)

## 🐛 Bug Fixes

- Exclude tests from builds ([#1339](https://github.com/rapidsai/cuspatial/pull/1339)) [@vyasr](https://github.com/vyasr)
- Make sure cudf uses pyarrow&#39;s libarrow during wheel builds ([#1334](https://github.com/rapidsai/cuspatial/pull/1334)) [@vyasr](https://github.com/vyasr)
- Replace cudf _from_columns with a public API ([#1326](https://github.com/rapidsai/cuspatial/pull/1326)) [@mroeschke](https://github.com/mroeschke)
- Update usage of cudf.core.column.arange to cudf.core.column.as_column ([#1323](https://github.com/rapidsai/cuspatial/pull/1323)) [@AjayThorve](https://github.com/AjayThorve)

## 🚀 New Features

- Use ListColumn instead of build_list_column ([#1327](https://github.com/rapidsai/cuspatial/pull/1327)) [@mroeschke](https://github.com/mroeschke)

## 🛠️ Improvements

- Remove usages of rapids-env-update ([#1321](https://github.com/rapidsai/cuspatial/pull/1321)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Add explicit initial size to all pools and avoid RMM detail APIs ([#1319](https://github.com/rapidsai/cuspatial/pull/1319)) [@harrism](https://github.com/harrism)
- refactor CUDA versions in dependencies.yaml ([#1317](https://github.com/rapidsai/cuspatial/pull/1317)) [@jameslamb](https://github.com/jameslamb)
- Update to CCCL 2.2.0. ([#1315](https://github.com/rapidsai/cuspatial/pull/1315)) [@bdice](https://github.com/bdice)
- Use cuda::proclaim_return_type on device lambdas. ([#1314](https://github.com/rapidsai/cuspatial/pull/1314)) [@bdice](https://github.com/bdice)
- Switch to scikit-build-core ([#1313](https://github.com/rapidsai/cuspatial/pull/1313)) [@vyasr](https://github.com/vyasr)
- Use Cython 3 for cuproj. ([#1309](https://github.com/rapidsai/cuspatial/pull/1309)) [@bdice](https://github.com/bdice)
- Update dependencies.yaml to new pip index ([#1305](https://github.com/rapidsai/cuspatial/pull/1305)) [@vyasr](https://github.com/vyasr)
- Remove CUSPATIAL_BUILD_WHEELS and standardize Python builds ([#1304](https://github.com/rapidsai/cuspatial/pull/1304)) [@vyasr](https://github.com/vyasr)

# cuSpatial 23.12.00 (6 Dec 2023)

## 🐛 Bug Fixes

- Update actions/labeler to v4 ([#1301](https://github.com/rapidsai/cuspatial/pull/1301)) [@raydouglass](https://github.com/raydouglass)
- Fix broken hausdorff test ([#1295](https://github.com/rapidsai/cuspatial/pull/1295)) [@isVoid](https://github.com/isVoid)
- Pin theme ([#1289](https://github.com/rapidsai/cuspatial/pull/1289)) [@vyasr](https://github.com/vyasr)

## 🛠️ Improvements

- Build concurrency for nightly and merge triggers ([#1298](https://github.com/rapidsai/cuspatial/pull/1298)) [@bdice](https://github.com/bdice)
- Update versioning strategy ([#1285](https://github.com/rapidsai/cuspatial/pull/1285)) [@vyasr](https://github.com/vyasr)
- Use branch-23.12 workflows. ([#1284](https://github.com/rapidsai/cuspatial/pull/1284)) [@bdice](https://github.com/bdice)
- Build CUDA 12.0 ARM conda packages. ([#1275](https://github.com/rapidsai/cuspatial/pull/1275)) [@bdice](https://github.com/bdice)
- Fix failed automerge Branch 23.12 merge 23.10 ([#1273](https://github.com/rapidsai/cuspatial/pull/1273)) [@harrism](https://github.com/harrism)

# cuSpatial 23.10.00 (11 Oct 2023)

## 🚨 Breaking Changes

- Update to Cython 3.0.0 ([#1235](https://github.com/rapidsai/cuspatial/pull/1235)) [@vyasr](https://github.com/vyasr)

## 🐛 Bug Fixes

- Add `curl` for notebook testing ([#1281](https://github.com/rapidsai/cuspatial/pull/1281)) [@raydouglass](https://github.com/raydouglass)
- Use `conda mambabuild` not `mamba mambabuild` ([#1265](https://github.com/rapidsai/cuspatial/pull/1265)) [@bdice](https://github.com/bdice)
- Add back deleted conda env files ([#1261](https://github.com/rapidsai/cuspatial/pull/1261)) [@harrism](https://github.com/harrism)

## 📖 Documentation

- [REVIEW] Add cudf example code for haversine distance into 23.10 ([#1260](https://github.com/rapidsai/cuspatial/pull/1260)) [@taureandyernv](https://github.com/taureandyernv)

## 🛠️ Improvements

- Benchmark `GeoSeries.Distance` ([#1277](https://github.com/rapidsai/cuspatial/pull/1277)) [@isVoid](https://github.com/isVoid)
- fix update-version.sh ([#1272](https://github.com/rapidsai/cuspatial/pull/1272)) [@trxcllnt](https://github.com/trxcllnt)
- Update image names ([#1269](https://github.com/rapidsai/cuspatial/pull/1269)) [@AyodeAwe](https://github.com/AyodeAwe)
- Update to clang 16.0.6. ([#1266](https://github.com/rapidsai/cuspatial/pull/1266)) [@bdice](https://github.com/bdice)
- Simplify wheel build scripts and allow alphas of RAPIDS dependencies ([#1264](https://github.com/rapidsai/cuspatial/pull/1264)) [@divyegala](https://github.com/divyegala)
- Update devcontainers to align with other RAPIDS repositories ([#1263](https://github.com/rapidsai/cuspatial/pull/1263)) [@trxcllnt](https://github.com/trxcllnt)
- Use `copy-pr-bot` ([#1262](https://github.com/rapidsai/cuspatial/pull/1262)) [@ajschmidt8](https://github.com/ajschmidt8)
- Add binary_predicates.ipynb workflow and test data to package. ([#1258](https://github.com/rapidsai/cuspatial/pull/1258)) [@thomcom](https://github.com/thomcom)
- Re-enable cuspatial tests and dependency. ([#1253](https://github.com/rapidsai/cuspatial/pull/1253)) [@thomcom](https://github.com/thomcom)
- Fix 23.08 -&gt; 23.10 automerge ([#1251](https://github.com/rapidsai/cuspatial/pull/1251)) [@thomcom](https://github.com/thomcom)
- Change path to trajectories_extracted.p to improve CI run consistency. ([#1247](https://github.com/rapidsai/cuspatial/pull/1247)) [@thomcom](https://github.com/thomcom)
- Update to Cython 3.0.0 ([#1235](https://github.com/rapidsai/cuspatial/pull/1235)) [@vyasr](https://github.com/vyasr)
- Branch 23.10 merge 23.08 ([#1230](https://github.com/rapidsai/cuspatial/pull/1230)) [@vyasr](https://github.com/vyasr)

# cuSpatial 23.08.00 (9 Aug 2023)

## 🚨 Breaking Changes

- Simplify point-in-polygon header only APIs ([#1192](https://github.com/rapidsai/cuspatial/pull/1192)) [@isVoid](https://github.com/isVoid)

## 🐛 Bug Fixes

- Update nightly wheel tests for cuproj ([#1250](https://github.com/rapidsai/cuspatial/pull/1250)) [@raydouglass](https://github.com/raydouglass)
- Update GeoDataFrame to Use the Structured GatherMap Class ([#1219](https://github.com/rapidsai/cuspatial/pull/1219)) [@isVoid](https://github.com/isVoid)
- Add streams to allocate_like call ([#1218](https://github.com/rapidsai/cuspatial/pull/1218)) [@isVoid](https://github.com/isVoid)
- Add CUDA major to `cuspatial`&#39;s `build/string` ([#1211](https://github.com/rapidsai/cuspatial/pull/1211)) [@jakirkham](https://github.com/jakirkham)
- Fix overflowing in `intersection_intermediates.remove_if` ([#1209](https://github.com/rapidsai/cuspatial/pull/1209)) [@isVoid](https://github.com/isVoid)
- Sort the mergeable segments before computing merged segments ([#1207](https://github.com/rapidsai/cuspatial/pull/1207)) [@isVoid](https://github.com/isVoid)
- Fix a small typo in pairwise_linestring_distance ([#1199](https://github.com/rapidsai/cuspatial/pull/1199)) [@isVoid](https://github.com/isVoid)
- Improve Polygon.contains(LineString) predicate logic. ([#1186](https://github.com/rapidsai/cuspatial/pull/1186)) [@thomcom](https://github.com/thomcom)
- Use grid_stride_range in kernel loops ([#1178](https://github.com/rapidsai/cuspatial/pull/1178)) [@harrism](https://github.com/harrism)
- Better support for binary predicates with large inputs. ([#1166](https://github.com/rapidsai/cuspatial/pull/1166)) [@thomcom](https://github.com/thomcom)

## 📖 Documentation

- cuProj Python and C++ Documentation ([#1237](https://github.com/rapidsai/cuspatial/pull/1237)) [@harrism](https://github.com/harrism)
- cuSpatial 23.08 README updates ([#1232](https://github.com/rapidsai/cuspatial/pull/1232)) [@jarmak-nv](https://github.com/jarmak-nv)
- Add author credit to zipcode counting notebook, fix cudf string processing argument ([#1201](https://github.com/rapidsai/cuspatial/pull/1201)) [@isVoid](https://github.com/isVoid)
- Remove `osmnx` dependency and use a small local dataset ([#1195](https://github.com/rapidsai/cuspatial/pull/1195)) [@isVoid](https://github.com/isVoid)

## 🚀 New Features

- Add `geoseries.distance` ([#1231](https://github.com/rapidsai/cuspatial/pull/1231)) [@isVoid](https://github.com/isVoid)
- Initial Python bindings for cuProj ([#1217](https://github.com/rapidsai/cuspatial/pull/1217)) [@harrism](https://github.com/harrism)
- WGS84 to UTM benchmark (C++) ([#1216](https://github.com/rapidsai/cuspatial/pull/1216)) [@harrism](https://github.com/harrism)
- Adding Spherical Surface Processing Capability to Point-in-Polygon Queries ([#1212](https://github.com/rapidsai/cuspatial/pull/1212)) [@ayasar70](https://github.com/ayasar70)
- WGS84 &lt;--&gt; UTM projection header-only API ([#1191](https://github.com/rapidsai/cuspatial/pull/1191)) [@harrism](https://github.com/harrism)
- External issue triage GHA ([#1177](https://github.com/rapidsai/cuspatial/pull/1177)) [@jarmak-nv](https://github.com/jarmak-nv)
- Binary Predicates Introduction and Benchmark Notebook ([#1156](https://github.com/rapidsai/cuspatial/pull/1156)) [@thomcom](https://github.com/thomcom)
- Add cmake infrastructure for internal projection library ([#1132](https://github.com/rapidsai/cuspatial/pull/1132)) [@harrism](https://github.com/harrism)

## 🛠️ Improvements

- Switch to new wheel building pipeline ([#1227](https://github.com/rapidsai/cuspatial/pull/1227)) [@vyasr](https://github.com/vyasr)
- Use Google Benchmark from rapids-cmake in cuproj. ([#1225](https://github.com/rapidsai/cuspatial/pull/1225)) [@bdice](https://github.com/bdice)
- Use rapids-cmake to supply Google Benchmark library. ([#1224](https://github.com/rapidsai/cuspatial/pull/1224)) [@bdice](https://github.com/bdice)
- Revert CUDA 12.0 CI workflows to branch-23.08. ([#1223](https://github.com/rapidsai/cuspatial/pull/1223)) [@bdice](https://github.com/bdice)
- Update to `rapidsai/devcontainers:23.08` ([#1214](https://github.com/rapidsai/cuspatial/pull/1214)) [@trxcllnt](https://github.com/trxcllnt)
- Add Comprehensive Test for Multigeometry Range Classes ([#1197](https://github.com/rapidsai/cuspatial/pull/1197)) [@isVoid](https://github.com/isVoid)
- Update to CMake 3.26.4 ([#1196](https://github.com/rapidsai/cuspatial/pull/1196)) [@vyasr](https://github.com/vyasr)
- Simplify point-in-polygon header only APIs ([#1192](https://github.com/rapidsai/cuspatial/pull/1192)) [@isVoid](https://github.com/isVoid)
- Forward merge `branch-23.06` into `branch-23.08` ([#1190](https://github.com/rapidsai/cuspatial/pull/1190)) [@raydouglass](https://github.com/raydouglass)
- use rapids-upload-docs script ([#1181](https://github.com/rapidsai/cuspatial/pull/1181)) [@AyodeAwe](https://github.com/AyodeAwe)
- Remove Stale Demo Scripts ([#1180](https://github.com/rapidsai/cuspatial/pull/1180)) [@isVoid](https://github.com/isVoid)
- Updates Build Instructions to Adjust for Devcontainer Instructions and Remove Stale Information ([#1179](https://github.com/rapidsai/cuspatial/pull/1179)) [@isVoid](https://github.com/isVoid)
- Fix branch-23.06 forward-merge into branch-23.08 ([#1175](https://github.com/rapidsai/cuspatial/pull/1175)) [@trxcllnt](https://github.com/trxcllnt)
- Remove documentation build scripts for Jenkins ([#1169](https://github.com/rapidsai/cuspatial/pull/1169)) [@ajschmidt8](https://github.com/ajschmidt8)
- cuSpatial: Build CUDA 12 packages ([#1044](https://github.com/rapidsai/cuspatial/pull/1044)) [@isVoid](https://github.com/isVoid)

# cuSpatial 23.06.00 (7 Jun 2023)

## 🚨 Breaking Changes

- Reorganize cuSpatial headers ([#1097](https://github.com/rapidsai/cuspatial/pull/1097)) [@harrism](https://github.com/harrism)
- Update minimum Python version to Python 3.9 ([#1089](https://github.com/rapidsai/cuspatial/pull/1089)) [@shwina](https://github.com/shwina)
- Move `experimental` headers into main `include/cuspatial` directory ([#1081](https://github.com/rapidsai/cuspatial/pull/1081)) [@harrism](https://github.com/harrism)
- Improve Hausdorff Many Column Performance ([#916](https://github.com/rapidsai/cuspatial/pull/916)) [@isVoid](https://github.com/isVoid)

## 🐛 Bug Fixes

- Fix scatter bug due to overlapping range in `pairwise_linestring_intersection` ([#1152](https://github.com/rapidsai/cuspatial/pull/1152)) [@isVoid](https://github.com/isVoid)
- Pin cuml dependency in notebook testing environment to nightlies ([#1110](https://github.com/rapidsai/cuspatial/pull/1110)) [@isVoid](https://github.com/isVoid)
- Fix a bug in point-in-polygon kernel: if the point is collinear with an edge, result is asserted false ([#1108](https://github.com/rapidsai/cuspatial/pull/1108)) [@isVoid](https://github.com/isVoid)
- Fix a bug in segment intersection primitive where two collinear segment touch at endpoints is miscomputed as a degenerate segment ([#1093](https://github.com/rapidsai/cuspatial/pull/1093)) [@isVoid](https://github.com/isVoid)
- Update `CMAKE_CUDA_ARCHITECTURE` to use new value ([#1070](https://github.com/rapidsai/cuspatial/pull/1070)) [@isVoid](https://github.com/isVoid)
- Bug fix in `pairwise_linestring_intersection` ([#1069](https://github.com/rapidsai/cuspatial/pull/1069)) [@isVoid](https://github.com/isVoid)

## 📖 Documentation

- Add documentation for `pairwise_linestring_polygon_distance`, `pairwise_polygon_distance` ([#1145](https://github.com/rapidsai/cuspatial/pull/1145)) [@isVoid](https://github.com/isVoid)
- Make User Guide appear in Docs page header ([#1133](https://github.com/rapidsai/cuspatial/pull/1133)) [@jarmak-nv](https://github.com/jarmak-nv)
- Add Hausdorff Clustering Notebooks ([#922](https://github.com/rapidsai/cuspatial/pull/922)) [@isVoid](https://github.com/isVoid)

## 🚀 New Features

- Add Benchmark to `pairwise_linestring_polygon_distance` ([#1153](https://github.com/rapidsai/cuspatial/pull/1153)) [@isVoid](https://github.com/isVoid)
- Adds `pairwise_point_polygon_distance` benchmark ([#1131](https://github.com/rapidsai/cuspatial/pull/1131)) [@isVoid](https://github.com/isVoid)
- Reorganize cuSpatial headers ([#1097](https://github.com/rapidsai/cuspatial/pull/1097)) [@harrism](https://github.com/harrism)
- Python API for `pairwise_polygon_distance` ([#1074](https://github.com/rapidsai/cuspatial/pull/1074)) [@isVoid](https://github.com/isVoid)
- Column API for `pairwise_polygon_distance` ([#1073](https://github.com/rapidsai/cuspatial/pull/1073)) [@isVoid](https://github.com/isVoid)
- Header only API for polygon-polygon distance ([#1065](https://github.com/rapidsai/cuspatial/pull/1065)) [@isVoid](https://github.com/isVoid)
- Python API for linestring polygon distance ([#1031](https://github.com/rapidsai/cuspatial/pull/1031)) [@isVoid](https://github.com/isVoid)
- Column API for linestring-polygon distance ([#1030](https://github.com/rapidsai/cuspatial/pull/1030)) [@isVoid](https://github.com/isVoid)

## 🛠️ Improvements

- Fix `cudf::column` constructor args ([#1151](https://github.com/rapidsai/cuspatial/pull/1151)) [@trxcllnt](https://github.com/trxcllnt)
- cuSpatial pip packages ([#1148](https://github.com/rapidsai/cuspatial/pull/1148)) [@trxcllnt](https://github.com/trxcllnt)
- Refactor `ST_Distance` header only API ([#1143](https://github.com/rapidsai/cuspatial/pull/1143)) [@isVoid](https://github.com/isVoid)
- Run docs nightly ([#1141](https://github.com/rapidsai/cuspatial/pull/1141)) [@AyodeAwe](https://github.com/AyodeAwe)
- Add `multilinestring_segment_manager` for segment related methods in multilinestring ranges ([#1134](https://github.com/rapidsai/cuspatial/pull/1134)) [@isVoid](https://github.com/isVoid)
- Improve zipcode counting notebook by adding GPU backed WKT parser ([#1130](https://github.com/rapidsai/cuspatial/pull/1130)) [@isVoid](https://github.com/isVoid)
- Delete add_issue_to_project.yml ([#1129](https://github.com/rapidsai/cuspatial/pull/1129)) [@jarmak-nv](https://github.com/jarmak-nv)
- Bump Gtest version following Rapids-cmake change ([#1126](https://github.com/rapidsai/cuspatial/pull/1126)) [@isVoid](https://github.com/isVoid)
- Refactor ST_Distance Column API and Cython ([#1124](https://github.com/rapidsai/cuspatial/pull/1124)) [@isVoid](https://github.com/isVoid)
- Reorganize src, tests, and benchmarks ([#1115](https://github.com/rapidsai/cuspatial/pull/1115)) [@harrism](https://github.com/harrism)
- Add Legal Terms to Trajectory Clustering Notebook ([#1111](https://github.com/rapidsai/cuspatial/pull/1111)) [@isVoid](https://github.com/isVoid)
- Enable sccache hits from local builds ([#1109](https://github.com/rapidsai/cuspatial/pull/1109)) [@AyodeAwe](https://github.com/AyodeAwe)
- Revert to branch-23.06 for shared-action-workflows ([#1107](https://github.com/rapidsai/cuspatial/pull/1107)) [@shwina](https://github.com/shwina)
- Update minimum Python version to Python 3.9 ([#1089](https://github.com/rapidsai/cuspatial/pull/1089)) [@shwina](https://github.com/shwina)
- Remove usage of rapids-get-rapids-version-from-git ([#1088](https://github.com/rapidsai/cuspatial/pull/1088)) [@jjacobelli](https://github.com/jjacobelli)
- Add `contains`predicate. ([#1086](https://github.com/rapidsai/cuspatial/pull/1086)) [@thomcom](https://github.com/thomcom)
- Binary Predicate Test Dispatching ([#1085](https://github.com/rapidsai/cuspatial/pull/1085)) [@thomcom](https://github.com/thomcom)
- Move `experimental` headers into main `include/cuspatial` directory ([#1081](https://github.com/rapidsai/cuspatial/pull/1081)) [@harrism](https://github.com/harrism)
- Update clang-format to 15.0.7 ([#1072](https://github.com/rapidsai/cuspatial/pull/1072)) [@bdice](https://github.com/bdice)
- Use ARC V2 self-hosted runners for GPU jobs ([#1066](https://github.com/rapidsai/cuspatial/pull/1066)) [@jjacobelli](https://github.com/jjacobelli)
- Implement and Test All non-multi-Feature Spatial Predicate Combinations ([#1064](https://github.com/rapidsai/cuspatial/pull/1064)) [@thomcom](https://github.com/thomcom)
- Reduced equals time and fixed a bug. ([#1051](https://github.com/rapidsai/cuspatial/pull/1051)) [@thomcom](https://github.com/thomcom)
- use make_device_vector in pairwise_point_in_polygon_test ([#1049](https://github.com/rapidsai/cuspatial/pull/1049)) [@cwharris](https://github.com/cwharris)
- Use thrust::host_vector instead of std::vector&lt;bool&gt; in tests ([#1048](https://github.com/rapidsai/cuspatial/pull/1048)) [@cwharris](https://github.com/cwharris)
- Branch 23.06 merge 23.04 (2) ([#1035](https://github.com/rapidsai/cuspatial/pull/1035)) [@harrism](https://github.com/harrism)
- Pairwise Multipoint Equals Count function ([#1022](https://github.com/rapidsai/cuspatial/pull/1022)) [@thomcom](https://github.com/thomcom)
- Branch 23.06 merge 23.04 ([#1021](https://github.com/rapidsai/cuspatial/pull/1021)) [@harrism](https://github.com/harrism)
- Add GTC 2023 Reverse GeoCoding Demo Notebook ([#1001](https://github.com/rapidsai/cuspatial/pull/1001)) [@thomcom](https://github.com/thomcom)
- Improve Hausdorff Many Column Performance ([#916](https://github.com/rapidsai/cuspatial/pull/916)) [@isVoid](https://github.com/isVoid)

# cuSpatial 23.04.00 (6 Apr 2023)

## 🚨 Breaking Changes

- Consistently validate polygon inputs for GeoArrow offset format ([#973](https://github.com/rapidsai/cuspatial/pull/973)) [@harrism](https://github.com/harrism)
- Remove cubic spline interpolation ([#959](https://github.com/rapidsai/cuspatial/pull/959)) [@harrism](https://github.com/harrism)
- Refactors API to accept GeoSeries input for projection and trajectory functions ([#955](https://github.com/rapidsai/cuspatial/pull/955)) [@isVoid](https://github.com/isVoid)
- Refactors `filtering.py`, `indexing.py` to Accept GeoSeries ([#938](https://github.com/rapidsai/cuspatial/pull/938)) [@isVoid](https://github.com/isVoid)
- Refactors `bounding.py` to accept GeoSeries Input ([#934](https://github.com/rapidsai/cuspatial/pull/934)) [@isVoid](https://github.com/isVoid)
- Remove shapefile reader, conda GDAL dependency, move cmake gdal dependency to cpp tests only ([#932](https://github.com/rapidsai/cuspatial/pull/932)) [@harrism](https://github.com/harrism)
- Refactor `directed_hausdorff_distance` and `haversine_distance` into `GeoSeries` Interface ([#924](https://github.com/rapidsai/cuspatial/pull/924)) [@isVoid](https://github.com/isVoid)

## 🐛 Bug Fixes

- Bug Fix: point-in-multipolygon reduction using the wrong key-compare functor ([#1043](https://github.com/rapidsai/cuspatial/pull/1043)) [@isVoid](https://github.com/isVoid)
- Fix quotes in backticks in Developer Guide ([#1034](https://github.com/rapidsai/cuspatial/pull/1034)) [@harrism](https://github.com/harrism)
- Attempt to Fix Broken C++ Build After `cudftestutil` is Made a Shared Lib ([#996](https://github.com/rapidsai/cuspatial/pull/996)) [@isVoid](https://github.com/isVoid)
- Consistently validate polygon inputs for GeoArrow offset format ([#973](https://github.com/rapidsai/cuspatial/pull/973)) [@harrism](https://github.com/harrism)
- Fix OB bug in `linestring_intersection_intermediates.remove_if` Function ([#945](https://github.com/rapidsai/cuspatial/pull/945)) [@isVoid](https://github.com/isVoid)
- Fix broken `point_indices` methods in `PolygonGeoColumnAccessor` ([#907](https://github.com/rapidsai/cuspatial/pull/907)) [@isVoid](https://github.com/isVoid)
- Fix multiple bugs in user guide ([#906](https://github.com/rapidsai/cuspatial/pull/906)) [@isVoid](https://github.com/isVoid)
- `_from_point_xy` improvements ([#905](https://github.com/rapidsai/cuspatial/pull/905)) [@isVoid](https://github.com/isVoid)
- Add `valid_count` and `has_nulls` to GeoColumn ([#894](https://github.com/rapidsai/cuspatial/pull/894)) [@thomcom](https://github.com/thomcom)

## 📖 Documentation

- : Fix linestring link in readme ([#1003](https://github.com/rapidsai/cuspatial/pull/1003)) [@jarmak-nv](https://github.com/jarmak-nv)
- : Move build instructions to dev guide ([#999](https://github.com/rapidsai/cuspatial/pull/999)) [@jarmak-nv](https://github.com/jarmak-nv)
- Add `pairwise_linestring_intersection` example in user guide ([#989](https://github.com/rapidsai/cuspatial/pull/989)) [@isVoid](https://github.com/isVoid)
- Update cuSpatial Readme ([#977](https://github.com/rapidsai/cuspatial/pull/977)) [@jarmak-nv](https://github.com/jarmak-nv)
- Add ZipCode Counting Notebook ([#919](https://github.com/rapidsai/cuspatial/pull/919)) [@isVoid](https://github.com/isVoid)

## 🚀 New Features

- Add segment Iterators, test multi*_range and miscellaneous lazy iterator additions ([#1026](https://github.com/rapidsai/cuspatial/pull/1026)) [@isVoid](https://github.com/isVoid)
- Add Header Only API for Linestring-Polygon Distance ([#1011](https://github.com/rapidsai/cuspatial/pull/1011)) [@isVoid](https://github.com/isVoid)
- Add `geometry_generator` factory for programmatic generation of geometry arrays ([#998](https://github.com/rapidsai/cuspatial/pull/998)) [@isVoid](https://github.com/isVoid)
- Add python API `pairwise_point_polygon_distance` ([#988](https://github.com/rapidsai/cuspatial/pull/988)) [@isVoid](https://github.com/isVoid)
- Add column API for `pairwise_point_polygon_distance` ([#984](https://github.com/rapidsai/cuspatial/pull/984)) [@isVoid](https://github.com/isVoid)
- Add Header-Only `point_polygon_distance`, add non-owning polygon objects ([#976](https://github.com/rapidsai/cuspatial/pull/976)) [@isVoid](https://github.com/isVoid)
- Remove cubic spline interpolation ([#959](https://github.com/rapidsai/cuspatial/pull/959)) [@harrism](https://github.com/harrism)
- Remove shapefile reader, conda GDAL dependency, move cmake gdal dependency to cpp tests only ([#932](https://github.com/rapidsai/cuspatial/pull/932)) [@harrism](https://github.com/harrism)
- Add `from_linestrings_xy` and `from_polygons_xy` ([#928](https://github.com/rapidsai/cuspatial/pull/928)) [@thomcom](https://github.com/thomcom)
- Implement `geom_equals` and binary predicates that depend only on it. ([#926](https://github.com/rapidsai/cuspatial/pull/926)) [@thomcom](https://github.com/thomcom)
- Add `apply_boolean_mask` Feature ([#918](https://github.com/rapidsai/cuspatial/pull/918)) [@isVoid](https://github.com/isVoid)
- Add C++ Column API and Python API for `pairwise_linestring_intersection` ([#862](https://github.com/rapidsai/cuspatial/pull/862)) [@isVoid](https://github.com/isVoid)

## 🛠️ Improvements

- Refactor spatial join tests ([#1019](https://github.com/rapidsai/cuspatial/pull/1019)) [@harrism](https://github.com/harrism)
- Reduce gtest times ([#1018](https://github.com/rapidsai/cuspatial/pull/1018)) [@harrism](https://github.com/harrism)
- Intersection only predicates ([#1016](https://github.com/rapidsai/cuspatial/pull/1016)) [@thomcom](https://github.com/thomcom)
- Updated binpred architecture ([#1009](https://github.com/rapidsai/cuspatial/pull/1009)) [@thomcom](https://github.com/thomcom)
- Add `dependency-file-generator` as `pre-commit` hook ([#1008](https://github.com/rapidsai/cuspatial/pull/1008)) [@ajschmidt8](https://github.com/ajschmidt8)
- Header-only quadtree_point_to_nearest_linestring ([#1005](https://github.com/rapidsai/cuspatial/pull/1005)) [@harrism](https://github.com/harrism)
- Add codespell as a linter ([#992](https://github.com/rapidsai/cuspatial/pull/992)) [@bdice](https://github.com/bdice)
- Pass `AWS_SESSION_TOKEN` and `SCCACHE_S3_USE_SSL` vars to conda build ([#982](https://github.com/rapidsai/cuspatial/pull/982)) [@ajschmidt8](https://github.com/ajschmidt8)
- Header-only `quadtree_point_in_polygon` ([#979](https://github.com/rapidsai/cuspatial/pull/979)) [@harrism](https://github.com/harrism)
- Update aarch64 to GCC 11 ([#978](https://github.com/rapidsai/cuspatial/pull/978)) [@bdice](https://github.com/bdice)
- Remove GDAL dependency in quadtree spatial join tests. ([#974](https://github.com/rapidsai/cuspatial/pull/974)) [@harrism](https://github.com/harrism)
- Enable discussions ([#966](https://github.com/rapidsai/cuspatial/pull/966)) [@jarmak-nv](https://github.com/jarmak-nv)
- Fix docs build to be `pydata-sphinx-theme=0.13.0` compatible ([#964](https://github.com/rapidsai/cuspatial/pull/964)) [@galipremsagar](https://github.com/galipremsagar)
- Update `.gitignore` for `ops-codeowners` ([#963](https://github.com/rapidsai/cuspatial/pull/963)) [@ajschmidt8](https://github.com/ajschmidt8)
- Update to GCC 11 ([#961](https://github.com/rapidsai/cuspatial/pull/961)) [@bdice](https://github.com/bdice)
- Add cuspatial devcontainers ([#960](https://github.com/rapidsai/cuspatial/pull/960)) [@trxcllnt](https://github.com/trxcllnt)
- Make docs builds less verbose ([#956](https://github.com/rapidsai/cuspatial/pull/956)) [@AyodeAwe](https://github.com/AyodeAwe)
- Refactors API to accept GeoSeries input for projection and trajectory functions ([#955](https://github.com/rapidsai/cuspatial/pull/955)) [@isVoid](https://github.com/isVoid)
- Update Notebook with GeoSeries Usage and Add Notebook Tests ([#953](https://github.com/rapidsai/cuspatial/pull/953)) [@isVoid](https://github.com/isVoid)
- Refactor functions in `join.py` to accept GeoSeries Input ([#948](https://github.com/rapidsai/cuspatial/pull/948)) [@isVoid](https://github.com/isVoid)
- Skip docs job in nightly runs ([#944](https://github.com/rapidsai/cuspatial/pull/944)) [@AyodeAwe](https://github.com/AyodeAwe)
- Refactors `filtering.py`, `indexing.py` to Accept GeoSeries ([#938](https://github.com/rapidsai/cuspatial/pull/938)) [@isVoid](https://github.com/isVoid)
- Refactors `bounding.py` to accept GeoSeries Input ([#934](https://github.com/rapidsai/cuspatial/pull/934)) [@isVoid](https://github.com/isVoid)
- Remove dead code from ContainsProperlyBinpred paths. ([#933](https://github.com/rapidsai/cuspatial/pull/933)) [@thomcom](https://github.com/thomcom)
- Refactor `directed_hausdorff_distance` and `haversine_distance` into `GeoSeries` Interface ([#924](https://github.com/rapidsai/cuspatial/pull/924)) [@isVoid](https://github.com/isVoid)
- Reduce error handling verbosity in CI tests scripts ([#912](https://github.com/rapidsai/cuspatial/pull/912)) [@AjayThorve](https://github.com/AjayThorve)
- Use quadtree for `.contains_properly` ([#910](https://github.com/rapidsai/cuspatial/pull/910)) [@thomcom](https://github.com/thomcom)
- Update shared workflow branches ([#909](https://github.com/rapidsai/cuspatial/pull/909)) [@ajschmidt8](https://github.com/ajschmidt8)
- Remove gpuCI scripts. ([#904](https://github.com/rapidsai/cuspatial/pull/904)) [@bdice](https://github.com/bdice)
- Allow initialization of a `GeoDataFrame` using a `cudf.DataFrame` ([#895](https://github.com/rapidsai/cuspatial/pull/895)) [@thomcom](https://github.com/thomcom)
- Move date to build string in `conda` recipe ([#882](https://github.com/rapidsai/cuspatial/pull/882)) [@ajschmidt8](https://github.com/ajschmidt8)
- Add docs build job ([#868](https://github.com/rapidsai/cuspatial/pull/868)) [@AyodeAwe](https://github.com/AyodeAwe)

# cuSpatial 23.02.00 (9 Feb 2023)

## 🚨 Breaking Changes

- Deprecate shapefile reader ([#872](https://github.com/rapidsai/cuspatial/pull/872)) [@harrism](https://github.com/harrism)
- Remove deprecated min_size parameter from quadtree_on_points ([#843](https://github.com/rapidsai/cuspatial/pull/843)) [@harrism](https://github.com/harrism)
- Remove deprecated lonlat_to_cartesian functions ([#842](https://github.com/rapidsai/cuspatial/pull/842)) [@harrism](https://github.com/harrism)

## 🐛 Bug Fixes

- pre-commit: Update isort version to 5.12.0 ([#877](https://github.com/rapidsai/cuspatial/pull/877)) [@wence-](https://github.com/wence-)
- Use vector equivalence macros consistently in header-only API tests. ([#873](https://github.com/rapidsai/cuspatial/pull/873)) [@harrism](https://github.com/harrism)
- Fix: Sort Intersection Result by Keys in `intersection_with_duplicate` Test ([#857](https://github.com/rapidsai/cuspatial/pull/857)) [@isVoid](https://github.com/isVoid)
- Add `reset_index` to `GeoSeries` and `GeoDataFrame` ([#856](https://github.com/rapidsai/cuspatial/pull/856)) [@thomcom](https://github.com/thomcom)
- Update `memory_usage` with tweaked sizes ([#855](https://github.com/rapidsai/cuspatial/pull/855)) [@thomcom](https://github.com/thomcom)
- Fix a bug preventing `align` from working with a basic slice. ([#850](https://github.com/rapidsai/cuspatial/pull/850)) [@thomcom](https://github.com/thomcom)
- Fix failed automerge Branch 23.02 merge 22.12 ([#811](https://github.com/rapidsai/cuspatial/pull/811)) [@harrism](https://github.com/harrism)

## 📖 Documentation

- Fix documentation author ([#866](https://github.com/rapidsai/cuspatial/pull/866)) [@bdice](https://github.com/bdice)

## 🚀 New Features

- Deprecate shapefile reader ([#872](https://github.com/rapidsai/cuspatial/pull/872)) [@harrism](https://github.com/harrism)
- Add Header Only API `pairwise_linestring_intersection` ([#852](https://github.com/rapidsai/cuspatial/pull/852)) [@isVoid](https://github.com/isVoid)
- Adds `remove_if` Function to `linestring_intersection_intermediates` ([#851](https://github.com/rapidsai/cuspatial/pull/851)) [@isVoid](https://github.com/isVoid)
- Add `find_point_on_segment` internal API ([#819](https://github.com/rapidsai/cuspatial/pull/819)) [@isVoid](https://github.com/isVoid)
- Add `find_and_combine_segment` internal API ([#818](https://github.com/rapidsai/cuspatial/pull/818)) [@isVoid](https://github.com/isVoid)
- Add `pairwise_linestring_intersection_with_duplicates` API ([#813](https://github.com/rapidsai/cuspatial/pull/813)) [@isVoid](https://github.com/isVoid)

## 🛠️ Improvements

- Update shared workflow branches ([#897](https://github.com/rapidsai/cuspatial/pull/897)) [@ajschmidt8](https://github.com/ajschmidt8)
- Add print_env.sh to improve bug reporting ([#874](https://github.com/rapidsai/cuspatial/pull/874)) [@jarmak-nv](https://github.com/jarmak-nv)
- Refactor calls to `data_array_view` ([#871](https://github.com/rapidsai/cuspatial/pull/871)) [@galipremsagar](https://github.com/galipremsagar)
- Build CUDA 11.8 and Python 3.10 Packages ([#865](https://github.com/rapidsai/cuspatial/pull/865)) [@bdice](https://github.com/bdice)
- Update workflows for nightly tests ([#864](https://github.com/rapidsai/cuspatial/pull/864)) [@ajschmidt8](https://github.com/ajschmidt8)
- Enable `Recently Updated` Check ([#863](https://github.com/rapidsai/cuspatial/pull/863)) [@ajschmidt8](https://github.com/ajschmidt8)
- Add header-only `cuspatial::join_quadtree_and_bounding_boxes` ([#861](https://github.com/rapidsai/cuspatial/pull/861)) [@trxcllnt](https://github.com/trxcllnt)
- Remove deprecated min_size parameter from quadtree_on_points ([#843](https://github.com/rapidsai/cuspatial/pull/843)) [@harrism](https://github.com/harrism)
- Remove deprecated lonlat_to_cartesian functions ([#842](https://github.com/rapidsai/cuspatial/pull/842)) [@harrism](https://github.com/harrism)
- Use Libcudf Test Macro ([#841](https://github.com/rapidsai/cuspatial/pull/841)) [@isVoid](https://github.com/isVoid)
- Add Missing License in `Distance.py` ([#840](https://github.com/rapidsai/cuspatial/pull/840)) [@isVoid](https://github.com/isVoid)
- Enable copy_prs. ([#837](https://github.com/rapidsai/cuspatial/pull/837)) [@bdice](https://github.com/bdice)
- Add GitHub Actions Workflows ([#836](https://github.com/rapidsai/cuspatial/pull/836)) [@bdice](https://github.com/bdice)
- Implement all binary operations that depend only on point-in-polygon ([#834](https://github.com/rapidsai/cuspatial/pull/834)) [@thomcom](https://github.com/thomcom)
- Header-only polygon_bounding_boxes and linestring_bounding_boxes, `make_geometry_id_iterator` utility, and `box&lt;T&gt;` class. ([#820](https://github.com/rapidsai/cuspatial/pull/820)) [@harrism](https://github.com/harrism)

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
