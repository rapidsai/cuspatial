/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cuspatial_test/vector_equality.hpp>

#include <cuspatial/detail/iterator.hpp>
#include <cuspatial/error.hpp>
#include <cuspatial/experimental/iterator_factory.cuh>
#include <cuspatial/experimental/point_linestring_distance.cuh>
#include <cuspatial/experimental/ranges/multilinestring_range.cuh>
#include <cuspatial/experimental/ranges/multipoint_range.cuh>
#include <cuspatial/vec_2d.hpp>

#include <rmm/device_vector.hpp>

#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <type_traits>

namespace cuspatial {
namespace test {

template <typename T>
struct PairwisePointLinestringDistanceTest : public ::testing::Test {
};

// float and double are logically the same but would require seperate tests due to precision.
using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(PairwisePointLinestringDistanceTest, TestTypes);

TYPED_TEST(PairwisePointLinestringDistanceTest, Empty)
{
  using T       = TypeParam;
  using CartVec = std::vector<vec_2d<T>>;

  std::vector<int> point_geometry_offsets{0};
  CartVec points{};
  std::vector<int> linestring_geometry_offsets{0};
  std::vector<int> linestring_part_offsets{0};
  CartVec linestring_points{};

  rmm::device_vector<int> d_point_geometries(point_geometry_offsets);
  rmm::device_vector<vec_2d<T>> d_points(points);
  rmm::device_vector<int> d_linestring_geometries(linestring_geometry_offsets);
  rmm::device_vector<int> d_linestring_parts(linestring_part_offsets);
  rmm::device_vector<vec_2d<T>> d_linestring_points(linestring_points);

  auto multipoints = make_multipoint_range(d_point_geometries, d_points);
  auto multilinestrings =
    make_multilinestring_range(d_linestring_geometries, d_linestring_parts, d_linestring_points);

  rmm::device_vector<T> got{};
  thrust::host_vector<T> expect{};

  auto ret = pairwise_point_linestring_distance(multipoints, multilinestrings, got.begin());

  expect_vector_equivalent(expect, got);
  EXPECT_EQ(ret, got.end());
}

TYPED_TEST(PairwisePointLinestringDistanceTest, OnePairFromVectorSingleComponent)
{
  using T       = TypeParam;
  using CartVec = std::vector<vec_2d<T>>;

  std::vector<int> point_geometry_offsets{0, 1};
  CartVec points{{0, 0}};
  std::vector<int> linestring_geometry_offsets{0, 1};
  std::vector<int> linestring_part_offsets{0, 3};
  CartVec linestring_points{{1, 1}, {2, 2}, {3, 3}};
  thrust::host_vector<T> expect(std::vector<T>{std::sqrt(T{2.0})});

  rmm::device_vector<int> d_point_geometries(point_geometry_offsets);
  rmm::device_vector<vec_2d<T>> d_points(points);
  rmm::device_vector<int> d_linestring_geometries(linestring_geometry_offsets);
  rmm::device_vector<int> d_linestring_parts(linestring_part_offsets);
  rmm::device_vector<vec_2d<T>> d_linestring_points(linestring_points);

  auto multipoints = make_multipoint_range(
    d_point_geometries.size() - 1, d_point_geometries.begin(), d_points.size(), d_points.begin());

  auto multilinestrings = make_multilinestring_range(d_linestring_geometries.size() - 1,
                                                     d_linestring_geometries.begin(),
                                                     d_linestring_parts.size() - 1,
                                                     d_linestring_parts.begin(),
                                                     d_linestring_points.size(),
                                                     d_linestring_points.begin());

  rmm::device_vector<T> got(points.size());

  auto ret = pairwise_point_linestring_distance(multipoints, multilinestrings, got.begin());

  expect_vector_equivalent(expect, got);
  EXPECT_EQ(ret, got.end());
}

TYPED_TEST(PairwisePointLinestringDistanceTest, OnePairFromCountingIteratorSingleComponent)
{
  using T       = TypeParam;
  using CartVec = std::vector<vec_2d<T>>;

  std::size_t constexpr num_pairs = 1;

  CartVec points{{0, 0}};
  std::vector<int> linestring_part_offsets{0, 2};
  CartVec linestring_points{{1.0, 0.0}, {0.0, 1.0}};
  thrust::host_vector<T> expect(std::vector<T>{std::sqrt(T{2.0}) / 2});

  auto d_point_geometries = thrust::make_counting_iterator(0);
  rmm::device_vector<vec_2d<T>> d_points(points);
  auto d_linestring_geometries = thrust::make_counting_iterator(0);
  rmm::device_vector<int> d_linestring_parts(linestring_part_offsets);
  rmm::device_vector<vec_2d<T>> d_linestring_points(linestring_points);

  auto multipoints =
    make_multipoint_range(num_pairs, d_point_geometries, d_points.size(), d_points.begin());

  auto multilinestrings = make_multilinestring_range(num_pairs,
                                                     d_linestring_geometries,
                                                     d_linestring_parts.size() - 1,
                                                     d_linestring_parts.begin(),
                                                     d_linestring_points.size(),
                                                     d_linestring_points.begin());

  rmm::device_vector<T> got(num_pairs);

  auto ret = pairwise_point_linestring_distance(multipoints, multilinestrings, got.begin());

  expect_vector_equivalent(expect, got);
  EXPECT_EQ(ret, got.end());
}

TYPED_TEST(PairwisePointLinestringDistanceTest, TwoPairFromVectorSingleComponent)
{
  using T       = TypeParam;
  using CartVec = std::vector<vec_2d<T>>;

  std::vector<int> point_geometry_offsets{0, 1, 2};
  CartVec points{{0, 0}, {0, 0}};
  std::vector<int> linestring_geometry_offsets{0, 1, 2};
  std::vector<int> linestring_part_offsets{0, 3, 6};
  CartVec linestring_points{{1, 1}, {2, 2}, {3, 3}, {-1, -1}, {-2, -2}, {-3, -3}};
  thrust::host_vector<T> expect(std::vector<T>{std::sqrt(T{2.0}), std::sqrt(T{2.0})});

  rmm::device_vector<int> d_point_geometries(point_geometry_offsets);
  rmm::device_vector<vec_2d<T>> d_points(points);
  rmm::device_vector<int> d_linestring_geometries(linestring_geometry_offsets);
  rmm::device_vector<int> d_linestring_parts(linestring_part_offsets);
  rmm::device_vector<vec_2d<T>> d_linestring_points(linestring_points);

  auto multipoints = make_multipoint_range(
    d_point_geometries.size() - 1, d_point_geometries.begin(), d_points.size(), d_points.begin());

  auto multilinestrings = make_multilinestring_range(d_linestring_geometries.size() - 1,
                                                     d_linestring_geometries.begin(),
                                                     d_linestring_parts.size() - 1,
                                                     d_linestring_parts.begin(),
                                                     d_linestring_points.size(),
                                                     d_linestring_points.begin());

  rmm::device_vector<T> got(points.size());

  auto ret = pairwise_point_linestring_distance(multipoints, multilinestrings, got.begin());

  expect_vector_equivalent(expect, got);
  EXPECT_EQ(ret, got.end());
}

template <typename T>
struct times_three_functor {
  T __device__ operator()(T i) { return i * 3; }
};

TYPED_TEST(PairwisePointLinestringDistanceTest, ManyPairsFromIteratorsSingleComponent)
{
  using T = TypeParam;

  auto const num_pairs             = 100;
  auto const num_linestring_points = num_pairs * 3;

  auto linestring_points_x = thrust::make_counting_iterator(T{0.0});
  auto linestring_points_y = thrust::make_constant_iterator(T{1.0});
  auto linestring_points   = make_vec_2d_iterator(linestring_points_x, linestring_points_y);
  auto offsets =
    cuspatial::detail::make_counting_transform_iterator(0, times_three_functor<int32_t>{});
  auto linestring_geometries = thrust::make_counting_iterator(0);

  auto points_x =
    cuspatial::detail::make_counting_transform_iterator(T{0.0}, times_three_functor<T>{});
  auto points_y         = thrust::make_constant_iterator(T{0.0});
  auto points           = make_vec_2d_iterator(points_x, points_y);
  auto point_geometries = thrust::make_counting_iterator(0);

  auto multipoints = make_multipoint_range(num_pairs, point_geometries, num_pairs, points);

  auto multilinestrings = make_multilinestring_range(
    num_pairs, linestring_geometries, num_pairs, offsets, num_linestring_points, linestring_points);

  std::vector<T> expect(num_pairs, T{1.0});
  rmm::device_vector<T> got(num_pairs);

  auto ret = pairwise_point_linestring_distance(multipoints, multilinestrings, got.begin());

  expect_vector_equivalent(expect, got);
  EXPECT_EQ(ret, got.end());
}

TYPED_TEST(PairwisePointLinestringDistanceTest, OnePartFiftyPairsCompareWithShapely)
{
  using T       = TypeParam;
  using CartVec = std::vector<vec_2d<T>>;

  // All point coordinates are confined in [-1e9, 0] inverval
  auto d_points_x = rmm::device_vector<T>(std::vector<T>{
    -561549155.2815765,  -539635297.7968571,  -749785291.9823582,  -505256232.2383667,
    -946988876.2831948,  -662300741.9693683,  -603237481.1662251,  -125581339.4351779,
    -517833167.6852695,  -957160196.2622645,  -491585807.73353755, -345450303.82460994,
    -487395641.79169536, -735699507.1257033,  -948019350.519085,   -421003493.70237565,
    -614443644.8464075,  -91784671.00703202,  -358379636.97942185, -716600896.394933,
    -344246357.45209974, -954628165.511387,   -585389868.7519523,  -942712185.6486846,
    -918383957.1706945,  -894335076.9398956,  -199429182.6890826,  -308066036.9332076,
    -68539892.81333561,  -47428714.27856632,  -860697716.8271636,  -649546547.8989385,
    -861397168.6094841,  -254088440.42574397, -922519680.0380477,  -971662526.6980333,
    -549662841.5967332,  -315022158.10620314, -621043135.2056639,  -752210211.5984685,
    -795704940.2086449,  -346633871.30775416, -256629810.47606537, -816036577.7906327,
    -352357002.88786775, -369130482.495009,   -261037800.80460483, -996942546.6529481,
    -916557372.6650971,  -470793334.3911819});

  auto d_points_y = rmm::device_vector<T>(std::vector<T>{
    -739351480.2301654,  -20887279.805103853, -777641495.0417496,  -125601344.62234807,
    -920272855.235259,   -979386805.0183806,  -532400563.0878669,  -890451846.152133,
    -189264617.43542865, -716724991.9062672,  -112862367.28799225, -437346182.85413873,
    -10050108.356858267, -756947856.81533,    -201662709.30643314, -170076205.5474354,
    -347998961.0882306,  -747977546.543758,   -450349384.1171753,  -596418684.4693943,
    -890311812.3973312,  -181592857.0650911,  -102687386.99020985, -228778684.4019939,
    -347707985.5682359,  -146089663.45215645, -576531773.4037402,  -44916711.39299691,
    -663829461.7550983,  -862331153.6802458,  -205683639.49777836, -706774022.2667997,
    -805347167.0543085,  -94244793.56957604,  -888483038.4109963,  -22626853.80827789,
    -987307795.9680042,  -907735202.344958,   -75385732.90465944,  -580412876.6046127,
    -237450679.54106826, -181080163.82408464, -431973802.7579398,  -818515080.5689257,
    -815538168.7101089,  -805543247.7590245,  -213668210.4341381,  -734828450.688499,
    -718181825.0574478,  -646533731.4391378});

  // All point coordinates are confined in [0, 1e9] inverval
  auto d_linestring_points_x = rmm::device_vector<T>(
    std::vector<T>{409740620.3189557,  136924984.5824146,  380403413.55319613, 870222466.1573819,
                   635018282.0970114,  525659618.01626,    571907477.7659363,  275007048.105586,
                   830765437.1530899,  659134318.1404505,  800981068.470331,   680105033.0209961,
                   409178513.8582775,  615319073.4289713,  328002044.8817617,  96745319.26689473,
                   246669282.40088674, 653482696.9295893,  189692008.0585241,  221621266.0554327,
                   511285142.46078247, 746304565.6457752,  151485481.01357278, 429159960.53841984,
                   938717939.99254,    57965845.38754873,  108834370.85326725, 594750407.8840696,
                   434994108.54894835, 56391721.360846415, 645661801.503712,   504821682.21993315,
                   209292586.5335849,  991198852.3629916,  938279813.1757035,  157868646.28264025,
                   192523013.1309204,  976120611.6006007,  143168293.4606015,  60348568.63084976,
                   181462906.4053826,  509151158.3184885,  279075603.0482738,  353938665.3946575,
                   683153524.9396951,  952756047.2879049,  677047896.6536566,  232576679.48775858,
                   925263588.8144358,  900946919.20675,    393430839.26884806, 177532669.9589867,
                   139099616.62440377, 753673190.3084364,  536599057.9678925,  217279045.11663502,
                   97054374.5361881,   534574220.50445724, 386604340.3392309,  879773316.2370002,
                   848032517.1589513,  638089254.7692642,  411005671.4264876,  108179694.60664795,
                   532956641.9580039,  821611105.193786,   776236296.6507373,  232549086.2229122,
                   142101776.38994443, 451643918.847285,   350002116.262242,   229208547.96908158,
                   646267608.4942452,  789155145.5078769,  883865809.2755675,  226221484.01556912,
                   793587065.0798844,  54967582.37802618,  187138679.50805324, 775590407.9474832,
                   135989009.40060416, 780614917.2842969,  454631779.6033516,  936928711.1147215,
                   583296498.5783049,  41124718.30773839,  2612098.6261324864, 533287411.5273525,
                   599126810.7859919,  385408444.91818047, 650852421.3350519,  918333432.9809265,
                   41455862.876842834, 893536323.9715917,  930233356.6986674,  82076970.25425225,
                   921394413.2421083,  531871141.19711286, 812913039.2377981,  61778947.21104521,
                   150507958.42193225, 894230228.012308,   699267079.8411367,  378043880.1254338,
                   566516979.8955611,  439431893.5233404,  828847798.6196955,  664400011.6943698,
                   875783995.1035005,  852148423.9637662,  147581346.26651093, 162904664.51064795,
                   620158731.8762205,  578412532.4345315,  613931731.6926718,  397905750.1289512,
                   674105283.0624243,  691243372.6658074,  370822640.1277301,  909036.3036469707,
                   949239285.9062591,  963122476.4149648,  421535309.9500861,  711629736.7364626,
                   199261861.3102121,  874851707.9150648,  749167264.8429024,  456000942.25790817,
                   675640479.8157237,  965917249.7607529,  26628655.839953054, 764316890.7849469,
                   902779618.5114611,  451457408.4987492,  790995118.6806517,  973566429.7698244,
                   290217518.080605,   3804982.336973961,  15589726.230492985, 192295745.74101478,
                   553017580.5420407,  69666843.10205656,  401190849.0547859,  235092974.15243518,
                   196141450.5531844,  161761722.11740464, 728059691.1791501,  784625776.5375279,
                   333201303.33409643, 240273600.7783876});

  auto d_linestring_points_y = rmm::device_vector<T>(
    std::vector<T>{427088748.262282,   15846326.456916582, 263715067.57081458, 744521879.19594,
                   572770812.5565724,  691186865.4068167,  336337240.689984,   787686206.7076751,
                   147487131.60595277, 84428241.31664456,  506921603.98541343, 193217645.2628232,
                   718535416.0410438,  90325588.48195876,  234506697.7554477,  850832804.3524027,
                   503598932.3183143,  962972968.0657562,  113138659.1840286,  336655731.1768273,
                   671618509.8323002,  858648173.8207241,  507617389.6212197,  515313326.084698,
                   491995133.5594163,  490624417.4986565,  620579501.7069384,  465909962.54382086,
                   312139650.71542287, 732445999.3335835,  999079375.4649274,  634807865.7394242,
                   212942895.61936525, 105492654.61970109, 686161246.238137,   839961597.175528,
                   824799270.3078759,  394023691.49011356, 894475911.9159102,  22452373.71468395,
                   883781875.3838649,  183451947.0224278,  940364735.2695354,  564815551.2642368,
                   203299616.9138765,  590083349.1478146,  500123660.0675716,  261576815.05937818,
                   419357654.0144578,  789278512.782339,   984928672.3111312,  206622832.8934326,
                   422073349.62365746, 426511779.26089364, 929811834.6504029,  694638504.9669654,
                   598958469.9031045,  637417308.2679808,  769572394.6288227,  879155002.1181698,
                   588687325.6835386,  757505254.7010162,  669195230.5654877,  452009740.5227253,
                   637837010.3060563,  668640848.2901171,  744941096.5359102,  425691025.0965489,
                   500664385.30749345, 675284105.7840127,  940718458.5428201,  396721894.66349375,
                   319201869.6257737,  617319544.2840747,  979660334.0198653,  554680943.7365212,
                   116769752.13308178, 278644063.3398319,  414669044.9874066,  759139950.8286334,
                   222638345.06086627, 911569572.8776335,  3739644.187957747,  455188819.3986085,
                   426177019.23666626, 677555098.262449,   462423923.1447131,  436326652.64049494,
                   119091589.96316288, 205891298.3275461,  177200062.0255138,  168525449.77325585,
                   984524410.4859962,  285100127.40400785, 340670092.2047182,  18182814.87563117,
                   78958640.30545191,  633971422.6006465,  814560194.5223289,  63861631.53716002,
                   556467184.4507445,  172789265.41557,    928439950.9482422,  809733911.8071963,
                   917311909.3598588,  606668843.36556,    809359300.8301904,  321459748.2580165,
                   842158368.8928964,  407998743.0353185,  626388442.3813977,  125264282.61080475,
                   511955140.65405303, 109662861.1176253,  865816459.5888377,  472872432.24885875,
                   414422826.37450093, 392141641.5915819,  804799056.8789232,  63461043.991374254,
                   730373213.8349088,  366324066.52896893, 886705692.0330912,  445545117.1188059,
                   462584973.93963146, 379183376.5660725,  830193863.8858793,  732573393.1503409,
                   932616236.1341246,  619437904.6852235,  663523018.0059164,  736936521.9745129,
                   296362801.44101405, 983012880.341103,   34357531.17606649,  148685139.51044032,
                   972635905.237423,   841390202.4023547,  128629767.16073488, 394167494.25139624,
                   325307611.81316274, 832766758.5649326,  57757175.60589072,  382309069.6748021,
                   354860973.2555975,  353267590.8484059,  871415919.5619106,  308110516.7161067,
                   781431191.6862057,  94534109.24270672});

  auto expect =
    std::vector<T>{1028683552.5484606, 1281367568.6299243, 1728531284.434097,  1183181212.148498,
                   1720208610.9127815, 1739388743.8832421, 1022476436.0970014, 1425259138.1953902,
                   890950954.4746181,  1722982654.7872393, 772903352.8770543,  1372896620.4566827,
                   1076690699.4901886, 1114072375.6856866, 1510822159.994183,  783256901.0141131,
                   1671079606.6047292, 991863758.0669209,  1143882472.7015646, 1753683481.815972,
                   1732763066.3699694, 1237341019.767269,  945133564.5748928,  1328400986.2592182,
                   1700853690.6191084, 1039987633.4181526, 866705395.0630785,  764248258.2826442,
                   1128498689.8016698, 1152582463.5701518, 1485887458.976279,  1029968634.4905895,
                   1967269282.7638612, 353227756.882895,   2139019958.7739916, 1545056807.1308582,
                   1757886517.1483865, 1138201205.2776566, 1157084200.2992508, 990839153.0335766,
                   1656925277.2401264, 843982796.3074187,  1365288526.7371252, 1704852375.309083,
                   1424632154.129706,  1642663437.0646927, 440103049.49988014, 1607161773.8117433,
                   1520123653.1998143, 1027033461.3751863};

  auto num_pairs             = d_points_x.size();
  auto num_linestring_points = d_linestring_points_x.size();

  auto point_geometries = thrust::make_counting_iterator(0);
  auto points           = make_vec_2d_iterator(d_points_x.begin(), d_points_y.begin());

  auto linestring_geometries = thrust::make_counting_iterator(0);
  auto linestring_parts =
    cuspatial::detail::make_counting_transform_iterator(0, times_three_functor<int32_t>{});
  auto linestring_points =
    make_vec_2d_iterator(d_linestring_points_x.begin(), d_linestring_points_y.begin());

  auto multipoints = make_multipoint_range(num_pairs, point_geometries, num_pairs, points);

  auto multilinestrings = make_multilinestring_range(num_pairs,
                                                     linestring_geometries,
                                                     num_pairs,
                                                     linestring_parts,
                                                     num_linestring_points,
                                                     linestring_points);

  rmm::device_vector<T> got(d_points_x.size());

  auto ret = pairwise_point_linestring_distance(multipoints, multilinestrings, got.begin());

  expect_vector_equivalent(expect, got);
  EXPECT_EQ(ret, got.end());
}

TYPED_TEST(PairwisePointLinestringDistanceTest, OnePairMultiPointMultiLinestring)
{
  using T       = TypeParam;
  using CartVec = std::vector<vec_2d<T>>;

  std::vector<int> point_geometry_offsets{0, 3};
  CartVec points{{0, 1}, {2, 3}, {4, 5}};
  std::vector<int> linestring_geometry_offsets{0, 3};
  std::vector<int> linestring_part_offsets{0, 3, 6, 8};
  CartVec linestring_points{
    {0, -1}, {-2, -3}, {-4, -5}, {-5, -6}, {7, 8}, {8, 9}, {9, 10}, {10, 11}};
  thrust::host_vector<T> expect(std::vector<T>{0.32539568672798425});

  rmm::device_vector<int> d_point_geometries(point_geometry_offsets);
  rmm::device_vector<vec_2d<T>> d_points(points);
  rmm::device_vector<int> d_linestring_geometries(linestring_geometry_offsets);
  rmm::device_vector<int> d_linestring_parts(linestring_part_offsets);
  rmm::device_vector<vec_2d<T>> d_linestring_points(linestring_points);

  auto multipoints = make_multipoint_range(
    d_point_geometries.size() - 1, d_point_geometries.begin(), d_points.size(), d_points.begin());

  auto multilinestrings = make_multilinestring_range(d_linestring_geometries.size() - 1,
                                                     d_linestring_geometries.begin(),
                                                     d_linestring_parts.size() - 1,
                                                     d_linestring_parts.begin(),
                                                     d_linestring_points.size(),
                                                     d_linestring_points.begin());

  rmm::device_vector<T> got(point_geometry_offsets.size() - 1);

  auto ret = pairwise_point_linestring_distance(multipoints, multilinestrings, got.begin());

  expect_vector_equivalent(expect, got);
  EXPECT_EQ(ret, got.end());
}

TYPED_TEST(PairwisePointLinestringDistanceTest, ThreePairMultiPointMultiLinestring)
{
  using T       = TypeParam;
  using CartVec = std::vector<vec_2d<T>>;

  std::vector<int> point_geometry_offsets{0, 1, 3, 5};
  CartVec points{{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}};
  std::vector<int> linestring_geometry_offsets{0, 2, 3, 5};
  std::vector<int> linestring_part_offsets{0, 2, 4, 6, 8};
  CartVec linestring_points{
    {0, -1}, {-2, -3}, {-4, -5}, {-5, -6}, {7, 8}, {8, 9}, {9, 10}, {10, 11}};
  thrust::host_vector<T> expect(std::vector<T>{2.0, 4.242640687119285, 1.4142135623730951});

  rmm::device_vector<int> d_point_geometries(point_geometry_offsets);
  rmm::device_vector<vec_2d<T>> d_points(points);
  rmm::device_vector<int> d_linestring_geometries(linestring_geometry_offsets);
  rmm::device_vector<int> d_linestring_parts(linestring_part_offsets);
  rmm::device_vector<vec_2d<T>> d_linestring_points(linestring_points);

  auto multipoints = make_multipoint_range(
    d_point_geometries.size() - 1, d_point_geometries.begin(), d_points.size(), d_points.begin());

  auto multilinestrings = make_multilinestring_range(d_linestring_geometries.size() - 1,
                                                     d_linestring_geometries.begin(),
                                                     d_linestring_parts.size() - 1,
                                                     d_linestring_parts.begin(),
                                                     d_linestring_points.size(),
                                                     d_linestring_points.begin());

  rmm::device_vector<T> got(point_geometry_offsets.size() - 1);

  auto ret = pairwise_point_linestring_distance(multipoints, multilinestrings, got.begin());

  expect_vector_equivalent(expect, got);
  EXPECT_EQ(ret, got.end());
}

}  // namespace test
}  // namespace cuspatial
