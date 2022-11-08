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

#include <cuspatial_test/random.cuh>

#include <cuspatial/detail/iterator.hpp>
#include <cuspatial/error.hpp>
#include <cuspatial/experimental/iterator_factory.cuh>
#include <cuspatial/experimental/point_distance.cuh>
#include <cuspatial/experimental/ranges/multipoint_range.cuh>
#include <cuspatial/vec_2d.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/device_vector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/normal_distribution.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <numeric>

namespace cuspatial {

/**
 * @brief Generate `num_points` points on device
 */
template <typename T>
struct PairwisePointDistanceTest : public ::testing::Test {
  rmm::device_vector<vec_2d<T>> generate_random_points(
    std::size_t num_points,
    std::size_t seed,
    rmm::cuda_stream_view stream = rmm::cuda_stream_default)
  {
    auto engine  = deterministic_engine(0);
    auto uniform = make_normal_dist<T>(0.0, 1.0);
    auto pgen    = point_generator(T{0.0}, T{1.0}, engine, uniform);
    rmm::device_vector<vec_2d<T>> points(num_points);
    auto counting_iter = thrust::make_counting_iterator(seed);
    thrust::transform(
      rmm::exec_policy(stream), counting_iter, counting_iter + num_points, points.begin(), pgen);
    return points;
  }

  /**
   * @brief Generate `num_multipoints` multipoint, returns offset and point vectors on device
   */
  std::pair<rmm::device_vector<std::size_t>, rmm::device_vector<vec_2d<T>>>
  generate_random_multipoints(std::size_t num_multipoints,
                              std::size_t max_points_per_multipoint,
                              std::size_t seed,
                              rmm::cuda_stream_view stream = rmm::cuda_stream_default)
  {
    std::vector<std::size_t> offset(num_multipoints + 1, 0);
    std::generate_n(offset.begin() + 1, num_multipoints, [max_points_per_multipoint]() {
      return std::rand() % max_points_per_multipoint;
    });
    std::inclusive_scan(offset.begin(), offset.end(), offset.begin());
    std::size_t num_points = offset.back();
    auto points            = generate_random_points(num_points, seed, stream);
    return {offset, points};
  }
};

/**
 * @brief Computes point distances on host
 *
 * @note Implicitly copies input vectors to host
 */
template <typename Cart2DVec>
auto compute_point_distance_host(Cart2DVec const& point1, Cart2DVec const& point2)
{
  using Cart2D = typename Cart2DVec::value_type;
  using T      = typename Cart2D::value_type;
  thrust::host_vector<Cart2D> h_point1(point1);
  thrust::host_vector<Cart2D> h_point2(point2);
  auto pair_iter =
    thrust::make_zip_iterator(thrust::make_tuple(h_point1.begin(), h_point2.begin()));
  auto result_iter = thrust::make_transform_iterator(pair_iter, [](auto p) {
    auto p0 = thrust::get<0>(p);
    auto p1 = thrust::get<1>(p);
    return std::sqrt(dot(p0 - p1, p0 - p1));
  });

  return thrust::host_vector<T>(result_iter, result_iter + point1.size());
}

/**
 * @brief Computes multipoint distances on host.
 *
 * @note Implicitly copies input vectors to host.
 * @note This function also tests the compatibility of `multipoint_range` on host.
 */
template <typename OffsetVec, typename Cart2DVec>
auto compute_multipoint_distance_host(OffsetVec const& lhs_offset,
                                      Cart2DVec const& lhs_points,
                                      OffsetVec const& rhs_offset,
                                      Cart2DVec const& rhs_points)
{
  using Cart2D    = typename Cart2DVec::value_type;
  using IndexType = typename OffsetVec::value_type;
  using T         = typename Cart2D::value_type;

  auto num_results = lhs_offset.size() - 1;
  thrust::host_vector<IndexType> h_offset1(lhs_offset);
  thrust::host_vector<Cart2D> h_point1(lhs_points);
  thrust::host_vector<IndexType> h_offset2(rhs_offset);
  thrust::host_vector<Cart2D> h_point2(rhs_points);

  auto h_multipoint_array1 =
    multipoint_range{h_offset1.begin(), h_offset1.end(), h_point1.begin(), h_point1.end()};
  auto h_multipoint_array2 =
    multipoint_range{h_offset2.begin(), h_offset2.end(), h_point2.begin(), h_point2.end()};

  std::vector<T> result(num_results, 0);

  std::transform(h_multipoint_array1.multipoint_begin(),
                 h_multipoint_array1.multipoint_end(),
                 h_multipoint_array2.multipoint_begin(),
                 result.begin(),
                 [](auto const& mp1, auto const& mp2) {
                   T min_distance_squared = std::numeric_limits<T>::max();
                   for (vec_2d<T> const& p1 : mp1)
                     for (vec_2d<T> const& p2 : mp2) {
                       T distance_squared   = dot((p1 - p2), (p1 - p2));
                       min_distance_squared = min(min_distance_squared, distance_squared);
                     }

                   return std::sqrt(min_distance_squared);
                 });
  return result;
}

using TestTypes = ::testing::Types<float, double>;

TYPED_TEST_CASE(PairwisePointDistanceTest, TestTypes);

TYPED_TEST(PairwisePointDistanceTest, Empty)
{
  using T         = TypeParam;
  using Cart2D    = vec_2d<T>;
  using Cart2DVec = std::vector<Cart2D>;

  rmm::device_vector<int32_t> multipoint_geom1(std::vector<int32_t>{0});
  rmm::device_vector<Cart2D> points1{};
  rmm::device_vector<int32_t> multipoint_geom2(std::vector<int32_t>{0});
  rmm::device_vector<Cart2D> points2{};

  rmm::device_vector<T> expected{};
  rmm::device_vector<T> got(points1.size());

  auto multipoint_1 = multipoint_range{
    multipoint_geom1.begin(), multipoint_geom1.end(), points1.begin(), points1.end()};
  auto multipoint_2 = multipoint_range{
    multipoint_geom2.begin(), multipoint_geom2.end(), points2.begin(), points2.end()};

  auto ret_it = pairwise_point_distance(multipoint_1, multipoint_2, got.begin());

  test::expect_vector_equivalent(expected, got);
  EXPECT_EQ(expected.size(), std::distance(got.begin(), ret_it));
}

TYPED_TEST(PairwisePointDistanceTest, OnePairSingleComponent)
{
  using T         = TypeParam;
  using Cart2D    = vec_2d<T>;
  using Cart2DVec = std::vector<Cart2D>;

  std::size_t constexpr num_pairs = 1;
  auto multipoint_geom1           = thrust::make_counting_iterator(0);
  rmm::device_vector<Cart2D> points1{Cart2DVec{{1.0, 1.0}}};
  auto multipoint_geom2 = thrust::make_counting_iterator(0);
  rmm::device_vector<Cart2D> points2{Cart2DVec{{0.0, 0.0}}};

  rmm::device_vector<T> expected{std::vector<T>{std::sqrt(T{2.0})}};
  rmm::device_vector<T> got(points1.size());

  auto multipoint_1 = multipoint_range{
    multipoint_geom1, multipoint_geom1 + num_pairs + 1, points1.begin(), points1.end()};
  auto multipoint_2 = multipoint_range{
    multipoint_geom2, multipoint_geom2 + num_pairs + 1, points2.begin(), points2.end()};

  auto ret_it = pairwise_point_distance(multipoint_1, multipoint_2, got.begin());

  test::expect_vector_equivalent(expected, got);
  EXPECT_EQ(expected.size(), std::distance(got.begin(), ret_it));
}

TYPED_TEST(PairwisePointDistanceTest, SingleComponentManyRandom)
{
  using T         = TypeParam;
  using Cart2D    = vec_2d<T>;
  using Cart2DVec = std::vector<Cart2D>;

  std::size_t constexpr num_pairs = 1000;

  auto multipoint_geom1 = thrust::make_counting_iterator(0);
  auto points1          = this->generate_random_points(num_pairs, 0);
  auto multipoint_geom2 = thrust::make_counting_iterator(0);
  auto points2          = this->generate_random_points(num_pairs, num_pairs);

  auto expected = compute_point_distance_host(points1, points2);
  rmm::device_vector<T> got(points1.size());

  auto multipoint_1 =
    make_multipoint_range(num_pairs, multipoint_geom1, points1.size(), points1.begin());
  auto multipoint_2 =
    make_multipoint_range(num_pairs, multipoint_geom2, points2.size(), points2.begin());

  auto ret_it = pairwise_point_distance(multipoint_1, multipoint_2, got.begin());
  thrust::host_vector<T> hgot(got);

  test::expect_vector_equivalent(hgot, expected);
  EXPECT_EQ(expected.size(), std::distance(got.begin(), ret_it));
}

TYPED_TEST(PairwisePointDistanceTest, SingleComponentCompareWithShapely)
{
  using T         = TypeParam;
  using Cart2D    = vec_2d<T>;
  using Cart2DVec = std::vector<Cart2D>;

  std::vector<T> x1{
    -12.309831056315302, -7.927059559371418,  -49.95705839647165,  -1.0512464476733485,
    -89.39777525663895,  -32.460148393873666, -20.64749623324501,  74.88373211296442,
    -3.566633537053898,  -91.4320392524529,   1.68283845329249,    30.90993923507801,
    2.5208716416609267,  -47.13990142514067,  -89.60387010381702,  15.799301259524867,
    -22.8887289692815,   81.6430657985936,    28.324072604115624,  -43.3201792789866,
    31.15072850958005,   -90.9256331022774,   -17.077973750390452, -88.54243712973691,
    -83.67679143413889,  -78.86701538797912,  60.11416346218348,   38.38679261335849,
    86.29202143733288,   90.51425714428673,   -72.13954336543273,  -29.909309579787713,
    -72.27943372189681,  49.182311914851205,  -84.50393600760954,  -94.33250533960667,
    -9.932568319346647,  36.99556837875937,   -24.20862704113279,  -50.442042319693705,
    -59.14098804172897,  30.673225738449172,  48.67403790478693,   -63.207315558126545,
    29.52859942242645,   26.173903500998197,  47.79243983907904,   -99.38850933058964,
    -83.31147453301942,  5.8413331217636255,  -47.87029604603307,  95.82254403897923,
    -55.52829900834991,  74.87973107553039,   -84.05457104705182,  -95.87736100367613,
    -6.480112617573386,  -78.09036923042659,  62.14707651291427,   -43.34499838125344,
    77.42752654240155,   12.530763429172254,  97.98997832862835,   -51.389571363066,
    59.66745813871337,   65.98475889051292,   30.40020778235388,   -49.595509308751595,
    9.930123176564942,   -19.283736893878867, -78.06236247946624,  63.68142858698178,
    79.46252260195803,   54.24426311960122,   30.458402886352822,  70.7820673095687,
    -15.306354680748024, 91.01665772140062,   -32.765892351019666, -72.46623073604916,
    58.863272100444334,  -41.35480445335994,  -61.06943341086172,  81.15104128608479,
    -77.69660768219927,  95.47462923834442,   -97.46155919360085,  -81.54704046899158,
    84.9228534190681,    -16.082575320922533, 52.509864091786355,  63.78396723518307,
    13.605239448412032,  -63.70301611378514,  -63.10763374202178,  -61.108649551804895,
    57.266357913172385,  -46.96569013769979,  -43.636365011489566, -29.306746287827558};
  std::vector<T> y1{
    -18.051875936208862, -72.61500308351708,  -23.919317289360777, 74.04449323147637,
    27.003656419402276,  5.131923603252009,   14.381495553187262,  -44.998590378882795,
    66.15308743061799,   31.82686362809011,   60.19621369406618,   36.02100660419922,
    -18.164297228344505, 23.06381468579426,   -34.39959102364766,  -80.65093614662105,
    -50.66614351982265,  30.696539385917852,  -62.06159838829518,  -55.67574678891346,
    2.2570284921564987,  49.260913129155036,  -69.70290379728544,  -14.168007892316037,
    87.743587998508,     -88.40683092249026,  -78.23312582934655,  18.950081576813904,
    -13.001178290210335, -88.72165572783072,  29.13236030074242,   0.9643364439866353,
    -58.14148269328302,  98.23977047259831,   87.65596263514071,   -68.42627074347195,
    -61.49539737381592,  95.22412232012014,   -71.3663413078797,   -87.93028627383005,
    -63.70741871892348,  1.83023166369769,    -44.184879390345245, -29.212266921068498,
    36.63070498793903,   90.55120945758097,   35.40957933073132,   -53.484664102448285,
    85.05271776288717,   80.18938384135001,   -21.313832146230382, -64.49346600820266,
    -72.18007667511924,  50.73463806168728,   7.319811593578507,   -56.54419097667299,
    -80.58912509276239,  6.9148441008914485,  -22.67913193215382,  75.95466324740005,
    69.60650343179027,   27.61785095385285,   -17.798865714702472, -78.36406107867042,
    6.59132839160077,    64.32222103875719,   55.24725933014744,   -53.49018275541756,
    -71.57964472201111,  -9.671216230543001,  -29.999576747551593, -54.15829040618368,
    29.253521698849028,  57.83102910157538,   76.77316185511351,   -54.755703196886174,
    58.71741301597688,   -89.00648352439477,  -62.572264098389354, 55.118081589496626,
    -72.80219811987917,  56.12298345685937,   -9.073644079329679,  87.3857422229443,
    16.65929971566098,   -91.77505633845232,  -99.4775802747735,   6.657482305470497,
    19.82536215719839,   -22.918311016363912, 30.170484267010387,  83.6666865961853,
    -91.70882742463144,  78.70726479431833,   86.04667133973348,   -83.58460594914955,
    84.27888264842167,   6.374228239422575,   62.58260784755962,   -87.64421055779096};

  std::vector<T> x2{
    -69.89840831561355,   78.8460456024616,    39.85341596822734,   -24.391223974913235,
    13.303395979112231,   -12.113621295331923, 65.76955972393912,   32.88000233887396,
    75.15679902070009,    70.42968479275325,   -70.48373074669782,  -67.41906709787041,
    24.0317463752441,     15.6825064869063,    22.786346338534358,  -20.418849974209763,
    34.82105661248487,    38.24867453316148,   -25.835471974453984, -99.8181927392706,
    89.84785718125181,    92.62449528299297,   -15.692938009982782, 42.32594734729251,
    -60.14762773795758,   74.97034158301297,   49.83345296858048,   -8.799811548418369,
    35.12809596314472,    93.18344995215058,   -94.67426883200939,  52.863378156989384,
    80.55592370229223,    -9.708518300250157,  58.19902373613033,   94.71328595396487,
    -41.956496383879006,  -99.23900353260521,  -96.8820547539014,   -61.540850851797046,
    10.60351610840815,    -86.06663137958869,  -19.76183018904282,  -52.98140516951296,
    -60.77170988936312,   -67.64765557651907,  45.61193823583003,   56.92515530750559,
    -33.35973933318071,   -51.94527984432248,  -14.582250347543601, -96.83073470861669,
    -47.25698648583708,   48.904375839188006,  14.554162511314495,  38.237373081363344,
    -32.7325518620032,    57.537241341535015,  -70.50257367880944,  -83.11435173667108,
    1.3843207970826832,   -61.35647094743536,  43.70708320820875,   -81.93488230360825,
    -53.098660448910465,  70.16656087048054,   0.7197864636628637,  92.59459361315123,
    -77.37226816319428,   -32.66885376463454,  34.32370196646004,   71.72963476414482,
    1.5234779242439433,   3.0626652169396085,  -1.600973288116736,  -1.875116500268692,
    24.115900341387686,   -6.818007491235834,  -37.57206985691543,  46.48919986671669,
    99.81587509298548,    26.961573147884856,  -57.411420876126954, -78.90146907605978,
    37.2322492476274,     67.99231943510561,   64.95985406157519,   -21.195261701977287,
    78.89518238318205,    -95.50952525706322,  76.75637507677297,   -63.30961059551444,
    88.07294705390709,    12.963110252847354,  -59.3400766172247,   18.016669829562915,
    0.024732013514316975, -47.68463698812436,  -16.12846919710843,  57.85570255646779};

  std::vector<T> y2{
    96.98573446222625,   -58.675433421313485, -15.58533007526851,  -14.697644147821276,
    85.96236693008059,   38.92770099339309,   19.791693980620906,  27.483461653596166,
    53.91447892576453,   75.83100042363395,   17.73746513670771,   51.50105094020323,
    33.83904611309756,   -9.59805189545494,   27.567402061211244,  33.72816965802343,
    48.98821930718205,   -14.861794980690213, 0.13287706149869294, 35.05682115680253,
    88.14369170856402,   -20.655621067301244, -36.15962607484525,  23.463908856814932,
    95.93206680397306,   10.936188747304243,  -76.64604957338365,  -44.27118733203363,
    -17.066191002518682, 51.827990165726675,  -55.472330987826744, 82.31391457552668,
    -99.25207116240846,  -8.9622361202783,    -14.764596152666753, 35.51101965248979,
    -7.515215371057382,  -12.734669471901016, -76.18168200736743,  -58.82174033449078,
    -64.55998759489724,  -66.29491004534883,  96.90488209719925,   -42.97997451919843,
    -31.865981559056365, -96.36343702487376,  -84.20827193890962,  26.79428452012931,
    62.912038904465774,  -87.227673692568,    11.2934368901489,    -65.442146916886,
    85.68799018964843,   61.94678236143925,   83.46238187197174,   21.333768673112008,
    61.8718601660381,    -35.70805034839669,  68.43167377857928,   -18.400251392936294,
    25.277688476279536,  -74.94714347783905,  2.391028130810602,   -78.06742777647494,
    73.16329191776757,   -5.425513550228256,  -17.11543472509981,  -21.571671681683625,
    60.95981137578463,   -87.30779120172515,  46.07464276698177,   -26.735186694206213,
    77.34113840661823,   -10.89097657623882,  -7.483005212073712,  -24.163324686785494,
    66.03877277717585,   46.514678630068175,  86.52324722682492,   23.88758093704468,
    32.70460360118328,   47.3873043949026,    -40.72743971179719,  96.60257606822059,
    -93.1284937647867,   -70.26297209791194,  94.52718104748459,   68.27804048047095,
    -74.27404656785302,  -21.16650114972075,  -34.93847763736745,  66.55335171298651,
    -88.44856487882186,  -23.53818606503958,  -29.02780534888051,  -29.346481830318815,
    74.28318391238213,   -38.37789665677865,  56.28623833724116,   -81.09317815145866};

  std::vector<T> expected{
    128.64717656028176, 87.88562670763609,  90.19632281028372,  91.76013021796666,
    118.4215357030851,  39.44788631062081,  86.58624490836462,  83.77327247860025,
    79.6690804001798,   167.7366440763836,  83.73027552297903,  99.54006861093508,
    56.276686562837135, 70.80573751073386,  128.34122090714868, 119.97639069191793,
    115.15820154183437, 62.91768450568626,  82.47065566268454,  106.88509910638807,
    104.02822613477268, 196.4153033352887,  33.57186030542483,  136.17156536458378,
    24.91330426477482,  183.12555244130633, 10.402491013960068, 78.8891909881514,
    51.325155608916646, 140.57498906651185, 87.55436962189877,  116.056329846112,
    158.26789618636312, 122.3127143880106,  175.65336769339257, 215.7342613973661,
    62.764576137605516, 173.82450721651924, 72.83278521088664,  31.152704497923047,
    69.74971493014701,  135.16371248533227, 156.8113160405468,  17.14989841904493,
    113.33993969348232, 209.1400727201678,  119.63772368951071, 175.72328059917774,
    54.63868144260578,  177.1094683844075,  46.59751045319419,  192.6556145241176,
    158.08460123131488, 28.29189388239395,  124.5848038188644,  155.08622923365692,
    144.85966618486546, 142.16736573734084, 160.92578604203126, 102.39361006963875,
    88.02052587541618,  126.40767969785988, 57.91601260573419,  30.546751247397513,
    130.95046326396607, 69.87298439379285,  78.21308650467851,  145.7285720718672,
    158.70858501146054, 78.78197209323828,  135.71261679147338, 28.579717281106852,
    91.58009372078648,  85.68704702725647,  90.14934991196503,  78.83501748640491,
    40.09634022929284,  167.14546691120552, 149.17295612658907, 122.98674172809133,
    113.17597316247064, 68.87263271597341,  31.86446035391618,  160.31767244865998,
    158.94024585517036, 34.900531808173085, 253.01889830507722, 86.25213267010419,
    94.2922665997649,   79.44626620532313,  69.47712008431841,  128.24056985459816,
    74.53904203761351,  127.79603731531678, 115.13613538697125, 95.93013225849919,
    58.10781125509778,  44.75789949605465,  28.21929483784659,  87.40828630126103};

  auto p1_geom = thrust::make_counting_iterator(0);
  auto p2_geom = thrust::make_counting_iterator(0);

  rmm::device_vector<T> dx1(x1), dy1(y1), dx2(x2), dy2(y2);
  rmm::device_vector<T> got(dx1.size());

  auto p1_begin = make_vec_2d_iterator(dx1.begin(), dy1.begin());
  auto p2_begin = make_vec_2d_iterator(dx2.begin(), dy2.begin());

  auto multipoints_1 = make_multipoint_range(dx1.size(), p1_geom, dx1.size(), p1_begin);
  auto multipoints_2 = make_multipoint_range(dx2.size(), p2_geom, dx2.size(), p2_begin);

  auto ret_it = pairwise_point_distance(multipoints_1, multipoints_2, got.begin());

  thrust::host_vector<T> hgot(got);
  test::expect_vector_equivalent(hgot, expected);
  EXPECT_EQ(expected.size(), std::distance(got.begin(), ret_it));
}

TYPED_TEST(PairwisePointDistanceTest, MultiComponentSinglePair)
{
  using T         = TypeParam;
  using Cart2D    = vec_2d<T>;
  using Cart2DVec = std::vector<Cart2D>;

  rmm::device_vector<int32_t> multipoint_geom1(std::vector<int32_t>{0, 3});
  rmm::device_vector<Cart2D> points1(Cart2DVec{{1.0, 1.0}, {2.5, 1.5}, {-0.1, -0.7}});
  rmm::device_vector<int32_t> multipoint_geom2(std::vector<int32_t>{0, 2});
  rmm::device_vector<Cart2D> points2(Cart2DVec{{1.8, 1.3}, {0.3, 0.6}});

  rmm::device_vector<T> expected{std::vector<T>{T{0.7280109889280517}}};
  rmm::device_vector<T> got(multipoint_geom1.size() - 1);

  auto multipoint_1 = multipoint_range{
    multipoint_geom1.begin(), multipoint_geom1.end(), points1.begin(), points1.end()};
  auto multipoint_2 = multipoint_range{
    multipoint_geom2.begin(), multipoint_geom2.end(), points2.begin(), points2.end()};

  auto ret_it = pairwise_point_distance(multipoint_1, multipoint_2, got.begin());

  test::expect_vector_equivalent(expected, got);
  EXPECT_EQ(expected.size(), std::distance(got.begin(), ret_it));
}

TYPED_TEST(PairwisePointDistanceTest, MultiComponentRandom)
{
  using T         = TypeParam;
  using Cart2D    = vec_2d<T>;
  using Cart2DVec = std::vector<Cart2D>;

  std::size_t constexpr num_pairs                 = 1000;
  std::size_t constexpr max_points_per_multipoint = 10;
  auto [mp0_offset, mp0_points] =
    this->generate_random_multipoints(num_pairs, max_points_per_multipoint, 0);
  auto [mp1_offset, mp1_points] =
    this->generate_random_multipoints(num_pairs, max_points_per_multipoint, num_pairs);

  auto expected = compute_multipoint_distance_host(mp0_offset, mp0_points, mp1_offset, mp1_points);
  auto got      = rmm::device_vector<T>(num_pairs);

  auto multipoint_1 =
    multipoint_range{mp0_offset.begin(), mp0_offset.end(), mp0_points.begin(), mp0_points.end()};
  auto multipoint_2 =
    multipoint_range{mp1_offset.begin(), mp1_offset.end(), mp1_points.begin(), mp1_points.end()};

  auto ret_it = pairwise_point_distance(multipoint_1, multipoint_2, got.begin());

  test::expect_vector_equivalent(expected, got);
  EXPECT_EQ(expected.size(), std::distance(got.begin(), ret_it));
}

}  // namespace cuspatial
