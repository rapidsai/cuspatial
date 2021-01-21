import argparse

from benchmark_helpers import SpeedComparison

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='run_benchmarks',
        description=r'''
        Command-line benchmark runner, logging results to
        stdout and/or CSV.
        Examples:
          # Run all benchmarks to time only cuspatial
          python run_benchmarks.py --data_dir python/cuspatial/benchmarks/data
          # Run all benchmarks and compare their timings with their
          # CPU equivalent
          python run_benchmarks.py \
                --data_dir python/cuspatial/benchmarks/data \
                --run_cpu True
          # Run point_in_polygon algorithm and compare their timings and values
          # with their CPU equivalent
          python run_benchmarks.py \
                --data_dir python/cuspatial/benchmarks/data \
                --run_algos='point_in_polygon' \
                --run_cpu True --compare_values True
        ''',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--run_cpu',
        default=True,
        help='Run the cpu equivalent algorithm from shapely.'
    )
    parser.add_argument(
        '--compare_values',
        default=False,
        help='Run the cpu equivalent algorithm from shapely'
             ' and compare the results with the cuspatial run.',
    )
    parser.add_argument(
        '--data_dir',
        default='python/cuspatial/benchmarks/data',
        help='Path to dataset',
    )
    parser.add_argument(
        '--n_reps',
        type=int,
        default=1,
        help='Number of times to run the benchmarks',
    )
    parser.add_argument(
        '--run_algos',
        default=["point_in_polygon",
                 "haversine_distance"],
        help='List of string containg the different algorithms to'
             ' be benchmarked',
    )

    args = parser.parse_args()
    runner = SpeedComparison(n_reps=args.n_reps,
                             data_dir=args.data_dir)

    results = runner.run_algos(run_algos=args.run_algos,
                               run_cpu=args.run_cpu,
                               compare_vals=args.compare_values)
    print(results)
