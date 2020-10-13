import os
import cudf

import numpy as np

from get_time import BenchmarkTimer
from point_in_polygon import cpu_points_in_polygon, \
    points_in_polygon
from urllib.request import urlretrieve


def data_progress_hook(block_number, read_size, total_filesize):
    if (block_number % 1000) == 0:
        perc_downloaded = 100 * (block_number * read_size) / total_filesize
        print(
            f" > percent complete: {perc_downloaded:.2f}\r",
            end="",
        )
    return 0


def download_dataset(url, file_name, data_dir):
    check_file = os.path.join(data_dir, file_name)

    if os.path.isfile(check_file):
        print(f" > File already exists. Ready to load at {file_name}")
    else:
        # Ensure folder exists
        os.makedirs(data_dir, exist_ok=True)

        urlretrieve(
            url=url,
            filename=check_file,
            reporthook=data_progress_hook,
        )

        print(f" > Download complete {file_name}")


def load_dataset(data_dir):
    tzone_name = 'its_4326_roi.shp'
    # TODO : We need to have the benchmark dataset in a place from where
    # anyone can download it
    # zone_url = "https://data.cityofnewyork.us/api/geospatial\
    #             /d3c5-ddgc?method=export&format=GeoJSON"

    dataset_name = 'taxi2015.csv'
    dataset_url = ("https://s3.amazonaws.com/nyc-tlc/"
                   "trip+data/yellow_tripdata_2015-01.csv")

    # check and download the json file containing the zone information
    # download_dataset(zone_url, tzone_name, data_dir)
    # check and download the taxi dataset
    download_dataset(dataset_url, dataset_name, data_dir)

    taxi_data_path = os.path.join(data_dir, dataset_name)
    taxi_dataset = cudf.read_csv(taxi_data_path)

    tzones_info_file = os.path.join(data_dir, tzone_name)

    return taxi_dataset[0:1300000], tzones_info_file


class SpeedComparison:

    def __init__(self,
                 n_reps=1,
                 data_dir=None):
        self.n_reps = n_reps
        self.data_dir = data_dir

    def run_algos(self,
                  run_algos=["point_in_polygon",
                             "haversine_distance"],
                  run_cpu=False,
                  compare_vals=False):
        taxi_dataset, tzones_info_file = load_dataset(self.data_dir)
        results = []

        if "point_in_polygon" in run_algos:
            polygon_timer = BenchmarkTimer(self.n_reps)
            for rep in polygon_timer.benchmark_runs():
                cuspatial_vals = points_in_polygon(taxi_dataset,
                                                   tzones_info_file)
            cu_polygon_time = np.min(polygon_timer.timings)

            if run_cpu:
                cpu_polygon_timer = BenchmarkTimer(self.n_reps)
                for rep in cpu_polygon_timer.benchmark_runs():
                    cpu_vals = cpu_points_in_polygon(taxi_dataset,
                                                     tzones_info_file)
                cpu_polygon_time = np.min(cpu_polygon_timer.timings)
                results.append({"algo": "point_in_polygon",
                                "cuspatial_time": cu_polygon_time,
                                "cpu_time": cpu_polygon_time})

                if compare_vals:
                    print(np.array_equal(cuspatial_vals.data.to_array(),
                                         cpu_vals))
            else:
                results.append({"algo": "point_in_polygon",
                                "cuspatial_time": cu_polygon_time})

        return results
        """
        if "haversine_distance" in run_algos:

            haversine_timer = BenchmarkTimer(n_reps)
            for rep in polygon_timer.benchmark_runs():
                points_in_polygon(taxi_dataset, tzones_info_file)
            cuspatial_haversine = np.min(polygon_timer.timings)

            if run_cpu:
                cpu_haversine_timer = BenchmarkTimer(n_reps)
                for rep in cpu_polygon_timer.benchmark_runs():
                    cpu_haversine_distance(taxi_dataset)
                cpu_haversine = np.min(cpu_haversine_timer.timings)
        """
