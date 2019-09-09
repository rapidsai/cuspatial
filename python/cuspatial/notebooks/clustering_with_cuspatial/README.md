# Trajectory Clustering Example with CUSPATIAL

[CUSPATIAL](https://github.com/rapidsai/cuspatial) is a C++ library with python bindings that is based on [CUDA](https://developer.nvidia.com/cuda-zone) and [CUDF](https://github.com/rapidsai/cudf).
CUSPATIAL provides significant GPU acceleration to common spatial and spatio-
temporal operations such as point in polygon, distance between trajectories, and
trajectory clustering. The speed-up ranges from 10x to 10000x for different
operations.

This repo shows an example of accelerating a clustering problem using CUSPATIAL on a real-world vehicle trajectory dataset. On this particular use case, CUSPATIAL brings the end-2-end computation time from around 11 hours down to less than 15 seconds. 

[//]: # (Image References)

[AC]: ./images/AC.png "AC"
[DB]: ./images/DB.png "DB"
[LOCUST_HILL_EB]: ./images/LOCUST_HILL_EB.png "LOCUST_HILL_EB"
[jitter_effect]: ./output_images/jitter_effect.png "jitter_effect"
[moviepy_saveclip]: ./images/moviepy_saveclip.png "moviepy_saveclip"

## Problem Statement - Trajectory Clustering 

The trajectory clustering task is grouping a set of trajectories in such a way that trajectories in the same group are more similar to each other than to those in other groups. It is useful for various problems such as motion pattern extraction, behavior analysis and more.

The clustering task consists of two major components in general:

1. Similarity Metric
2. Searching Algorithm

Given a specific similarity metric, different clustering algorithm will have different searching mechanism and different complexity. But in some case people may want to precompute all pairwise similarities. One of the many reasons could be people may need to perform multiple iterations of hyperparameter search for the clustering algorithm to get a good result and we don't want to redo the time-consuming similarity computation each time. 

In this trajectory clustering example, we work on a particular vehicle trajectory dataset. Below is a quick summary stats of the dataset (More description of this dataset can be found in the following section):

| num_trajectories | 10070 |
|------------------|-------|
| max_length       | 961   |
| min_length       | 30    |
| avg_length       | 87    |

We use `Hausdorff distance` as the similarity metric between trajectories. We compute the pairwise similarity matrix and apply AgglomerativeClustering (AC) and DBSCAN afterwards.

In this example, CUSPATIAL can significantly accelerate the computation of the `Hausdorff distance` similarity matrix. Comparing to the typical scikit-learn implementation of `Hausdorff distance` (single CPU thread), CUSPATICAL reduces the computation time from about 11 hours to 7.9 seconds on this particular dataset.

Since we pre-computed the similarity matrix, we can experiment with different clustering algorithm and different hyperparameter sets much easier. In this example we do it in an interactive way using ipython widgets, you may play with the hyperparameter set and see how the clustering result responds. Below shows one example each for AC and DBSCAN:  

![alt text][AC]

![alt text][DB]

## Dataset Used

The real-world trajectory dataset we used here for this example is collected from a traffic intersection located in Dubuque, Iowa. We applied the following steps to extract vehicle trajectories from multiple cameras implemented at this intersection:

* Detect vehicle locations in camera domain using AI based detectors;
* Apply tracking algorithm to assign the same id to the same vehicle;
* Project the vehicle location from camera domain to latitudes and longitudes;
* Create trajectories based on vehicle id. 

A python pickle file (created using python 3.7) of this dataset can be downloaded [here](https://drive.google.com/file/d/1GE-_z9HgLp3eV7Lgo_KOl53QuMgiCUMS/view?usp=sharing). 


## Install Dependencies

1. cuSPATIAL: [instructions](https://github.com/rapidsai/cuspatial/blob/branch-0.10/README.md#compile--install-c-backend)
2. scikit-image: 	`pip install scikit-image` (DO NOT use conda install)
3. opencv: 			`conda install -c conda-forge opencv`
4. ipywidgets: 		`conda install -c anaconda ipywidgets`
5. scikit-learn: 	`conda install -c anaconda scikit-learn`
