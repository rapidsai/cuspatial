set -e

echo "Building libcuspatial"
gpuci_conda_retry build conda/recipes/libcuspatial

