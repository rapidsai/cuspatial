set -e

echo "Building cuspatial"
gpuci_conda_retry build conda/recipes/cuspatial --python=$PYTHON

