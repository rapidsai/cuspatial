set -e

echo "Building libcuspatial"
conda build conda/recipes/libcuspatial --python=$PYTHON

