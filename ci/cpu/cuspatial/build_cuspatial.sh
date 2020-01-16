set -e

echo "Building cuspatial"
conda build conda/recipes/cuspatial --python=$PYTHON

