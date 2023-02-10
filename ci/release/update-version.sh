#!/bin/bash
#############################
# cuSpatial Version Updater #
#############################

## Usage
# bash update-version.sh <new_version>


# Format is YY.MM.PP - no leading 'v' or trailing 'a'
NEXT_FULL_TAG=$1

# Get current version
CURRENT_TAG=$(git tag --merged HEAD | grep -xE '^v.*' | sort --version-sort | tail -n 1 | tr -d 'v')
CURRENT_MAJOR=$(echo $CURRENT_TAG | awk '{split($0, a, "."); print a[1]}')
CURRENT_MINOR=$(echo $CURRENT_TAG | awk '{split($0, a, "."); print a[2]}')
CURRENT_PATCH=$(echo $CURRENT_TAG | awk '{split($0, a, "."); print a[3]}')
CURRENT_SHORT_TAG=${CURRENT_MAJOR}.${CURRENT_MINOR}

#Get <major>.<minor> for next version
NEXT_MAJOR=$(echo $NEXT_FULL_TAG | awk '{split($0, a, "."); print a[1]}')
NEXT_MINOR=$(echo $NEXT_FULL_TAG | awk '{split($0, a, "."); print a[2]}')
NEXT_SHORT_TAG=${NEXT_MAJOR}.${NEXT_MINOR}

echo "Preparing release $CURRENT_TAG => $NEXT_FULL_TAG"

# Inplace sed replace; workaround for Linux and Mac
function sed_runner() {
    sed -i.bak ''"$1"'' $2 && rm -f ${2}.bak
}

# python/cpp update
sed_runner 's/'"CUSPATIAL VERSION .* LANGUAGES"'/'"CUSPATIAL VERSION ${NEXT_FULL_TAG} LANGUAGES"'/g' cpp/CMakeLists.txt
sed_runner 's/'"cuspatial_version .*)"'/'"cuspatial_version ${NEXT_FULL_TAG})"'/g' python/cuspatial/CMakeLists.txt

# RTD update
sed_runner 's/version = .*/version = '"'${NEXT_SHORT_TAG}'"'/g' docs/source/conf.py
sed_runner 's/release = .*/release = '"'${NEXT_FULL_TAG}'"'/g' docs/source/conf.py

# rapids-cmake version
sed_runner 's/'"branch-.*\/RAPIDS.cmake"'/'"branch-${NEXT_SHORT_TAG}\/RAPIDS.cmake"'/g' fetch_rapids.cmake
sed_runner 's/'"branch-.*\/RAPIDS.cmake"'/'"branch-${NEXT_SHORT_TAG}\/RAPIDS.cmake"'/g' python/cuspatial/CMakeLists.txt

# bump cudf
for FILE in dependencies.yaml conda/environments/*.yaml; do
  sed_runner "s/cudf=${CURRENT_SHORT_TAG}/cudf=${NEXT_SHORT_TAG}/g" ${FILE};
  sed_runner "s/rmm=${CURRENT_SHORT_TAG}/rmm=${NEXT_SHORT_TAG}/g" ${FILE};
  sed_runner "s/libcudf=${CURRENT_SHORT_TAG}/libcudf=${NEXT_SHORT_TAG}/g" ${FILE};
  sed_runner "s/librmm=${CURRENT_SHORT_TAG}/librmm=${NEXT_SHORT_TAG}/g" ${FILE};
done

# Doxyfile update
sed_runner "/PROJECT_NUMBER[ ]*=/ s|=.*|= ${NEXT_FULL_TAG}|g" cpp/doxygen/Doxyfile
sed_runner "/TAGFILES/ s|[0-9]\+.[0-9]\+|${NEXT_SHORT_TAG}|g" cpp/doxygen/Doxyfile

for FILE in .github/workflows/*.yaml; do
  sed_runner "/shared-action-workflows/ s/@.*/@branch-${NEXT_SHORT_TAG}/g" "${FILE}"
done
