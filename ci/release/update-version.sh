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
sed_runner 's/'"CUPROJ VERSION .* LANGUAGES"'/'"CUPROJ VERSION ${NEXT_FULL_TAG} LANGUAGES"'/g' cpp/cuproj/CMakeLists.txt
sed_runner 's/'"cuspatial_version .*)"'/'"cuspatial_version ${NEXT_FULL_TAG})"'/g' python/cuspatial/CMakeLists.txt
sed_runner 's/'"cuproj_version .*)"'/'"cuproj_version ${NEXT_FULL_TAG})"'/g' python/cuproj/CMakeLists.txt
sed_runner 's/'"cuproj_version .*)"'/'"cuproj_version ${NEXT_FULL_TAG})"'/g' python/cuproj/cuproj/cuprojshim/CMakeLists.txt

# RTD update
sed_runner 's/version = .*/version = '"'${NEXT_SHORT_TAG}'"'/g' docs/source/conf.py
sed_runner 's/release = .*/release = '"'${NEXT_FULL_TAG}'"'/g' docs/source/conf.py
sed_runner 's/version = .*/version = '"'${NEXT_SHORT_TAG}'"'/g' docs/cuproj/source/conf.py
sed_runner 's/release = .*/release = '"'${NEXT_FULL_TAG}'"'/g' docs/cuproj/source/conf.py

# Centralized version file update
echo "${NEXT_FULL_TAG}" | tr -d '"' > VERSION


# rapids-cmake version
sed_runner 's/'"branch-.*\/RAPIDS.cmake"'/'"branch-${NEXT_SHORT_TAG}\/RAPIDS.cmake"'/g' fetch_rapids.cmake

# Doxyfile update - cuspatial
sed_runner "/PROJECT_NUMBER[ ]*=/ s|=.*|= ${NEXT_FULL_TAG}|g" cpp/doxygen/Doxyfile
sed_runner "/TAGFILES/ s|[0-9]\+.[0-9]\+|${NEXT_SHORT_TAG}|g" cpp/doxygen/Doxyfile

#Doxyfile update - cuproj
sed_runner "/PROJECT_NUMBER[ ]*=/ s|=.*|= ${NEXT_FULL_TAG}|g" cpp/cuproj/doxygen/Doxyfile
sed_runner "/TAGFILES/ s|[0-9]\+.[0-9]\+|${NEXT_SHORT_TAG}|g" cpp/cuproj/doxygen/Doxyfile

# CI files
for FILE in .github/workflows/*.yaml; do
  sed_runner "/shared-action-workflows/ s/@.*/@branch-${NEXT_SHORT_TAG}/g" "${FILE}"
done
sed_runner "s/RAPIDS_VERSION_NUMBER=\".*/RAPIDS_VERSION_NUMBER=\"${NEXT_SHORT_TAG}\"/g" ci/build_docs.sh

# Need to distutils-normalize the original version
NEXT_SHORT_TAG_PEP440=$(python -c "from setuptools.extern import packaging; print(packaging.version.Version('${NEXT_SHORT_TAG}'))")

DEPENDENCIES=(
  cudf
  cuml
  libcudf
  librmm
  rmm
  cuspatial
  cuproj
)

for DEP in "${DEPENDENCIES[@]}"; do
  for FILE in dependencies.yaml conda/environments/*.yaml; do
    sed_runner "/-.* ${DEP}==/ s/==.*/==${NEXT_SHORT_TAG_PEP440}.*/g" ${FILE}
  done
  sed_runner "s/${DEP}==.*\",/${DEP}==${NEXT_SHORT_TAG_PEP440}.*\",/g" python/cuspatial/pyproject.toml
  sed_runner "s/${DEP}==.*\",/${DEP}==${NEXT_SHORT_TAG_PEP440}.*\",/g" python/cuproj/pyproject.toml
done

# Dependency versions in dependencies.yaml
sed_runner "/-cu[0-9]\{2\}==/ s/==.*/==${NEXT_SHORT_TAG_PEP440}.*/g" dependencies.yaml

# Version in cuspatial_api_examples.ipynb
sed_runner "s/rapids-[0-9]*\.[0-9]*/rapids-${NEXT_SHORT_TAG}/g" docs/source/user_guide/cuspatial_api_examples.ipynb
sed_runner "s/cuproj=[0-9]*\.[0-9]*/cuproj=${NEXT_SHORT_TAG}/g" docs/source/user_guide/cuspatial_api_examples.ipynb
sed_runner "s/cuspatial=[0-9]*\.[0-9]*/cuspatial=${NEXT_SHORT_TAG}/g" docs/source/user_guide/cuspatial_api_examples.ipynb
# Version in cuproj_api_examples.ipynb
sed_runner "s/rapids-[0-9]*\.[0-9]*/rapids-${NEXT_SHORT_TAG}/g" docs/cuproj/source/user_guide/cuproj_api_examples.ipynb
sed_runner "s/cuproj=[0-9]*\.[0-9]*/cuproj-${NEXT_SHORT_TAG}/g" docs/cuproj/source/user_guide/cuproj_api_examples.ipynb
sed_runner "s/cuspatial=[0-9]*\.[0-9]*/cuspatial=${NEXT_SHORT_TAG}/g" docs/cuproj/source/user_guide/cuproj_api_examples.ipynb

# Versions in README.md
sed_runner "s/cuspatial:[0-9]\+\.[0-9]\+/cuspatial:${NEXT_SHORT_TAG}/g" README.md
sed_runner "s/cuspatial=[0-9]\+\.[0-9]\+/cuspatial=${NEXT_SHORT_TAG}/g" README.md
sed_runner "s/notebooks:[0-9]\+\.[0-9]\+/notebooks:${NEXT_SHORT_TAG}/g" README.md

# .devcontainer files
find .devcontainer/ -type f -name devcontainer.json -print0 | while IFS= read -r -d '' filename; do
    sed_runner "s@rapidsai/devcontainers:[0-9.]*@rapidsai/devcontainers:${NEXT_SHORT_TAG}@g" "${filename}"
    sed_runner "s@rapidsai/devcontainers/features/rapids-build-utils:[0-9.]*@rapidsai/devcontainers/features/rapids-build-utils:${NEXT_SHORT_TAG_PEP440}@" "${filename}"
done
