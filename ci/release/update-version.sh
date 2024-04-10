#!/bin/bash
# Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

# Centralized version file update
echo "${NEXT_FULL_TAG}" > VERSION

# CI files
for FILE in .github/workflows/*.yaml; do
  sed_runner "/shared-workflows/ s/@.*/@branch-${NEXT_SHORT_TAG}/g" "${FILE}"
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
    sed_runner "/-.* ${DEP}\(-cu[[:digit:]]\{2\}\)\{0,1\}==/ s/==.*/==${NEXT_SHORT_TAG_PEP440}.*/g" "${FILE}"
  done
  sed_runner "s/${DEP}\(-cu[[:digit:]]\{2\}\)\{0,1\}==.*\",/${DEP}==${NEXT_SHORT_TAG_PEP440}.*\",/g" python/cuspatial/pyproject.toml
  sed_runner "s/${DEP}\(-cu[[:digit:]]\{2\}\)\{0,1\}==.*\",/${DEP}==${NEXT_SHORT_TAG_PEP440}.*\",/g" python/cuproj/pyproject.toml
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
