set -ex
MIKE="mike"
CN_SITE="_site"
EN_SITE="_site/en"
LATEST_VERSION="latest"
OUTPUT_BRANCH="docs_output"

CN_OPTIONS="--prefix ${CN_SITE} -b ${OUTPUT_BRANCH}"
EN_OPTIONS="--prefix ${EN_SITE} -b ${OUTPUT_BRANCH}"

cd cn

# master
${MIKE} delete --all ${CN_OPTIONS}
git checkout master
${MIKE} deploy master ${LATEST_VERSION} -u ${CN_OPTIONS}
${MIKE} set-default ${LATEST_VERSION} ${CN_OPTIONS}
cd ../en
${MIKE} delete --all ${EN_OPTIONS}
${MIKE} deploy master ${LATEST_VERSION} -u ${EN_OPTIONS}
${MIKE} set-default ${LATEST_VERSION} ${EN_OPTIONS}

# v0.4.0
VERSION="v0.4.0"
cd .. && git checkout ${VERSION}
cd cn
${MIKE} deploy ${VERSION} -u ${CN_OPTIONS}
cd ../en
${MIKE} deploy ${VERSION} -u ${EN_OPTIONS}

# v0.7.0
VERSION="v0.7.0"
cd .. && git checkout ${VERSION}
cd cn
${MIKE} deploy ${VERSION} -u ${CN_OPTIONS}
cd ../en
${MIKE} deploy ${VERSION} -u ${EN_OPTIONS}

# v0.8.0
VERSION="v0.8.0"
cd .. && git checkout ${VERSION}
cd cn
${MIKE} deploy ${VERSION} -u ${CN_OPTIONS}
cd ../en
${MIKE} deploy ${VERSION} -u ${EN_OPTIONS}