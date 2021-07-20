set -ex
git config --global user.name "YaoChi"
git config --global user.email "later@usopp.net"

MIKE="mike"
CN_SITE="_site"
EN_SITE="_site/en"
LATEST_VERSION="latest"
OUTPUT_BRANCH="docs_output"

CN_OPTIONS="--prefix ${CN_SITE} -b ${OUTPUT_BRANCH}"
EN_OPTIONS="--prefix ${EN_SITE} -b ${OUTPUT_BRANCH}"

cd cn
${MIKE} deploy master ${LATEST_VERSION} -u ${CN_OPTIONS}
${MIKE} set-default ${LATEST_VERSION} ${CN_OPTIONS}
cd ../en
${MIKE} deploy master ${LATEST_VERSION} -u ${EN_OPTIONS}
${MIKE} set-default ${LATEST_VERSION} ${EN_OPTIONS}
