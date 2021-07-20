set -ex
cd cn
mike deploy 0.5.0 latest -u --prefix _site -b docs_output
mike set-default latest --prefix _site -b docs_output
cd ../en
mike deploy 0.5.0 latest -u --prefix _site/en -b docs_output
mike set-default latest --prefix _site/en -b docs_output
