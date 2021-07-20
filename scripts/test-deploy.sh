set -ex
target_branch=test-mike-2
mike deploy --update-aliases 0.1 latest -b ${target_branch}
mike deploy --update-aliases 0.2 latest -b ${target_branch}
mike deploy --update-aliases 0.3 latest -b ${target_branch}
mike set-default latest -b ${target_branch}
mike serve -b test-mike-2
