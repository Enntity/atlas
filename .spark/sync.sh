#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Keep Enntity/atlas `spark` = Atlas-Inf main + every open Atlas-Inf PR from
# Enntity. SparkQwen and SparkGLM pin a commit of this branch.
#
#   .spark/sync.sh                 merge main and the open PRs, push spark
#   .spark/sync.sh --pin DIR...    also pin each product checkout to the result
#
# Merge commits only, so spark only moves forward and never needs a forced
# push. A PR that has merged upstream adds nothing; with no PRs open, spark's
# tree equals main's. A conflict stops the script with the merge in progress.
set -euo pipefail
upstream=Atlas-Inf/atlas
cd "$(git rev-parse --show-toplevel)"
[[ -z $(git status --porcelain) ]] || { echo 'working tree has changes' >&2; exit 2; }
git checkout -q spark
git pull -q --ff-only origin spark
git fetch -q "https://github.com/$upstream.git" main
git merge -q --no-edit -m "spark: merge Atlas-Inf main $(git rev-parse --short=9 FETCH_HEAD)" FETCH_HEAD
gh pr list --repo "$upstream" --state open --limit 100 --json number,headRefName,headRepositoryOwner \
  --jq '.[] | select(.headRepositoryOwner.login == "Enntity") | "\(.number) \(.headRefName)"' |
while read -r number ref; do
  git fetch -q origin "$ref"
  git merge -q --no-edit -m "spark: merge #$number ($ref)" FETCH_HEAD
done
git push -q origin spark
commit=$(git rev-parse HEAD) tree=$(git rev-parse 'HEAD^{tree}')
echo "spark $commit tree $tree"
[[ ${1:-} == --pin ]] || exit 0
shift
for product in "$@"; do
  python3 - "$product/install/atlas-source.json" "$commit" "$tree" <<'PY'
import json, sys
path, commit, tree = sys.argv[1:]
pin = json.load(open(path))
pin.update(repository='https://github.com/Enntity/atlas.git', branch='spark', commit=commit, tree=tree)
open(path, 'w').write(json.dumps(pin, indent=2) + '\n')
PY
  echo "pinned $product"
done
