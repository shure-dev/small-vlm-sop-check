#!/usr/bin/env bash
# GoalStep アノテーション一式を GitHub から取得する(ライセンス不要・数十MB)。
# 動画本体は Ego4D ライセンス取得後に ego4d CLI で:
#   ego4d -o data/goalstep --version v2 --datasets video_540ss --no-metadata -y \
#         --video_uid_file <uidリスト>
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/goalstep/annotations
cd data/goalstep/annotations
for f in goal_trainval.json goalstep_train.json goalstep_valid.json goalstep_test.json goalstep_video_groups.tsv; do
  curl -sL -o "$f" "https://raw.githubusercontent.com/facebookresearch/ego4d-goalstep/main/data/$f" &
done
wait
ls -lh
