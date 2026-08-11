# data/ — Git管理外

Ego4D GoalStep の実体データを置くディレクトリ。`README.md` と `manifest.json` 以外はGitに入れない。

```text
data/
├── manifest.json             使用動画の選定結果 (select_videos.py が生成・Git管理)
└── goalstep/
    ├── annotations/          scripts/fetch_annotations.sh で取得 (ライセンス不要)
    ├── v2/video_540ss/       ego4d CLI で取得した動画 (Ego4Dライセンス必要)
    ├── frames/<uid>/         scripts/extract_frames.py の出力 (1fps・540p)
    ├── sft_window/           窓タスクのSFTデータ (make_window_data.py)
    └── sft_span/             スパンタスクのSFTデータ (make_span_data.py)
```

動画の利用条件は Ego4D License に従う。再配布しないこと。
