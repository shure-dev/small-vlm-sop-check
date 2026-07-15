# Local media — Git管理外

動画、音声、抽出フレーム、アノテーション用previewを置くディレクトリです。`README.md` 以外はGitへ追加されません。

```text
data/<dataset_id>/units/<unit_id>/
├── video.mp4                 任意の元動画または切り出し動画
├── frames/f0000.jpg          canonical sampling frames
└── .annotation-preview.mp4   framesから自動生成されるローカルcache
```

公開可能なmetadata、日本語イベント、人手区間、split、hashは `datasets/<dataset_id>/` に置きます。媒体の利用条件を確認せず再配布しないでください。
