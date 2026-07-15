# ADR 0002: Adopt src-layout and packaged applications

- Status: accepted
- Date: 2026-07-10

## Context

Python moduleが `src/` 直下にあり、testsとbrowser toolsが `sys.path` を書き換えて裸importしていました。また、ユーザー機能と一度きりの移行scriptがどちらも `tools/` に置かれ、配布境界が不明確でした。

## Decision

- import可能なコードを `src/small_vlm_sop_check/` へ集約する
- core、inference、appsを依存方向で分ける
- 書き込み可能な `sop-app` と読み取り専用の `sop-view` を同じFastAPI launcherで提供する
- UI、保存層、媒体処理、評価処理をpackage内で分離する
- testsはunit/integration、schemaは責務別fileへ分ける
- `pyproject.toml` をpackage・依存・test設定の正本にする

## Consequences

開発時はeditable installが基本になります。代わりにcheckout固有のimport hackがなくなり、wheel/entry pointとtestが同じpackageを使います。VLM依存はoptional extraなので、判定・評価だけなら重いML dependencyを導入しません。
