# Factory Ego accuracy comparison dataset

全体のフォルダ境界と評価条件は[ベンチマーク文書](../../docs/benchmark/README.md)を参照してください。

Egocentric-10K の6工場から**作業の種類が満遍なく入るように層化抽出した20 unit**（各20秒・2fps・40フレーム）で、VLMがフレームごとの質問にどれだけ正しく答えられるか、そしてその回答から決定論的ルールで**手順を判定**できるかを比較するための開発用データです。

## 境界

- `units/`: 出典、区間、選定根拠、ローカル2fpsフレームのSHA manifest
- `sops/`: 比較時に使う質問・イベント仕様。現時点では `provisional`（暫定・人手レビュー前）
- `annotations/`: 人手で検証した正解だけを置く。現在の20 unitには人手GTがない
- `splits/`: factory/workerを跨がせない分割。既存unitは全て `dev_seen`

Fable、Opus、Qwenなどモデルの出力は事実層には置かず、リポジトリ直下の `runs/` に保存します。モデル間一致は予備比較であり、人手GTができるまで「精度」とは扱いません。

## データ概要（上流と選定母集団）

### 上流: builddotai/Egocentric-10K

- 実工場の一人称視点（ヘルメットカメラ）映像。**85工場・約10,000時間・192,900クリップ・約10.8億フレーム**
- 1920×1080・30fps。WebDataset形式（`factory_xxx/workers/worker_xxx/*.tar` 内に `.mp4`+`.json`）
- ライセンスはApache-2.0だが、**連絡先共有への同意が必要なgated dataset**

### 選定母集団: annotated-egocentric-10k-dataset

[fit-alessandro-berti/annotated-egocentric-10k-dataset](https://github.com/fit-alessandro-berti/annotated-egocentric-10k-dataset)（Apache-2.0）は、Egocentric-10Kのうち **factory_001〜006（59 worker・387クリップ）** にタイムスタンプ付きの作業書き起こしと、工場語彙に制約したプロセスマイニング用イベントログを付けた派生リポジトリです。当データセットはこれを**クリップ選定のためだけ**に使います。

重要な注意: この派生アノテーションは**全てLLM生成で人手検証がありません**。タイムスタンプには±数秒の誤差があり、実際にイベント定義を実フレームと照合すると「窓内で起きていないイベント」が多発しました。このため**イベント定義はtranscriptionからではなく、抽出フレームの目視で行います**（正本: [イベント定義ガイド](../../docs/benchmark/events.md)）。85工場のうち書き起こしが存在するのはこの6工場だけなので、サンプリング母集団もこの範囲です。

工場ごとの規模と主な作業種類（processが登場するクリップ数上位。イベントログが空の65クリップを除く322クリップで集計）:

| factory | clips | workers | 主なprocess（クリップ数上位5） |
|---|---|---|---|
| factory_001 | 97 | 11 | Material replenishment(60)、Manual mechanical assembly(60)、Finished goods transport(34)、Component bagging(25)、Metal stamping(19) |
| factory_002 | 54 | 10 | Fabric folding(19)、Garment bagging(16)、Material transport(14)、Floor auditing(12)、Garment ironing(11) |
| factory_003 | 72 | 11 | Surface finishing and polishing(42)、Wax pattern assembly(19)、Defect patching(15)、Mold preparation and cleaning(10)、Material sorting and sifting(8) |
| factory_004 | 51 | 10 | Material transport(30)、Thread trimming(18)、Overlock seaming(16)、Garment folding(14)、Quality inspection(13) |
| factory_005 | 22 | 11 | part sorting and staging(9)、material transport and logistics(8)、manual lathe machining(6)、drill press machining(6)、component deburring and polishing(4) |
| factory_006 | 26 | 6 | Pattern cleaning and coating(12)、Machine cleaning and maintenance(11)、Mold assembly and closing(9)、Sand mold creation(8)、Component staging and arrangement(8) |

業種はおおまかに、金属プレス・組立（001）、縫製後工程・梱包（002）、鋳造・成形（003）、縫製（004）、機械加工・モーター（005）、鋳型・成形（006）。

## サンプリング方法

`tools/benchmark/sample_units.py` による決定論的な層化抽出です（乱数なし・再実行で結果不変・データを増やすときは `--n` を増やして再実行すれば既存選定は変わらない追記型）。

1. annotated datasetのイベントログから「窓内にイベント開始が3〜6個入る遷移の多い区間」を持つクリップを候補化（単一動作の反復区間は除外される）
2. 作業種類（main process）の重複を避けつつ、各工場から最低2件・同一workerは最大2件の制約で選出
3. 窓は**20秒・2fps（0.5秒刻み）・40フレーム**（`f0000.jpg`〜`f0039.jpg`、t = 開始秒 + idx×0.5）

結果: 6工場・18 worker・**20種類の異なる作業**をカバー。各unitの選定根拠は `units/<unit_id>/meta.json` の `selection` に記録しています。

## イベント定義

**イベントの作り方は[イベント定義ガイド](../../docs/benchmark/events.md)が正本**です。要点:

- イベント＝手順の1ステップ（動作）。実フレームを目視して、**窓内で実際に起きた動作だけ**を定義する
- 質問は日本語の単文「作業者は〜しているか？」（前置きなし）。イベントidは英語snake_case
- 秒数はSOPに書かない（区間はアノテーションが記録する）
- unitあたり3〜4イベント。同一質問が2区間で成立するunitは `occurrence` で時系列N番目を指定

## unit一覧（イベント定義）

### f001_w004_material_replenishment — Material replenishment
- クリップ: `factory001_worker004_00000` の 82–102秒
- イベント: `assemble_part_with_driver` → `take_parts_from_bag` → `roll_up_bag`
  - 作業者は電動ドライバーで部品を組み立てているか？ / 袋から部品を取り出しているか？ / 袋を丸めているか？

### f001_w011_metal_stamping — Metal stamping
- クリップ: `factory001_worker011_00000` の 182–202秒
- イベント: `bundle_scrap` → `walk_to_next_station` → `pick_up_metal_sheets` → `gather_metal_sheets`
  - 作業者はスクラップをまとめているか？ / 歩いて移動しているか？ / 金属の板を手に取っているか？ / 金属の板を揃えてまとめているか？

### f002_w002_garment_bagging — Garment bagging
- クリップ: `factory002_worker002_00005` の 652–672秒
- イベント: `pick_garment_from_stack` → `insert_garment_into_bag` → `seal_bag_flap` → `place_package_right`
  - 作業者は積まれた服のスタックから服を1枚取っているか？ / 服をプラスチックの袋に入れているか？ / 袋の口を折り込んで閉じているか？ / 袋詰めした服を右側に置いているか？

### f002_w003_fabric_folding — Fabric folding
- クリップ: `factory002_worker003_00004` の 323–343秒
- イベント: `spread_garment_on_table` → `smooth_and_align_garment` → `fold_garment` → `take_hanger`
  - 作業者は衣類を作業台の上に広げているか？ / 広げた衣類を手でならして整えているか？ / 衣類を折りたたんでいるか？ / ハンガーを手に取っているか？

### f002_w005_garment_ironing — Garment ironing
- クリップ: `factory002_worker005_00007` の 52–72秒
- イベント: `place_folding_board` → `fold_shirt_around_board` → `flip_and_neaten_folded_shirt` → `stack_folded_shirt`
  - 作業者は白い折り畳み板をシャツの上に置いて位置を合わせているか？ / シャツを板に沿って折り畳んでいるか？ / 畳んだシャツを表に返して形を整えているか？ / 畳んだシャツを完成品の山に重ねているか？

### f003_w005_metal_casting — Metal casting separation
- クリップ: `factory003_worker005_00004` の 46–66秒
- イベント: `knock_parts_off_casting`（叩き1回目） → `carry_casting_bundle` → `put_casting_into_crate` → `resume_hammering_on_drum`（叩き2回目）
  - 作業者はハンマーで鋳物を叩いているか？（occurrence 1/2） / 鋳物の束を手に持って運んでいるか？ / 鋳物を木箱に入れているか？

### f003_w007_wax_pattern — Wax pattern assembly
- クリップ: `factory003_worker007_00003` の 157–177秒
- イベント: `assemble_parts_by_hand` → `join_parts_with_tool` → `press_wax_with_fingers` → `place_tree_on_rack`
  - 作業者はワックス部品を手に持って組み付けているか？ / 工具をワックス部品に当てて接合しているか？ / ワックス片を指で押し固めているか？ / ワックスツリーを金網ラックの上で移し替えているか？

### f003_w009_injection_molding — Injection molding
- クリップ: `factory003_worker009_00003` の 736–756秒
- イベント: `remove_molded_parts` → `place_parts_on_table` → `walk_to_next_position` → `set_inserts_into_mold`
  - 作業者は金型から成形品を取り外しているか？ / 成形品を作業台に置いているか？ / 歩いて移動しているか？ / 金型の中に手を入れて部品をセットしているか？

### f003_w010_mold_preparation — Mold preparation and cleaning
- クリップ: `factory003_worker010_00004` の 155–175秒
- イベント: `lift_pattern_plate` → `carry_sand_mold` → `walk_between_benches` → `pick_parts_from_box`
  - 作業者は黄色い型板を両手で持ち上げているか？ / 砂の入った型枠を運んでいるか？ / 作業場内を歩いて移動しているか？ / 木箱から小さな部品を取り出しているか？

### f004_w002_thread_trimming — Thread trimming
- クリップ: `factory004_worker002_00006` の 37–57秒
- イベント: `spread_garment_flat` → `pick_next_garment` → `trim_threads_upper` → `trim_threads_hem`
  - 作業者は服を台の上に広げて平らに整えているか？ / 次の服を布の山から取り上げているか？ / 糸切りばさみで服の襟や肩まわりの糸を切っているか？ / 服の裾をめくり上げて糸を切っているか？

### f004_w004_continuous_fabric — Continuous fabric chaining
- クリップ: `factory004_worker004_00001` の 75–95秒
- イベント: `stow_black_fabric` → `bring_garment_to_machine` → `align_garment_edge` → `set_under_presser_foot`
  - 作業者は黒い生地を作業台の上に広げて置いているか？ / 黒い衣類を手に取ってミシン台へ引き寄せているか？ / 衣類の縁を両手で揃えているか？ / 生地をミシンの押さえ金の下にセットしているか？

### f004_w005_heat_press — Heat press labeling
- クリップ: `factory004_worker005_00000` の 50–70秒
- イベント: `open_heat_press` → `move_fabric_to_table` → `peel_transfer_paper` → `smooth_arrange_shirt`
  - 作業者はヒートプレス機のハンドルを引いて上盤を開けているか？ / プレス台から生地を取り出して作業台へ移しているか？ / Tシャツから転写紙を剥がしているか？ / Tシャツを台の上に広げて手で整えているか？

### f004_w005_overlock_seaming — Overlock seaming
- クリップ: `factory004_worker005_00001` の 18–38秒
- イベント: `pick_piece_from_pile` → `unfold_and_orient_piece` → `check_seam_edge_by_hand` → `set_piece_under_needle`
  - 作業者は布の山から布を取り上げているか？ / 布を手元で広げて向きを整えているか？ / 布端の縫い目を指でつまんで確認しているか？ / 布をミシンの針元に合わせて置いているか？

### f004_w006_curvilinear_seam — Curvilinear seam joining
- クリップ: `factory004_worker006_00002` の 1188–1208秒（クリップ終端1200秒以降は最終フレームの繰り返し）
- イベント: `align_fabric_edges` → `position_fabric_under_needle` → `fine_finger_work_at_needle`
  - 作業者は布の端を手で揃えているか？ / 布をミシンの針の下に置いて位置を合わせているか？ / 針元で指先を使った細かい作業をしているか？

### f004_w006_edge_binding — Edge binding attachment
- クリップ: `factory004_worker006_00003` の 0–20秒
- イベント: `handle_fabric_at_needle` → `trim_thread_with_scissors` → `flip_and_check_binding`
  - 作業者はミシンの針元で生地を扱っているか？ / はさみで糸を切っているか？ / 生地を手で裏返して縁を確かめているか？

### f005_w001_semi_automatic — semi-automatic stator wire insertion
- クリップ: `factory005_worker001_00002` の 32–52秒
- イベント: `adjust_coil_leads` → `reach_machine_base` → `lace_stator_cord` → `press_panel_button`
  - 作業者はコイルのリード線を手で整えているか？ / 機械の下部に手を伸ばして作業しているか？ / 白い紐をステータに通しているか？ / 制御盤のボタンを押しているか？

### f005_w010_manual_lathe — manual lathe machining
- クリップ: `factory005_worker010_00000` の 126–146秒
- イベント: `tighten_workpiece_nut` → `place_wrench_on_carriage` → `start_spindle` → `run_spindle`
  - 作業者はスパナでナットを締めているか？ / スパナを往復台の上に置いているか？ / 主軸のレバーを操作しているか？ / 工作物を回転させているか？

### f005_w011_cnc_machine — CNC machine tending
- クリップ: `factory005_worker011_00000` の 963–983秒
- イベント: `carry_workpiece_to_machine` → `load_workpiece_into_chuck` → `press_control_panel_button` → `arrange_parts_on_table`
  - 作業者は部品を手に持って機械まで歩いているか？ / 機械のチャックに部品を取り付けているか？ / 操作盤のボタンを押しているか？ / 作業台の上で部品を並べているか？

### f006_w004_bulk_material — Bulk material transport
- クリップ: `factory006_worker004_00001` の 515–535秒
- イベント: `pick_parts_from_pile` → `put_parts_into_box` → `reposition_parts_box`
  - 作業者は山積みの部品の中から筒状の部品を拾い上げているか？ / 筒状の部品を鉄製の箱の中に入れているか？ / 部品の入った鉄箱を引いて位置を直しているか？

### f006_w005_compression_molding — Compression molding
- クリップ: `factory006_worker005_00002` の 849–869秒
- イベント: `place_mold_on_press` → `operate_control_panel` → `stand_by_press`
  - 作業者は金型を手でプレスの台に置いて位置を合わせているか？ / 操作盤のスイッチやバルブを手で操作しているか？ / プレスに触れずに操作盤の前に立って様子を確認しているか？

## 出典と配布

- Upstream: `builddotai/Egocentric-10K`（Apache-2.0、gated: 連絡先共有への同意が必要）
- 選定母集団: `fit-alessandro-berti/annotated-egocentric-10k-dataset`（Apache-2.0、LLM生成・人手検証なし。`dataset.yaml` の `selection_source.revision` に参照コミットを固定）
- Sampling: 1920x1080映像から2fps（0.5秒刻み）で抽出（JPEG quality 85。`frames.sha256.json` とバイト一致する条件）

抽出フレームは公開repositoryへ含めません。upstreamのgated accessを通して取得したローカル媒体を使い、`frames.sha256.json` で同一性を検証します。再配布する場合はupstreamの最新ライセンスだけでなく、アクセス時に同意した条件とプライバシー要件を確認してください。

ローカル媒体の再構成は `tools/benchmark/fetch_factory_ego.py` で自動化されています（unitの `meta.json` から対象factory/worker・fps・窓を導出するため、データが増えても引数は不要）。gated accessに同意した後の手順は[ベンチマーク運用ガイド](../../docs/benchmark/operations.md)を参照してください。
