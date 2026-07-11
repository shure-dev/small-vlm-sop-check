# Factory Ego accuracy comparison dataset

全体のフォルダ境界と評価条件は[ベンチマーク文書](../../docs/benchmark/README.md)を参照してください。

Egocentric-10K の6工場から**作業の種類が満遍なく入るように層化抽出した20 unit**（各10秒・1fps・10フレーム）で、VLMがフレームごとの質問にどれだけ正しく答えられるか、そしてその回答から決定論的ルールで**手順を判定**できるかを比較するための開発用データです。

## 境界

- `units/`: 出典、区間、選定根拠、ローカル1fpsフレームのSHA manifest
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

[fit-alessandro-berti/annotated-egocentric-10k-dataset](https://github.com/fit-alessandro-berti/annotated-egocentric-10k-dataset)（Apache-2.0）は、Egocentric-10Kのうち **factory_001〜006（59 worker・387クリップ）** にタイムスタンプ付きの作業書き起こしと、工場語彙に制約したプロセスマイニング用イベントログを付けた派生リポジトリです。当データセットはこれを**クリップ選定と暫定SOP設計のためだけ**に使います。

重要な注意: この派生アノテーションは**全てLLM生成で人手検証がありません**。タイムスタンプには±数秒の誤差を前提とし、ground truthとしては一切使いません（正はローカル抽出フレームと将来の人手GT）。85工場のうち書き起こしが存在するのはこの6工場だけなので、サンプリング母集団もこの範囲です。

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

1. イベントログから「**10秒間にイベント開始が3〜6個入る**遷移の多い区間」を持つクリップを候補化（単一動作の反復区間は除外される）。イベント種類数→イベント数→早い区間の順でスコア付け
2. 作業種類（main process）の重複を避けつつ、各工場から最低2件・同一workerは最大2件の制約で20件選出
3. 窓は 開始秒〜開始秒+9 の10秒（1fpsで10フレーム、`f0000.jpg`〜`f0009.jpg`）

結果: 6工場・18 worker・**20種類の異なる作業**をカバー。各unitの選定根拠（窓内のイベント列とtranscription抜粋）は `units/<unit_id>/meta.json` の `selection` に記録しています。

## SOP設計方針（手順判定が目的）

- 各イベント＝手順の1ステップ（例:「金型に部品を置く」「プレスを作動させる」）。unitあたり3〜5イベント
- questionsは動画解析として設計: **直近数フレームの動きを文脈に、最新フレーム時点の状態**をyes/noで問う
- 窓が10フレームと短いため `min_frames` は1〜2。境界±数フレームのズレは注釈ではなくtIoUしきい値側で吸収する
- relationsは現時点では使わない（`relations: []`）。イベントは将来 `before`/`overlaps` で手順順序を表現できる粒度にしてある
- 期待されるイベント順序と窓内オフセットの見立て（LLM由来timestampからの推定）は各SOPの `benchmark.review.expected_sequence` に記録

## unit一覧（イベント定義レビュー用）

各unitの「期待イベント列」の秒数は窓内オフセットの**見立て**です（LLM生成timestampに由来。実フレームでの検証はアノテーション段階で行う）。抜粋は選定に使ったtranscriptionの該当部分（英語・LLM生成）。

### f001_w004_material_replenishment — Material replenishment
- クリップ: `factory001_worker004_00000` の 82–91秒
- 期待イベント列: scoop_parts(0-2s) → drive_fastener(2-6s) → handle_bag(6-9s)
- 抜粋: [82s-84s] The person reaches back to their left, grabs a handful of black parts, and drops them into the left blue bin. / [84s-88s] The person resumes the assembly cycle for a few seconds. / [88s-91s] The person briefly picks up a small clear plastic bag, handles it for a moment, and then drops it down.

### f001_w011_metal_stamping — Metal stamping
- クリップ: `factory001_worker011_00000` の 182–191秒
- 期待イベント列: carry_scrap(0-5s) → stage_blanks(5-8s) → press_cycle(8-9s)
- 抜粋: [182s-187s] The wearer walks a few steps, drops the scrap metal pieces into a blue plastic crate on the floor, and turns around to walk back to the machine. / [187s-190s] The wearer returns to the machine and uses both hands to slide a heavy stack of flat metal pieces closer to the working area. / [190s-] resumes the pressing cycle.

### f002_w002_garment_bagging — Garment bagging
- クリップ: `factory002_worker002_00005` の 652–661秒
- 期待イベント列: fold_garment(0-4s) → insert_bag(4-7s) → seal_bag(7-9s)
- 抜粋: [652s-846s] For each garment, the wearer folds the fabric around the cardboard insert, places it into the clear plastic bag, peels the protective strip off the bag's adhesive flap, and seals the bag shut.

### f002_w003_fabric_folding — Fabric folding
- クリップ: `factory002_worker003_00004` の 323–332秒
- 期待イベント列: lay_garment(0-2s) → fold_bottom(2-6s) → fold_sides(6-9s)
- 抜粋: [323s-359s] They reach to a stack on their left, pick up a light blue garment (appearing to be pants), and lay it flat on the table. Using both hands, they fold the bottom leg section upwards toward the waistband area. They then fold the left and right outer edges inward to the center.

### f002_w005_garment_ironing — Garment ironing
- クリップ: `factory002_worker005_00007` の 52–61秒
- 期待イベント列: iron_press(0-4s) → remove_board(4-8s) → final_fold(9s)
- 抜粋: [52s-56s] They pick up the iron again, press the folded lower section of the shirt, and set the iron down. / [56s-61s] The wearer grasps the white folding board with their right hand and slides it smoothly out from the collar opening of the folded shirt. / [61s-68s] They flip the folded shirt over.

### f003_w005_metal_casting — Metal casting separation
- クリップ: `factory003_worker005_00004` の 46–55秒
- 期待イベント列: sweep_bar(0-3s) → scoop_parts(3-7s) → hammer_strike(7-9s)
- 抜粋: [46s-49s] Back at the workstation, the wearer pushes a long, flat metal guide bar upwards along the sloped surface. / [49s-53s] They immediately turn right, walk to the bin, and scoop another handful of parts. / [53s-70s] The wearer places individual parts on the surface and strikes them firmly with the hammer.

### f003_w007_wax_pattern — Wax pattern assembly
- クリップ: `factory003_worker007_00003` の 157–166秒
- 期待イベント列: brush_paste(0-5s) → insert_mold(5-8s) → scrape_piece(8-9s)
- 抜粋: [157s-162s] They brush the liquid onto the fifth piece. / [162s-165s] They insert the fifth piece into the assembly mold. / [165s-180s] They pick up a sixth piece and clean it thoroughly with the metal tool.

### f003_w009_injection_molding — Injection molding
- クリップ: `factory003_worker009_00003` の 736–745秒
- 期待イベント列: extract_sprue(0-4s) → snap_parts(2-5s) → empty_box(5-9s)
- 抜粋: [736s-741s] The molding cycle resumes (extracting parts, breaking them into the box). / [741s-745s] The box is emptied onto the tray.

### f003_w010_mold_preparation — Mold preparation and cleaning
- クリップ: `factory003_worker010_00004` の 155–164秒
- 期待イベント列: load_insert(0-2s) → close_mold(2-4s) → press_mold(4-5s) → walk_away(5-9s)
- 抜粋: [155s-160s] The person places the large clay cup onto the center of the mold, places the top half of the mold over it, and pushes the assembly into the machine. / [160s-187s] The person steps away from the machine and walks around the busy workshop area.

### f004_w002_thread_trimming — Thread trimming
- クリップ: `factory004_worker002_00006` の 37–46秒
- 期待イベント列: fold_stack(0-4s) → pick_new(4-8s) → snip_threads(8-9s)
- 抜粋: [37s-41s] The person folds the onesie in half and then again, placing the finished garment onto a neat pile. / [41s-45s] The person reaches to the left, picks up a new white onesie from the unfinished pile, and lays it out. / [45s-66s] The person inspects the new garment while using the snipping scissors to remove loose threads.

### f004_w004_continuous_fabric — Continuous fabric chaining
- クリップ: `factory004_worker004_00001` の 75–84秒
- 期待イベント列: sew_fabric(0-5s) → remove_piece(5-7s) → grab_new(7-9s)
- 抜粋: [0s-81s] The wearer sits at a sewing machine and sews a piece of black fabric... At the end, they pull the fabric from the machine, break the thread, and throw the finished piece onto a pile. / [81s-264s] The wearer picks up another piece of black fabric from the right, unfolds it, and aligns the edges.

### f004_w005_heat_press — Heat press labeling
- クリップ: `factory004_worker005_00000` の 50–59秒
- 期待イベント列: press_closed(0-4s) → remove_garment(4-6s) → inspect_logo(6-9s)
- 抜粋: [44s-59s] They pull the orange handle of the machine down to clamp it shut. A digital timer counts down and beeps. The wearer lifts the handle, removes the hot onesie from the press, places it back on the table, and unfolds the fabric to reveal a small print.

### f004_w005_overlock_seaming — Overlock seaming
- クリップ: `factory004_worker005_00001` の 18–27秒
- 期待イベント列: finish_seam(0-2s) → remove_toss(2-5s) → pick_unfold(5-9s)
- 抜粋: [0s-23s] Upon finishing the seam, the wearer pulls the fabric away from the needle, breaks the thread, and tosses the sewn piece onto a pile. / [23s-52s] The wearer reaches to a large pile of unsewn pink fabric, picks up a single piece, unfolds and orients it, aligning two edges.

### f004_w006_curvilinear_seam — Curvilinear seam joining
- クリップ: `factory004_worker006_00002` の 1188–1197秒
- 期待イベント列: cut_thread(0-2s) → set_aside(2-4s) → pick_new(4-6s) → align_edges(6-9s)
- 抜粋: [1188s-1191s] They stop sewing, use the small snips to sever the thread, and pull the finished fabric piece out. / [1191s-1200s] The completed piece is placed to the side. The person picks up a new piece of grey fabric, unfolds it, and begins aligning its edges.

### f004_w006_edge_binding — Edge binding attachment
- クリップ: `factory004_worker006_00003` の 0–9秒
- 期待イベント列: sew_binding(0-5s) → cut_thread(5-7s) → swap_pieces(7-9s)
- 抜粋: [0s-7s] The person is sewing a piece of grey fabric... They finish the seam, pull the fabric slightly away from the needle, and use scissors to cut the thread. / [7s-14s] The person places the finished piece on a pile to their left, then picks up a new piece of grey fabric.

### f005_w001_semi_automatic — semi-automatic stator wire insertion
- クリップ: `factory005_worker001_00002` の 32–41秒
- 期待イベント列: unload_stator(0-2s) → carry_stator(2-6s) → mount_stator(6-9s)
- 抜粋: [32s-38s] The machine stops. The person removes the stator from the mount, turns, and walks over to the metal table. They stand idle for a few seconds holding the stator. / [38s-43s] They turn back around, return to the machine, and remount the exact same stator back onto the central fixture.

### f005_w010_manual_lathe — manual lathe machining
- クリップ: `factory005_worker010_00000` の 126–135秒
- 期待イベント列: hand_thread(0-5s) → wrench_tighten(5-8s) → engage_spindle(8-9s)
- 抜粋: [120s-131s] The person mounts the new part onto the lathe's fixture, then threads the washer and nut onto the central rod by hand. / [131s-139s] They pick up the large wrench to tighten the nut securely. They then turn handwheels and pull a lever to engage the machine.

### f005_w011_cnc_machine — CNC machine tending
- クリップ: `factory005_worker011_00000` の 963–972秒
- 期待イベント列: open_door(0-2s) → swap_parts(2-6s) → close_door(6-8s) → start_cycle(8-9s)
- 抜粋: [963s-976s] The wearer opens the machine door, swaps the parts, closes it, starts the machine, and brings the finished part to the table.

### f006_w004_bulk_material — Bulk material transport
- クリップ: `factory006_worker004_00001` の 515–524秒
- 期待イベント列: push_barrow(0-5s) → dump_load(5-8s) → load_casing(8-9s)
- 抜粋: [515s-524s] The wearer pushes the loaded wheelbarrow a short distance and tips it forward, dumping the finned casings onto a pile on the floor. / [524s-575s] The wearer returns to loading the wheelbarrow, picking up finned casings from the floor.

### f006_w005_compression_molding — Compression molding
- クリップ: `factory006_worker005_00002` の 849–858秒
- 期待イベント列: eject_block(0-4s) → clean_mold(3-6s) → load_mold(6-8s) → press_controls(8-9s)
- 抜粋: [849s-859s] Back at the hydraulic press, they remove a mold, extract a block, clean the mold, replace it, and operate the press. / [859s-868s] They interact with the control panel.

## 出典と配布

- Upstream: `builddotai/Egocentric-10K`（Apache-2.0、gated: 連絡先共有への同意が必要）
- 選定母集団: `fit-alessandro-berti/annotated-egocentric-10k-dataset`（Apache-2.0、LLM生成・人手検証なし。`dataset.yaml` の `selection_source.revision` に参照コミットを固定）
- Sampling: 1920x1080映像から1fpsで抽出（JPEG quality 85。`frames.sha256.json` とバイト一致する条件）

抽出フレームは公開repositoryへ含めません。upstreamのgated accessを通して取得したローカル媒体を使い、`frames.sha256.json` で同一性を検証します。再配布する場合はupstreamの最新ライセンスだけでなく、アクセス時に同意した条件とプライバシー要件を確認してください。

ローカル媒体の再構成は `tools/benchmark/fetch_factory_ego.py` で自動化されています（unitの `meta.json` から対象factory/workerを導出するため、データが増えても引数は不要）。gated accessに同意した後の手順は[ベンチマーク運用ガイド](../../docs/benchmark/operations.md)を参照してください。
