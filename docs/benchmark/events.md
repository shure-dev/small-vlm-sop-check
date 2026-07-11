# Event definitions

イベントは動画unitごとのSOPで定義します。`questions:` がVLMへのフレームごとの質問、`events:` が回答からイベント区間を作る決定論的条件です。

## Konro Inspection

[正しい手順SOP](../../datasets/konro_inspection/sops/konro_inspection/correct.yaml)を正本とし、`wrong_order.yaml` と `missing_step.yaml` は同じquestions定義に対する違反条件です。

## Factory Ego

手順判定を目的に、各イベントは手順の1ステップ（動作）に対応させています。questionsは「直近数フレームの動きを文脈に、最新フレーム時点の状態」を問う動画解析設計です。窓は10秒・1fps・10フレームで、unitあたり3〜5イベントが入ります。期待されるイベント順序の見立ては各SOPの `benchmark.review.expected_sequence`、選定根拠は[データセットREADME](../../datasets/factory_ego/README.md)のunit一覧を参照してください。

| unit | events | SOP |
|---|---|---|
| f001_w004_material_replenishment | `scoop_parts`、`drive_fastener`、`handle_bag` | [v001](../../datasets/factory_ego/sops/f001_w004_material_replenishment/v001.yaml) |
| f001_w011_metal_stamping | `carry_scrap`、`stage_blanks`、`press_cycle` | [v001](../../datasets/factory_ego/sops/f001_w011_metal_stamping/v001.yaml) |
| f002_w002_garment_bagging | `fold_garment`、`insert_bag`、`seal_bag` | [v001](../../datasets/factory_ego/sops/f002_w002_garment_bagging/v001.yaml) |
| f002_w003_fabric_folding | `lay_garment`、`fold_bottom`、`fold_sides` | [v001](../../datasets/factory_ego/sops/f002_w003_fabric_folding/v001.yaml) |
| f002_w005_garment_ironing | `iron_press`、`remove_board`、`final_fold` | [v001](../../datasets/factory_ego/sops/f002_w005_garment_ironing/v001.yaml) |
| f003_w005_metal_casting | `sweep_bar`、`scoop_parts`、`hammer_strike` | [v001](../../datasets/factory_ego/sops/f003_w005_metal_casting/v001.yaml) |
| f003_w007_wax_pattern | `brush_paste`、`insert_mold`、`scrape_piece` | [v001](../../datasets/factory_ego/sops/f003_w007_wax_pattern/v001.yaml) |
| f003_w009_injection_molding | `extract_sprue`、`snap_parts`、`empty_box` | [v001](../../datasets/factory_ego/sops/f003_w009_injection_molding/v001.yaml) |
| f003_w010_mold_preparation | `load_insert`、`close_mold`、`press_mold`、`walk_away` | [v001](../../datasets/factory_ego/sops/f003_w010_mold_preparation/v001.yaml) |
| f004_w002_thread_trimming | `fold_stack`、`pick_new`、`snip_threads` | [v001](../../datasets/factory_ego/sops/f004_w002_thread_trimming/v001.yaml) |
| f004_w004_continuous_fabric | `sew_fabric`、`remove_piece`、`grab_new` | [v001](../../datasets/factory_ego/sops/f004_w004_continuous_fabric/v001.yaml) |
| f004_w005_heat_press | `press_closed`、`remove_garment`、`inspect_logo` | [v001](../../datasets/factory_ego/sops/f004_w005_heat_press/v001.yaml) |
| f004_w005_overlock_seaming | `finish_seam`、`remove_toss`、`pick_unfold` | [v001](../../datasets/factory_ego/sops/f004_w005_overlock_seaming/v001.yaml) |
| f004_w006_curvilinear_seam | `cut_thread`、`set_aside`、`pick_new`、`align_edges` | [v001](../../datasets/factory_ego/sops/f004_w006_curvilinear_seam/v001.yaml) |
| f004_w006_edge_binding | `sew_binding`、`cut_thread`、`swap_pieces` | [v001](../../datasets/factory_ego/sops/f004_w006_edge_binding/v001.yaml) |
| f005_w001_semi_automatic | `unload_stator`、`carry_stator`、`mount_stator` | [v001](../../datasets/factory_ego/sops/f005_w001_semi_automatic/v001.yaml) |
| f005_w010_manual_lathe | `hand_thread`、`wrench_tighten`、`engage_spindle` | [v001](../../datasets/factory_ego/sops/f005_w010_manual_lathe/v001.yaml) |
| f005_w011_cnc_machine | `open_door`、`swap_parts`、`close_door`、`start_cycle` | [v001](../../datasets/factory_ego/sops/f005_w011_cnc_machine/v001.yaml) |
| f006_w004_bulk_material | `push_barrow`、`dump_load`、`load_casing` | [v001](../../datasets/factory_ego/sops/f006_w004_bulk_material/v001.yaml) |
| f006_w005_compression_molding | `eject_block`、`clean_mold`、`load_mold`、`press_controls` | [v001](../../datasets/factory_ego/sops/f006_w005_compression_molding/v001.yaml) |

Factory EgoのSOPはすべて `status: provisional` です（人手レビュー前）。イベント定義はannotated-egocentric-10kのLLM生成transcriptionから設計したもので、区間の見立ては±数秒の誤差を前提とします。
