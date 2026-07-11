# small-vlm-sop-check

[日本語](README.md) | [Documentation](docs/README.md)

**Edge AI that flags procedure mistakes in work videos — skipped steps, wrong order — without footage ever leaving the site.**

An experimental framework for checking whether a work video follows a standard operating procedure (SOP). The local VLM only answers per-frame yes / no questions; **deterministic rules** make the pass / fail judgement.

- Produces `PASS` / `FAIL` with violation reasons from a video and an SOP
- Evaluates observation quality separately from judgement quality using human ground truth
- Provides CLI tools for annotation, inference, evaluation, and replay
- Includes results from 15 local VLMs tested under the same demo conditions

<p align="center">
  <img src="docs/assets/replay_demo.gif" alt="Replay viewer showing VLM answers, detected events, ground-truth spans, and the final verdict" width="640">
</p>

## Quick start

Python 3.10 or newer is required. The first example runs the deterministic rule engine against the bundled observation log, so no VLM dependency is needed.

```bash
python3 -m pip install -e .

sop-check judge \
  --sop datasets/konro_inspection/sops/konro_inspection/correct.yaml \
  --answer-log datasets/konro_inspection/fixtures/reference_outputs/answer_log.json
```

Expected result:

```text
>>> 総合判定: PASS <<<
[正解照合] verdict ✓  =>  箇所特定 ✓
```

Generate a self-contained replay viewer with:

```bash
sop-replay
```

## Design

The VLM only answers visual questions for each frame. A deterministic rule engine handles ordering, overlap, and prohibited actions.

```mermaid
flowchart LR
    A[Video] --> B[Frame extraction]
    B --> C[VLM observation<br/>yes / no per question]
    C --> D[Event spans]
    D --> E[Deterministic rules]
    E --> F[PASS / FAIL<br/>violation reasons]
```

This separation makes it possible to distinguish visual errors from rule errors, rerun an SOP against a saved observation log, and keep human facts separate from model predictions. See [ADR 0001](docs/decisions/0001-separate-facts-predictions-evaluations.md) for the rationale.

## Run with a local VLM

The reference MLX backend requires macOS on Apple Silicon.

```bash
python3 -m pip install -e ".[vlm]"

sop-check run \
  --sop datasets/konro_inspection/sops/konro_inspection/correct.yaml \
  --video datasets/konro_inspection/units/konro_inspection/media/konro_inspection.mp4 \
  --model qwen3-4b \
  --out-dir out/qwen3-4b
```

The model is downloaded on first use. Run `sop-check models` to list registered model aliases.

## Evaluation

The project evaluates two independent questions:

| Axis | Question | Reference |
|---|---|---|
| Observation | Did the VLM see what actually happened? | Human `ground_truth.json` |
| Judgement | Did it return the expected verdict and reason? | SOP `expect` |

```bash
sop-check eval \
  --sop datasets/konro_inspection/sops/konro_inspection/correct.yaml \
  --ground-truth datasets/konro_inspection/annotations/human-v001/konro_inspection.json \
  --answer-log datasets/konro_inspection/fixtures/reference_outputs/answer_log.json
```

Observation metrics include relation agreement, mean temporal IoU, and per-question frame agreement.

## Bundled benchmarks

### Konro Inspection

A complete demo with one 16-frame gas-stove inspection video, three SOP conditions, and human ground truth. Among the 15 tested models, Qwen3-VL-4B was the only model to achieve both 3/3 judgement correctness and 6/6 relation agreement. Its frame agreement was 96% and mean tIoU was 0.80.

This is a result for one short demo video, not an estimate of general factory performance. See the [full benchmark tables and reproduction commands](docs/benchmark/konro-results.md).

### Factory Ego

An in-progress comparison dataset of 20 units (20 seconds, 2 fps, 40 frames each) stratified across six factories and twenty work types in Egocentric-10K, aimed at procedure-compliance judgment. Clip selection uses the LLM-generated transcripts of [annotated-egocentric-10k-dataset](https://github.com/fit-alessandro-berti/annotated-egocentric-10k-dataset), which are never treated as ground truth; event definitions are authored by viewing the extracted frames (see docs/benchmark/events.md). Each SOP defines 3-4 procedure-step events with Japanese single-sentence questions. All current units are `dev_seen`. Human ground truth is not available yet, so formal precision, recall, F1, and tIoU are not reported. Because the upstream dataset is gated, extracted frames are excluded from the public repository and only SHA manifests are tracked. After accepting the upstream gated terms, `tools/benchmark/fetch_factory_ego.py` reconstructs byte-identical local media (see the [operations guide](docs/benchmark/operations.md)).

See the [Factory Ego dataset notes](datasets/factory_ego/README.md) and [current comparison report](reports/model_comparison.md).

## Repository layout

```text
src/            Python package and CLI
datasets/       media units, SOPs, and human annotations
runs/           immutable model predictions
evaluations/    prediction-to-ground-truth evaluations
reports/        cross-run comparisons
schemas/        benchmark JSON Schemas
tools/          migration and quality checks
tests/          unit and integration tests
docs/           design, operations, and decisions
```

Start with the [documentation index](docs/README.md), [SOP format](docs/reference/sop-format.md), and [benchmark overview](docs/benchmark/README.md).

## Development checks

```bash
python3 -m pip install -e ".[test]"
pytest
python3 tools/benchmark/validate.py
python3 tools/quality/check_docs.py
python3 tools/quality/check_public.py
```

## License

The code is released under the MIT License. See [LICENSE](LICENSE). External datasets and models remain subject to their respective licenses and terms.
