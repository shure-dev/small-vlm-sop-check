# industrial-vlm-temporal-grounding

[日本語](README.md) | [Documentation](docs/README.md)

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![Task: Temporal Grounding](https://img.shields.io/badge/task-temporal%20grounding-155eef)
![Models: ≤4B](https://img.shields.io/badge/models-%E2%89%A44B-7c3aed)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Status: experimental](https://img.shields.io/badge/status-experimental-orange)

**Toward a Connected Worker platform that understands factory procedures in real time and catches critical deviations before they cause harm.**

<p align="center">
  <img src="docs/assets/factory_ego_temporal_grounding.gif" alt="Factory Ego interface comparing video, human spans, Marlin-2B spans, and event-level tIoU" width="960"><br>
  <sub>Three Factory Ego experiments. Orange shows human spans; blue shows Marlin-2B predictions.</sub>
</p>

## Catch procedural mistakes before they become incidents

In a factory, a skipped step, incorrect order, unsafe tool operation, or missed inspection can lead to life-threatening incidents, equipment damage, quality defects, or production downtime. Reviewing recorded footage after the event cannot support the worker at the moment it matters.

The intended system continuously analyzes first-person video from a wearable camera with a small VLM. It should understand the current operation, completed steps, and steps that have not yet occurred, then notify a worker or supervisor when a critical deviation begins to emerge—before an incident or loss is final.

- **Support work as it happens** — frontline workers should be able to continue hands-free work while receiving an immediate indication of a skipped step or unsafe action
- **Prevent incidents and loss** — supervisors and manufacturing engineers should be able to respond before a deviation becomes an injury, defect, or equipment stop
- **Run locally with low latency** — models with at most 4B parameters are the primary target, allowing a future path to continuous inference on a wearable or nearby edge computer without sending sensitive footage to an external API
- **Accumulate site-specific knowledge** — precise event descriptions and spans can capture tools, parts, grips, and placements, while human corrections flow into fine-tuning and re-evaluation

As a foundation for that system, this repository focuses on **temporal grounding**: locating an event in time.

```text
Input:  video + "The worker turns a bag upside down and drops parts into a bin"
Output: the event's start/end timestamps, or absent
```

Real-time procedure analysis requires more than recognizing an object or action. The system must know when each operation starts and ends, in what order it occurs, and what has not occurred. Accurate timestamps can feed higher-level logic for the current step, omissions, order violations, and abnormal duration.

The current work first establishes this temporal capability on short clips. Temporal IoU (tIoU) measures the difference from human spans and provides one contract for improving event definitions, prompts, models, and training data. The purpose is to support safe and correct work while it is happening.

**The source of advantage is not a larger general-purpose model. It is the ability to encode safety- and quality-critical site procedures as precise event definitions and timestamps, then continuously transfer that knowledge into a small model that can eventually run in real time.**

## Validating temporal understanding with Factory Ego

The primary pilot uses 20 fixed industrial first-person clips from [Egocentric-10K](https://huggingface.co/datasets/builddotai/Egocentric-10K). Each clip is 20 seconds at 2 fps. A human watches the footage, writes Japanese event descriptions, and marks the reference spans. External machine-generated annotations are not used as ground truth.

Six clips with 25 reference occurrences are currently annotated and compared with Marlin-2B temporal-grounding output. The GIF above cycles through three examples.

| Factory Ego experiment | Events | mean tIoU |
|---|---:|---:|
| Metal stamping | 4 | 0.816 |
| Garment bagging | 4 | 0.725 |
| Shirt folding | 4 | 0.645 |

Across all 25 reference occurrences, mean tIoU is `0.516` and `tIoU@0.5` F1 is `0.708`. The current Marlin `find()` interface returns one span per query, so its mean tIoU on the 21 single-span event IDs is `0.591`.

These are development diagnostics on clips and prompts used during iteration, not held-out benchmark accuracy. Fixed inputs and raw outputs are in [`runs/20260714-factory_ego-marlin-2b-reviewed6-tuned/`](runs/20260714-factory_ego-marlin-2b-reviewed6-tuned/); metrics and hashes are in [`evaluations/factory_ego_marlin_reviewed6.json`](evaluations/factory_ego_marlin_reviewed6.json).

## Improvement loop

```mermaid
flowchart LR
    A[Operational video] --> B[Record events and spans<br/>by hand]
    B --> C[Video VLM<br/>predicts spans]
    C --> D[Compare on a timeline<br/>and with tIoU]
    D --> E[Improve definitions,<br/>prompts, and models]
    E --> C
    D --> F[Export training data]
    F --> E
```

One web app switches between two workspaces:

- **Annotation** — save Japanese event descriptions, multiple spans, and an explicit absent state
- **Results** — compare human and model spans on one timeline and inspect low-tIoU events first

Translation, inference, and training remain reproducible CLI stages outside the app. The original Japanese annotation is human ground truth and is never overwritten by a model prediction.

## Quick start

Python 3.10+ and ffmpeg are required. Factory Ego also requires accepting the access terms for `builddotai/Egocentric-10K` on Hugging Face.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test,fetch]"
python tools/benchmark/fetch_factory_ego.py --apply
python tools/benchmark/validate.py --require-media
sop-app --dataset factory_ego
```

Annotation follows a video-first workflow:

1. Watch the clip and decide which operational events need distinct labels.
2. Describe the visible actor, object, and action precisely in Japanese.
3. Add each occurrence at frame-level resolution.
4. Use still frames and one-frame movement to refine boundaries.
5. Review the full timeline for missing or overlapping events.

Edits autosave to `datasets/<dataset>/annotations/human/<unit>.json`. See the [annotation guide](docs/reference/annotator.md) for details.

Open the results in read-only mode with:

```bash
sop-view --dataset factory_ego
```

The writable `sop-app` is localhost-only. For network sharing, do not expose its unauthenticated editing API; use `sop-view --host 0.0.0.0`.

## Two inference methods

| Method | VLM input | VLM output | Span construction | Role |
|---|---|---|---|---|
| Temporal Grounding | Video + event description | Start/end timestamps, or absent | Preserve the VLM span output | **Primary** |
| Frame Classification | One image at a time + question | Per-frame yes/no | Convert the answer sequence with rules | Baseline |

The primary method gives a video to a Video VLM and asks the model to produce temporal spans directly. The frame-classification rule engine is a comparison experiment for handling duration, short noise, and repeated occurrences in a yes/no sequence; it does not replace temporal-grounding output.

Both methods normalize to the same prediction format and use the same tIoU evaluation.

```bash
sop-check eval \
  --ground-truth datasets/<dataset>/annotations/human/<unit>.json \
  --prediction runs/<run-id>/predictions/<unit>.json
```

## Bring your own video

```bash
sop-dataset init --dataset my_factory --name "My Factory"
sop-dataset add-video --dataset my_factory --unit clip_001 \
  --video /path/to/private.mp4
sop-app --dataset my_factory
```

Events do not need to be predefined on the command line. Reviewers create them while watching the video. See the [bring-your-own-data guide](docs/guides/bring-your-own-data.md).

## Training and data contract

Completed annotations can be exported with `sop-export-ms-swift` as video-SFT JSONL. LoRA/QLoRA runs use [ms-swift](docs/training/ms-swift.md) as the external backend, and pre/post-tuning models are compared under the same contract and split.

```text
datasets/       versioned metadata, event definitions, human GT, splits, hashes
data/           ignored videos, frames, audio, and preview media
runs/           immutable inference runs, raw output, normalized predictions
evaluations/    metrics locked to annotation and prediction hashes
training_runs/  training configuration and input locks; weights/logs ignored
```

Intervals use video-relative half-open seconds, `[start_s, end_s)`. Human facts, model predictions, and evaluations remain separate. See the [data contract](docs/benchmark/data-contract.md).

## Current scope

The current scope is offline annotation, prediction review, evaluation, and training export for short video clips. This stage builds the temporal model and ground truth required for real-time procedure analysis. Wearable deployment, streaming inference, alert logic, power consumption, and latency have not yet been validated.

The repository output alone is not intended to automate life- or equipment-critical safety decisions. Production deployment requires site-specific risk assessment, fail-safe controls, human oversight, and a clear boundary with existing safety systems.

Full videos, ordinary extracted frames, model weights, and personal data are not committed. The hero GIF is the only downsampled demonstration derivative containing Egocentric-10K imagery; attribution and terms are documented in [`docs/assets/README.md`](docs/assets/README.md).

## Validation

```bash
python -m pytest -q
python tools/benchmark/validate.py
sop-dataset validate --dataset factory_ego
python tools/quality/check_docs.py
python tools/quality/check_public.py
```

Code is released under the [MIT License](LICENSE). External data, models, and checkpoints retain their own licenses and access terms.
