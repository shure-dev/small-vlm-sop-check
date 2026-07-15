"""動画アノテーションSPA向けの小さなHTTP API。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .annotation_store import load_document, save_sop_and_document
from .catalog import Catalog, Unit, discover
from .comparison import discover_run_comparisons
from .media import preview_video
from .resources import repository_root


def _find_unit(catalog: Catalog, dataset: str, unit_id: str) -> Unit:
    unit = next(
        (item for item in catalog.units if item.dataset == dataset and item.unit_id == unit_id),
        None,
    )
    if unit is None:
        raise HTTPException(status_code=404, detail="動画が見つかりません")
    return unit


def _unit_summary(unit: Unit) -> dict[str, Any]:
    summary = unit.summary()
    if unit.gt_path.is_file():
        try:
            document = load_document(unit)
            known = {event["id"] for event in unit.load_sop().get("events", [])}
            summary["annotation_state"] = (
                "complete" if set(document["events"]) == known else "in_progress"
            )
        except (OSError, ValueError):
            summary["annotation_state"] = "invalid"
    else:
        summary["annotation_state"] = "not_started"
    return summary


def create_app(
    *,
    root: Path | None = None,
    read_only: bool = False,
    initial_dataset: str | None = None,
    initial_unit: str | None = None,
    frontend_dir: Path | None = None,
) -> FastAPI:
    """APIとビルド済みSPAを同じプロセスで配信する。"""
    repo_root = (root or repository_root()).resolve()
    app = FastAPI(title="Small VLM Video Annotation", docs_url="/api/docs")

    def catalog() -> Catalog:
        return discover(repo_root)

    @app.get("/api/bootstrap")
    def bootstrap() -> dict[str, Any]:
        current = catalog()
        units = [_unit_summary(unit) for unit in current.units]
        datasets = list(dict.fromkeys(unit["dataset"] for unit in units))
        return {
            "datasets": datasets,
            "units": units,
            "read_only": read_only,
            "initial_dataset": initial_dataset,
            "initial_unit": initial_unit,
        }

    @app.get("/api/units/{dataset}/{unit_id}")
    def get_unit(dataset: str, unit_id: str) -> dict[str, Any]:
        unit = _find_unit(catalog(), dataset, unit_id)
        return {
            **_unit_summary(unit),
            "n_frames": unit.n_frames or len(unit.frame_paths()),
            "sop": unit.load_sop(),
            "annotation": load_document(unit),
            "media_url": f"/api/units/{dataset}/{unit_id}/media",
            "frame_url_template": f"/api/units/{dataset}/{unit_id}/frames/{{index}}",
        }

    @app.put("/api/units/{dataset}/{unit_id}")
    def put_unit(dataset: str, unit_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        if read_only:
            raise HTTPException(status_code=403, detail="読み取り専用モードです")
        unit = _find_unit(catalog(), dataset, unit_id)
        sop = payload.get("sop")
        annotation = payload.get("annotation")
        if not isinstance(sop, dict) or not isinstance(annotation, dict):
            raise HTTPException(status_code=422, detail="sopとannotationが必要です")
        try:
            save_sop_and_document(unit, sop, annotation)
        except (KeyError, OSError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"ok": True, "annotation": load_document(unit), "sop": unit.load_sop()}

    @app.get("/api/units/{dataset}/{unit_id}/media")
    def get_media(dataset: str, unit_id: str) -> FileResponse:
        unit = _find_unit(catalog(), dataset, unit_id)
        try:
            path = preview_video(unit)
        except RuntimeError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if path is None:
            raise HTTPException(status_code=404, detail="動画または抽出フレームが見つかりません")
        return FileResponse(path, media_type="video/mp4", filename=f"{unit_id}.mp4")

    @app.get("/api/units/{dataset}/{unit_id}/frames/{index}")
    def get_frame(dataset: str, unit_id: str, index: int) -> FileResponse:
        unit = _find_unit(catalog(), dataset, unit_id)
        frames = unit.frame_paths()
        if index < 0 or index >= len(frames):
            raise HTTPException(status_code=404, detail="フレームが見つかりません")
        return FileResponse(frames[index], media_type="image/jpeg")

    @app.get("/api/comparisons/{dataset}/{unit_id}")
    def get_comparisons(dataset: str, unit_id: str) -> dict[str, Any]:
        unit = _find_unit(catalog(), dataset, unit_id)
        result: list[dict[str, Any]] = []
        for run_id, comparison in discover_run_comparisons(repo_root).items():
            if comparison.dataset_id != dataset or comparison.reference_revision != "human":
                continue
            unit_comparison = comparison.for_unit(unit)
            if unit_comparison is not None:
                result.append(
                    {
                        "run_id": run_id,
                        "model": comparison.run.get("model", {}),
                        "comparison": unit_comparison,
                    }
                )
        return {"runs": result}

    static_dir = frontend_dir or Path(__file__).with_name("frontend_dist")
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")

    return app
