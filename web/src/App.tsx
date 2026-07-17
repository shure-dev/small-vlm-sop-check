import { Fragment, useEffect, useMemo, useRef, useState } from "react";

type Span = { start_s: number; end_s: number };
type SopEvent = { id: string; ask: string; values?: string[] };
type Sop = { sop?: Record<string, unknown>; events: SopEvent[]; [key: string]: unknown };
type Annotation = {
  unit_id: string;
  annotation_revision: string;
  interval_convention: string;
  event_labels: Record<string, string>;
  events: Record<string, Span[] | null>;
};
type UnitSummary = {
  dataset: string;
  unit_id: string;
  fps: number;
  duration_s: number;
  annotation_state: string;
  excluded?: boolean;
};
type UnitData = UnitSummary & {
  n_frames: number;
  sop: Sop;
  annotation: Annotation;
  media_url: string;
  frame_url_template: string;
};
type Bootstrap = {
  datasets: string[];
  units: UnitSummary[];
  read_only: boolean;
  initial_dataset?: string;
  initial_unit?: string;
};
type ComparisonEvent = {
  event_id: string;
  text: string;
  reference_spans: Span[] | null;
  prediction_spans: Span[] | null;
  tiou: number | null;
};
type ComparisonRun = {
  run_id: string;
  model: { id?: string; name?: string };
  comparison: {
    summary: {
      mean_tiou: number | null;
      gt_occurrences: number;
      predicted_occurrences: number;
      thresholds: Record<string, { f1: number | null }>;
    };
    events: ComparisonEvent[];
  };
};
type TimelineRow = { id: string; label: string; human: Span[] | null | undefined; model: Span[] | null | undefined; tiou: number | null };

const modelKey = (run: ComparisonRun) => run.model.id || run.model.name || "unknown-model";

const clone = <T,>(value: T): T => structuredClone(value);
const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));
const time = (value: number) => `${value.toFixed(1)}秒`;

async function getJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${response.status}`);
  }
  return response.json();
}

function mergeSpans(spans: Span[]): Span[] {
  const sorted = spans.filter((span) => span.end_s > span.start_s).slice().sort((a, b) => a.start_s - b.start_s);
  const merged: Span[] = [];
  for (const span of sorted) {
    const last = merged[merged.length - 1];
    if (last && span.start_s <= last.end_s) last.end_s = Math.max(last.end_s, span.end_s);
    else merged.push({ ...span });
  }
  return merged;
}

function spanSetIoU(a: Span[] | null | undefined, b: Span[] | null | undefined): number | null {
  const left = mergeSpans(a ?? []);
  const right = mergeSpans(b ?? []);
  if (!left.length && !right.length) return null;
  const length = (spans: Span[]) => spans.reduce((sum, span) => sum + (span.end_s - span.start_s), 0);
  let intersection = 0;
  for (const x of left) for (const y of right) intersection += Math.max(0, Math.min(x.end_s, y.end_s) - Math.max(x.start_s, y.start_s));
  const union = length(left) + length(right) - intersection;
  return union > 0 ? intersection / union : null;
}

const TICK_STEPS = [0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300];
function buildTicks(duration: number) {
  const interval = TICK_STEPS.find((value) => duration / value <= 12) ?? 600;
  const ticks: number[] = [];
  for (let value = 0; value <= duration + 1e-6; value += interval) ticks.push(Math.round(value * 100) / 100);
  return { interval, ticks };
}

function formatSpans(spans: Span[] | null) {
  if (spans === null) return "該当なし";
  if (!spans?.length) return "—";
  return spans.map((span) => `${Number(span.start_s.toFixed(2))}–${Number(span.end_s.toFixed(2))}秒`).join(", ");
}

function computeLiveStats(unit: UnitData, run: ComparisonRun) {
  const predictions = new Map(run.comparison.events.map((event) => [event.event_id, event.prediction_spans]));
  const scores: number[] = [];
  let tp = 0, fp = 0, fn = 0, gtCount = 0, predCount = 0;
  unit.sop.events.forEach((event) => {
    const gt = unit.annotation.events[event.id];
    const pred = predictions.get(event.id) ?? null;
    gtCount += gt?.length ?? 0;
    predCount += pred?.length ?? 0;
    const iou = spanSetIoU(gt, pred);
    if (iou !== null) scores.push(iou);
    const hasGt = !!gt?.length;
    const hasPred = !!pred?.length;
    if (hasGt && hasPred && (iou ?? 0) >= 0.5) tp += 1;
    else {
      if (hasPred) fp += 1;
      if (hasGt) fn += 1;
    }
  });
  return {
    mean: scores.length ? scores.reduce((sum, value) => sum + value, 0) / scores.length : null,
    f1: tp + fp + fn > 0 ? (2 * tp) / (2 * tp + fp + fn) : null,
    gtCount,
    predCount,
  };
}

const STATE_LABEL: Record<string, string> = { complete: "完了", in_progress: "途中", invalid: "要修正", not_started: "未着手" };
const STATE_MARK: Record<string, string> = { complete: "✓", in_progress: "△", invalid: "！" };
const tiouClass = (value: number) => (value < 0.5 ? "bad" : value < 0.6 ? "check" : "good");

function Sidebar({ bootstrap, dataset, unitId, allRuns, modelId, modelOptions, liveMean, onDataset, onUnit, onModel }: {
  bootstrap: Bootstrap;
  dataset: string;
  unitId: string;
  allRuns: Record<string, ComparisonRun[]>;
  modelId: string;
  modelOptions: [string, string][];
  liveMean?: number | null;
  onDataset: (value: string) => void;
  onUnit: (value: string) => void;
  onModel: (value: string) => void;
}) {
  const [sortBy, setSortBy] = useState<"name" | "tiou">("name");
  const units = bootstrap.units.filter((unit) => unit.dataset === dataset);
  const done = units.filter((unit) => unit.annotation_state === "complete").length;
  const unitTiou = (id: string) => {
    if (id === unitId && liveMean !== undefined) return liveMean;
    return allRuns[id]?.find((run) => modelKey(run) === modelId)?.comparison.summary.mean_tiou ?? null;
  };
  const scores = units.map((unit) => unitTiou(unit.unit_id)).filter((value): value is number => value !== null);
  const average = scores.length ? scores.reduce((sum, value) => sum + value, 0) / scores.length : null;
  const sorted = sortBy === "name" || !modelId ? units : [...units].sort((a, b) => {
    const scoreA = unitTiou(a.unit_id);
    const scoreB = unitTiou(b.unit_id);
    if (scoreA === null && scoreB === null) return a.unit_id.localeCompare(b.unit_id);
    if (scoreA === null) return 1;
    if (scoreB === null) return -1;
    return scoreA - scoreB;
  });
  return (
    <aside className="sidebar">
      <header className="sb-head">
        <h1>動画イベント</h1>
        <p className="sb-progress">
          <span>アノテーション <strong>{done} / {units.length}</strong> 完了{units.some((unit) => unit.excluded) ? <>　除外 <strong>{units.filter((unit) => unit.excluded).length}</strong></> : null}</span>
          {average !== null && <span>平均 tIoU <strong className={tiouClass(average)}>{average.toFixed(3)}</strong></span>}
        </p>
        <div className="sort-switch" role="group" aria-label="並び順">
          <button className={sortBy === "name" ? "active" : ""} onClick={() => setSortBy("name")}>名前順</button>
          <button className={sortBy === "tiou" ? "active" : ""} disabled={!modelId} onClick={() => setSortBy("tiou")}>tIoU低い順</button>
        </div>
      </header>
      <div className="video-gallery" role="list" aria-label="動画一覧">
        {sorted.map((unit) => {
          const tiou = unitTiou(unit.unit_id);
          return (
            <button key={unit.unit_id} role="listitem" className={`video-card ${unit.unit_id === unitId ? "selected" : ""} ${unit.excluded ? "excluded" : ""}`}
              title={`${unit.unit_id}（${STATE_LABEL[unit.annotation_state] ?? unit.annotation_state}${unit.excluded ? "・除外中" : ""}）`}
              onClick={() => onUnit(unit.unit_id)}>
              <span className="thumb">
                <img src={`/api/units/${dataset}/${unit.unit_id}/frames/${Math.max(0, Math.round(unit.fps * unit.duration_s / 2) - 1)}`} loading="lazy" alt=""
                  onError={(event) => {
                    const fallback = `/api/units/${dataset}/${unit.unit_id}/frames/0`;
                    if (event.currentTarget.src.endsWith("/frames/0")) event.currentTarget.style.visibility = "hidden";
                    else event.currentTarget.src = fallback;
                  }} />
                {tiou !== null && <i className={`score ${tiouClass(tiou)}`}>tIoU {tiou.toFixed(2)}</i>}
                {unit.annotation_state !== "complete" && <i className={`state ${unit.annotation_state}`}>{STATE_MARK[unit.annotation_state] ? `${STATE_MARK[unit.annotation_state]} ` : ""}{STATE_LABEL[unit.annotation_state] ?? unit.annotation_state}</i>}
                {unit.excluded && <i className="excluded-tag">除外</i>}
              </span>
              <span className="video-name">{unit.unit_id}</span>
            </button>
          );
        })}
      </div>
      <footer className="sb-foot">
        <label className="field-label" htmlFor="dataset">データセット</label>
        <select id="dataset" value={dataset} onChange={(event) => onDataset(event.target.value)}>
          {bootstrap.datasets.map((name) => <option key={name}>{name}</option>)}
        </select>
        <label className="field-label" htmlFor="model">モデル予測</label>
        <select id="model" value={modelId} onChange={(event) => onModel(event.target.value)} disabled={!modelOptions.length}>
          {modelOptions.length ? <>
            <option value="">表示しない</option>
            {modelOptions.map(([id, label]) => <option key={id} value={id}>{label}</option>)}
          </> : <option value="">推論結果なし</option>}
        </select>
      </footer>
    </aside>
  );
}

function TimelineEditor({ rows, duration, current, selectedId, editable, hasModel, onSeek, onSelect, onChange, onAdd }: {
  rows: TimelineRow[];
  duration: number;
  current: number;
  selectedId: string;
  editable: boolean;
  hasModel: boolean;
  onSeek: (value: number) => void;
  onSelect: (id: string) => void;
  onChange?: (id: string, spans: Span[]) => void;
  onAdd?: () => void;
}) {
  const [preview, setPreview] = useState<{ id: string; start_s: number; end_s: number } | null>(null);
  const safeDuration = Math.max(duration, 0.001);
  const { interval, ticks } = useMemo(() => buildTicks(safeDuration), [safeDuration]);
  const pct = (value: number) => `${(value / safeDuration) * 100}%`;
  const timeAt = (track: HTMLElement, clientX: number) => {
    const rect = track.getBoundingClientRect();
    return clamp(((clientX - rect.left) / rect.width) * safeDuration, 0, duration);
  };
  const drag = (move: (pointer: PointerEvent) => void, up?: () => void) => {
    const onUp = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", onUp); up?.(); };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", onUp);
  };
  const scrub = (event: React.PointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    const ruler = event.currentTarget;
    onSeek(timeAt(ruler, event.clientX));
    drag((pointer) => onSeek(timeAt(ruler, pointer.clientX)));
  };
  const trackDown = (event: React.PointerEvent<HTMLDivElement>, row: TimelineRow) => {
    if ((event.target as HTMLElement).closest(".clip")) return;
    event.preventDefault();
    const track = event.currentTarget;
    const origin = event.clientX;
    const start = timeAt(track, origin);
    let created: Span | null = null;
    drag((pointer) => {
      if (created === null && Math.abs(pointer.clientX - origin) <= 5) return;
      if (!editable || !onChange) { onSeek(timeAt(track, pointer.clientX)); return; }
      const value = timeAt(track, pointer.clientX);
      created = { start_s: Math.min(start, value), end_s: Math.max(start, value) };
      setPreview({ id: row.id, ...created });
    }, () => {
      setPreview(null);
      if (created) {
        onChange?.(row.id, [...(row.human ?? []), { start_s: created.start_s, end_s: Math.max(created.end_s, created.start_s + 0.05) }]);
        onSelect(row.id);
      } else onSeek(start);
    });
  };
  const clipDown = (event: React.PointerEvent<HTMLDivElement>, row: TimelineRow, index: number) => {
    if ((event.target as HTMLElement).closest(".handle")) return;
    event.preventDefault();
    event.stopPropagation();
    const track = event.currentTarget.closest(".tl-track") as HTMLElement;
    const rect = track.getBoundingClientRect();
    const origin = event.clientX;
    const original = clone(row.human ?? []);
    let moved = false;
    drag((pointer) => {
      if (!editable || !onChange) return;
      if (!moved && Math.abs(pointer.clientX - origin) <= 4) return;
      moved = true;
      const delta = ((pointer.clientX - origin) / rect.width) * safeDuration;
      const width = original[index].end_s - original[index].start_s;
      const start_s = clamp(original[index].start_s + delta, 0, Math.max(0, duration - width));
      const next = clone(original);
      next[index] = { start_s, end_s: start_s + width };
      onChange(row.id, next);
    }, () => { if (!moved) onSelect(row.id); });
  };
  const handleDown = (event: React.PointerEvent, row: TimelineRow, index: number, edge: "start" | "end") => {
    if (!onChange || !row.human) return;
    event.preventDefault();
    event.stopPropagation();
    const track = (event.currentTarget as HTMLElement).closest(".tl-track") as HTMLElement;
    const original = clone(row.human);
    drag((pointer) => {
      const value = timeAt(track, pointer.clientX);
      const next = clone(original);
      if (edge === "start") next[index].start_s = Math.min(value, next[index].end_s - 0.001);
      else next[index].end_s = Math.max(value, next[index].start_s + 0.001);
      onChange(row.id, next);
      onSeek(value);
    });
  };
  return (
    <section className="timeline" aria-label="タイムライン" style={{ "--cursor": clamp(current / safeDuration, 0, 1) } as React.CSSProperties}>
      <div className="tl-scroll">
        <div className="tl-grid">
          <div className="tl-corner">
            {onAdd && <button className="add-event" onClick={onAdd}>＋ イベント追加</button>}
          </div>
          <div className="tl-ruler" onPointerDown={scrub} aria-label="時間ルーラー">
            {ticks.map((tick) => <span className="tick" key={tick} style={{ left: pct(tick) }}><label>{interval < 1 ? tick.toFixed(1) : Math.round(tick)}秒</label></span>)}
            <span className="playhead-cap" />
          </div>
          {rows.map((row, index) => <Fragment key={row.id}>
            <button className={`tl-label ${row.id === selectedId ? "selected" : ""}`} onClick={() => onSelect(row.id)}>
              <span className="idx">{index + 1}</span>
              <span className="text">{row.label}</span>
              {hasModel && <span className={`tiou ${row.tiou === null ? "none" : row.tiou < 0.5 ? "bad" : row.tiou < 0.6 ? "check" : "good"}`}>{row.tiou === null ? "—" : row.tiou.toFixed(2)}</span>}
            </button>
            <div className={`tl-track ${hasModel ? "" : "single"}`} style={{ backgroundSize: `${(interval / safeDuration) * 100}% 100%` }} onPointerDown={(event) => trackDown(event, row)}>
              {row.human === null && <span className="absent-tag">動画内で発生しない</span>}
              {(row.human ?? []).map((span, spanIndex) => (
                <div className={`clip human ${row.id === selectedId ? "selected" : ""}`} key={`h-${spanIndex}`}
                  style={{ left: pct(span.start_s), width: pct(span.end_s - span.start_s) }}
                  title={`人手 ${time(span.start_s)}–${time(span.end_s)}`}
                  onPointerDown={(event) => clipDown(event, row, spanIndex)}>
                  {editable && <>
                    <span className="handle start" onPointerDown={(event) => handleDown(event, row, spanIndex, "start")} />
                    <span className="handle end" onPointerDown={(event) => handleDown(event, row, spanIndex, "end")} />
                  </>}
                </div>
              ))}
              {preview?.id === row.id && <div className="clip preview" style={{ left: pct(preview.start_s), width: pct(preview.end_s - preview.start_s) }} />}
              {hasModel && (row.model ?? []).map((span, spanIndex) => (
                <div className="clip model" key={`m-${spanIndex}`}
                  style={{ left: pct(span.start_s), width: pct(span.end_s - span.start_s) }}
                  title={`モデル ${time(span.start_s)}–${time(span.end_s)}`}
                  onPointerDown={(event) => event.stopPropagation()}
                  onClick={() => onSelect(row.id)} />
              ))}
            </div>
          </Fragment>)}
          {!rows.length && <div className="tl-empty">イベントがまだありません。「＋ イベント追加」から、映像で見える動作を登録してください。</div>}
          <span className="playhead" />
        </div>
      </div>
    </section>
  );
}

function Inspector({ unit, event, index, current, readOnly, comparisonEvent, tiou, backup, setBackup, mutate, onClose, onDelete }: {
  unit: UnitData;
  event: SopEvent;
  index: number;
  current: number;
  readOnly: boolean;
  comparisonEvent?: ComparisonEvent;
  tiou: number | null;
  backup: Record<string, Span[]>;
  setBackup: React.Dispatch<React.SetStateAction<Record<string, Span[]>>>;
  mutate: (fn: (next: UnitData) => void) => void;
  onClose: () => void;
  onDelete: () => void;
}) {
  const spans = unit.annotation.events[event.id];
  const step = 1 / unit.fps;
  const update = (value: Span[] | null) => mutate((next) => { next.annotation.events[event.id] = value; });
  return (
    <aside className="inspector" aria-label={`イベント${index + 1}の編集`}>
      <header>
        <span className="eyebrow">イベント {index + 1}</span>
        <button className="close" onClick={onClose} aria-label="閉じる">✕</button>
      </header>
      <label className="field-label" htmlFor={`label-${event.id}`}>日本語イベント文</label>
      <textarea id={`label-${event.id}`} rows={3} value={event.ask} readOnly={readOnly} onChange={(change) => mutate((next) => {
        const target = next.sop.events.find((item) => item.id === event.id)!;
        target.ask = change.target.value;
        next.annotation.event_labels[event.id] = change.target.value;
      })} />
      <label className="absent"><input type="checkbox" checked={spans === null} disabled={readOnly} onChange={(change) => {
        if (change.target.checked) { if (spans) setBackup((value) => ({ ...value, [event.id]: spans })); update(null); }
        else { const start = clamp(current, 0, Math.max(0, unit.duration_s - step)); update(backup[event.id] || [{ start_s: start, end_s: Math.min(unit.duration_s, start + Math.max(1, step)) }]); }
      }} />このイベントは動画内で起きていない</label>
      {spans !== null && <div className="span-list">
        {(spans || []).map((span, spanIndex) => <div className="span-editor" key={spanIndex}>
          <div className="span-title"><strong>区間 {spanIndex + 1}</strong><span>{time(span.start_s)} – {time(span.end_s)}</span></div>
          <div className="span-inputs">
            <label>開始<input type="number" min="0" max={span.end_s - step} step={step} value={span.start_s} readOnly={readOnly} onChange={(change) => { const next = clone(spans || []); next[spanIndex].start_s = Number(change.target.value); update(next); }} /></label>
            <button disabled={readOnly} onClick={() => { const next = clone(spans || []); next[spanIndex].start_s = Math.min(current, next[spanIndex].end_s - step); update(next); }}>現在位置</button>
            <label>終了<input type="number" min={span.start_s + step} max={unit.duration_s} step={step} value={span.end_s} readOnly={readOnly} onChange={(change) => { const next = clone(spans || []); next[spanIndex].end_s = Number(change.target.value); update(next); }} /></label>
            <button disabled={readOnly} onClick={() => { const next = clone(spans || []); next[spanIndex].end_s = Math.max(current, next[spanIndex].start_s + step); update(next); }}>現在位置</button>
            <button className="danger-text" disabled={readOnly || spans?.length === 1} onClick={() => update((spans || []).filter((_, item) => item !== spanIndex))}>削除</button>
          </div>
        </div>)}
        {!readOnly && <button className="wide" onClick={() => { const start = clamp(current, 0, Math.max(0, unit.duration_s - step)); update([...(spans || []), { start_s: start, end_s: Math.min(unit.duration_s, start + Math.max(1, step)) }].sort((a, b) => a.start_s - b.start_s)); }}>＋ 区間を追加（現在位置から）</button>}
      </div>}
      {comparisonEvent && <div className="prediction-note">
        <span className="chip model">モデル予測</span>
        <p>{formatSpans(comparisonEvent.prediction_spans)}{tiou !== null ? `　tIoU ${tiou.toFixed(3)}` : ""}</p>
      </div>}
      {!readOnly && <details className="danger-zone"><summary>イベントを削除</summary>
        <button className="danger" onClick={() => window.confirm("このイベントを削除しますか？") && onDelete()}>削除する</button>
      </details>}
    </aside>
  );
}

function EditorWorkspace({ unit, run, readOnly, onChange }: {
  unit: UnitData;
  run?: ComparisonRun;
  readOnly: boolean;
  onChange: (next: UnitData) => void;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [current, setCurrent] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [rate, setRate] = useState(1);
  const [showFrame, setShowFrame] = useState(false);
  const [selectedId, setSelectedId] = useState("");
  const [creating, setCreating] = useState(false);
  const [draftLabel, setDraftLabel] = useState("");
  const [backup, setBackup] = useState<Record<string, Span[]>>({});
  const step = 1 / unit.fps;
  useEffect(() => { setCurrent(0); setPlaying(false); setSelectedId(""); setCreating(false); setDraftLabel(""); setBackup({}); }, [unit.unit_id]);
  useEffect(() => { if (videoRef.current) videoRef.current.playbackRate = rate; }, [rate]);
  useEffect(() => {
    if (!playing) return;
    let raf = requestAnimationFrame(function tick() {
      const video = videoRef.current;
      if (video) setCurrent(video.currentTime);
      raf = requestAnimationFrame(tick);
    });
    return () => cancelAnimationFrame(raf);
  }, [playing]);
  const seek = (value: number) => {
    const next = clamp(value, 0, Math.max(0, unit.duration_s - step));
    if (videoRef.current) videoRef.current.currentTime = next;
    setCurrent(next);
  };
  const togglePlayback = () => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) void video.play();
    else video.pause();
  };
  const mutate = (fn: (next: UnitData) => void) => { const next = clone(unit); fn(next); onChange(next); };
  const updateSpans = (eventId: string, spans: Span[]) => mutate((next) => {
    next.annotation.events[eventId] = spans.map((span) => {
      const start_s = clamp(Math.round(span.start_s * unit.fps) / unit.fps, 0, Math.max(0, unit.duration_s - step));
      return { start_s, end_s: clamp(Math.round(span.end_s * unit.fps) / unit.fps, start_s + step, unit.duration_s) };
    }).sort((a, b) => a.start_s - b.start_s);
  });
  const setBoundary = (edge: "start" | "end") => {
    if (readOnly || !selectedId) return;
    const spans = unit.annotation.events[selectedId];
    if (!spans?.length) return;
    let index = spans.findIndex((span) => current >= span.start_s && current <= span.end_s);
    if (index < 0) {
      let best = Infinity;
      spans.forEach((span, spanIndex) => {
        const distance = current < span.start_s ? span.start_s - current : current - span.end_s;
        if (distance < best) { best = distance; index = spanIndex; }
      });
    }
    const next = clone(spans);
    if (edge === "start") next[index].start_s = Math.min(current, next[index].end_s - step);
    else next[index].end_s = Math.max(current, next[index].start_s + step);
    updateSpans(selectedId, next);
  };
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target?.isContentEditable || (target && ["INPUT", "TEXTAREA", "SELECT", "BUTTON"].includes(target.tagName))) return;
      if (event.code === "Space") { event.preventDefault(); togglePlayback(); }
      else if (event.key === "ArrowLeft") { event.preventDefault(); seek(current - (event.shiftKey ? 1 : step)); }
      else if (event.key === "ArrowRight") { event.preventDefault(); seek(current + (event.shiftKey ? 1 : step)); }
      else if (event.key === "Home") { event.preventDefault(); seek(0); }
      else if (event.key === "End") { event.preventDefault(); seek(unit.duration_s); }
      else if (event.key === "i" || event.key === "I") { event.preventDefault(); setBoundary("start"); }
      else if (event.key === "o" || event.key === "O") { event.preventDefault(); setBoundary("end"); }
      else if (event.key === "Escape") { setSelectedId(""); setCreating(false); }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  });
  const comparison = useMemo(() => {
    const map = new Map<string, ComparisonEvent>();
    run?.comparison.events.forEach((event) => map.set(event.event_id, event));
    return map;
  }, [run]);
  const rows: TimelineRow[] = unit.sop.events.map((event) => {
    const human = unit.annotation.events[event.id];
    const prediction = run ? comparison.get(event.id)?.prediction_spans ?? [] : undefined;
    return { id: event.id, label: event.ask, human, model: prediction, tiou: run ? spanSetIoU(human, prediction) : null };
  });
  const addEvent = () => {
    const label = draftLabel.trim();
    if (!label) return;
    const used = new Set(unit.sop.events.map((event) => event.id));
    let index = 1;
    while (used.has(`event_${String(index).padStart(3, "0")}`)) index += 1;
    const id = `event_${String(index).padStart(3, "0")}`;
    const start = clamp(current, 0, Math.max(0, unit.duration_s - step));
    mutate((next) => {
      next.sop.events.push({ id, ask: label, values: ["yes", "no"] });
      next.annotation.event_labels[id] = label;
      next.annotation.events[id] = [{ start_s: start, end_s: Math.min(unit.duration_s, start + Math.max(1, step)) }];
    });
    setSelectedId(id);
    setCreating(false);
    setDraftLabel("");
  };
  const selectedIndex = unit.sop.events.findIndex((event) => event.id === selectedId);
  const selected = selectedIndex >= 0 ? unit.sop.events[selectedIndex] : undefined;
  const frame = Math.min(unit.n_frames - 1, Math.max(0, Math.round(current * unit.fps)));
  return (
    <div className="editor">
      <section className="stage" aria-label="動画">
        <video ref={videoRef} src={unit.media_url} preload="metadata"
          onClick={togglePlayback}
          onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} onEnded={() => setPlaying(false)}
          onTimeUpdate={(event) => { if (!playing) setCurrent(event.currentTarget.currentTime); }}
          onLoadedMetadata={(event) => { event.currentTarget.playbackRate = rate; }} />
        {showFrame && <img className="frame-peek" src={unit.frame_url_template.replace("{index}", String(frame))} alt={`${time(current)}のフレーム`} />}
      </section>
      <div className="transport">
        <button className="play" onClick={togglePlayback} aria-label={playing ? "一時停止" : "再生"}>{playing ? "❚❚" : "▶"}</button>
        <div className="timecode"><strong>{current.toFixed(2)}</strong><span> / {unit.duration_s.toFixed(2)} 秒</span></div>
        <button className="step" onClick={() => seek(current - step)} disabled={current <= 0}>◀ 1フレーム</button>
        <button className="step" onClick={() => seek(current + step)} disabled={current >= unit.duration_s - step}>1フレーム ▶</button>
        <select className="rate" aria-label="再生速度" value={rate} onChange={(event) => setRate(Number(event.target.value))}>
          {[0.25, 0.5, 1, 1.5, 2].map((value) => <option key={value} value={value}>×{value}</option>)}
        </select>
        <button className={showFrame ? "frame-toggle active" : "frame-toggle"} aria-pressed={showFrame} onClick={() => setShowFrame(!showFrame)}>静止画</button>
        <span className="spacer" />
        <span className="chip human">人手区間</span>
        {run && <span className="chip model">モデル予測</span>}
      </div>
      <TimelineEditor rows={rows} duration={unit.duration_s} current={current} selectedId={creating ? "" : selectedId}
        editable={!readOnly} hasModel={!!run}
        onSeek={seek}
        onSelect={(id) => { setSelectedId(id); setCreating(false); }}
        onChange={readOnly ? undefined : updateSpans}
        onAdd={readOnly ? undefined : () => { setCreating(true); setSelectedId(""); setDraftLabel(""); }} />
      <footer className="hints">
        <span><kbd>Space</kbd>再生 / 停止</span>
        <span><kbd>←</kbd><kbd>→</kbd>1フレーム</span>
        <span><kbd>Shift</kbd>+<kbd>←→</kbd>1秒</span>
        <span><kbd>I</kbd><kbd>O</kbd>選択イベントの開始 / 終了を現在位置へ</span>
        <span><kbd>Esc</kbd>パネルを閉じる</span>
      </footer>
      {creating && !readOnly && <aside className="inspector" aria-label="イベント作成">
        <header><span className="eyebrow">新しいイベント</span><button className="close" onClick={() => setCreating(false)} aria-label="閉じる">✕</button></header>
        <label className="field-label" htmlFor="new-label">日本語イベント文</label>
        <textarea id="new-label" rows={3} value={draftLabel} onChange={(event) => setDraftLabel(event.target.value)} placeholder="映像で見える動作を具体的に書く" autoFocus />
        <button className="primary wide" disabled={!draftLabel.trim()} onClick={addEvent}>現在位置から作成</button>
      </aside>}
      {selected && !creating && <Inspector unit={unit} event={selected} index={selectedIndex} current={current} readOnly={readOnly}
        comparisonEvent={comparison.get(selected.id)} tiou={rows[selectedIndex]?.tiou ?? null}
        backup={backup} setBackup={setBackup} mutate={mutate}
        onClose={() => setSelectedId("")}
        onDelete={() => {
          mutate((next) => {
            next.sop.events = next.sop.events.filter((event) => event.id !== selected.id);
            delete next.annotation.events[selected.id];
            delete next.annotation.event_labels[selected.id];
          });
          setSelectedId("");
        }} />}
    </div>
  );
}

export default function App() {
  const [bootstrap, setBootstrap] = useState<Bootstrap | null>(null);
  const [dataset, setDataset] = useState("");
  const [unitId, setUnitId] = useState("");
  const [unit, setUnit] = useState<UnitData | null>(null);
  const [allRuns, setAllRuns] = useState<Record<string, ComparisonRun[]>>({});
  const [modelId, setModelId] = useState("");
  const [showPredictions, setShowPredictions] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [saveState, setSaveState] = useState<"idle" | "saving" | "saved" | "error">("idle");
  const [error, setError] = useState("");
  const saveTimer = useRef<number | undefined>(undefined);
  useEffect(() => { getJson<Bootstrap>("/api/bootstrap").then((value) => {
    setBootstrap(value);
    const initialDataset = value.initial_dataset && value.datasets.includes(value.initial_dataset) ? value.initial_dataset : value.datasets[0] || "";
    const units = value.units.filter((item) => item.dataset === initialDataset);
    setDataset(initialDataset);
    setUnitId(value.initial_unit && units.some((item) => item.unit_id === value.initial_unit) ? value.initial_unit : units[0]?.unit_id || "");
  }).catch((reason) => setError(String(reason))); }, []);
  useEffect(() => {
    if (!bootstrap || !dataset) return;
    const units = bootstrap.units.filter((item) => item.dataset === dataset);
    let cancelled = false;
    setAllRuns({});
    Promise.all(units.map((item) =>
      getJson<{ runs: ComparisonRun[] }>(`/api/comparisons/${dataset}/${item.unit_id}`)
        .then((value) => [item.unit_id, value.runs] as const)
        .catch(() => [item.unit_id, []] as const),
    )).then((entries) => { if (!cancelled) setAllRuns(Object.fromEntries(entries)); });
    return () => { cancelled = true; };
  }, [bootstrap, dataset]);
  const modelOptions = useMemo(() => {
    const models = new Map<string, string>();
    Object.values(allRuns).forEach((runs) => {
      runs.forEach((run) => {
        const key = modelKey(run);
        models.set(key, run.model.name || key);
      });
    });
    return [...models.entries()];
  }, [allRuns]);
  useEffect(() => {
    if (!modelOptions.length) { setModelId(""); return; }
    setModelId((current) => modelOptions.some(([id]) => id === current) ? current : modelOptions[0][0]);
  }, [modelOptions]);
  useEffect(() => {
    if (!dataset || !unitId) return;
    setUnit(null);
    getJson<UnitData>(`/api/units/${dataset}/${unitId}`).then((value) => { setUnit(value); setSaveState("idle"); setError(""); }).catch((reason) => setError(String(reason)));
  }, [dataset, unitId]);
  const persist = async (next: UnitData) => {
    setSaveState("saving");
    try {
      await getJson(`/api/units/${next.dataset}/${next.unit_id}`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ sop: next.sop, annotation: next.annotation }) });
      setSaveState("saved");
      setError("");
    } catch (reason) {
      setSaveState("error");
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  };
  const changeUnit = (next: UnitData) => { setUnit(next); setSaveState("saving"); window.clearTimeout(saveTimer.current); saveTimer.current = window.setTimeout(() => void persist(next), 450); };
  const changeDataset = (value: string) => { setDataset(value); setUnitId(bootstrap?.units.find((item) => item.dataset === value)?.unit_id || ""); };
  const currentSummary = bootstrap?.units.find((item) => item.dataset === dataset && item.unit_id === unitId);
  const toggleExcluded = async () => {
    if (!bootstrap || !currentSummary) return;
    const next = !currentSummary.excluded;
    try {
      await getJson(`/api/curation/${dataset}/${unitId}`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ excluded: next }) });
      setBootstrap({ ...bootstrap, units: bootstrap.units.map((item) => item.dataset === dataset && item.unit_id === unitId ? { ...item, excluded: next } : item) });
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  };
  if (error && !bootstrap) return <main className="fatal"><h1>起動できませんでした</h1><p>{error}</p></main>;
  if (!bootstrap || !dataset || !unitId) return <main className="loading">読み込み中…</main>;
  const run = showPredictions ? (allRuns[unitId] ?? []).find((item) => modelKey(item) === modelId) : undefined;
  const liveStats = unit && run ? computeLiveStats(unit, run) : null;
  return <div className="app-shell">
    {sidebarOpen && <Sidebar bootstrap={bootstrap} dataset={dataset} unitId={unitId} allRuns={allRuns} modelId={showPredictions ? modelId : ""} modelOptions={modelOptions}
      liveMean={liveStats ? liveStats.mean : undefined}
      onDataset={changeDataset} onUnit={setUnitId}
      onModel={(value) => { if (!value) setShowPredictions(false); else { setShowPredictions(true); setModelId(value); } }} />}
    <main className="main">
      <header className="topbar">
        <button className="icon-btn" onClick={() => setSidebarOpen(!sidebarOpen)} aria-label="サイドバーを切り替え" aria-expanded={sidebarOpen}>☰</button>
        <strong className="unit-name">{unitId}</strong>
        {unit && <span className="unit-meta">{unit.duration_s.toFixed(1)}秒　{unit.fps}fps　{unit.n_frames}フレーム{bootstrap.read_only ? "　閲覧専用" : ""}</span>}
        <span className="spacer" />
        {liveStats && <span className="topbar-stats" aria-label="この動画の評価">
          <span>mean tIoU <strong className={liveStats.mean !== null ? tiouClass(liveStats.mean) : ""}>{liveStats.mean?.toFixed(3) ?? "—"}</strong></span>
          <span>F1@0.5 <strong>{liveStats.f1?.toFixed(3) ?? "—"}</strong></span>
          <span>人手 <strong>{liveStats.gtCount}</strong></span>
          <span>モデル <strong>{liveStats.predCount}</strong></span>
        </span>}
        {!bootstrap.read_only && currentSummary && <button className={`exclude-toggle ${currentSummary.excluded ? "on" : ""}`} aria-pressed={!!currentSummary.excluded}
          title={currentSummary.excluded ? "この動画はデータセットから除外されています。クリックで戻します" : "この動画をデータセットから除外する"}
          onClick={() => void toggleExcluded()}>{currentSummary.excluded ? "除外中" : "除外"}</button>}
        {(saveState === "saving" || saveState === "error") && <span className={`save-state ${saveState}`} aria-live="polite">{saveState === "saving" ? "保存中…" : "保存エラー"}</span>}
      </header>
      {error && <div className="error-banner" role="alert">{error}<button onClick={() => setError("")}>閉じる</button></div>}
      {unit ? <EditorWorkspace unit={unit} run={run} readOnly={bootstrap.read_only} onChange={changeUnit} /> : <div className="loading">動画を読み込み中…</div>}
    </main>
  </div>;
}
