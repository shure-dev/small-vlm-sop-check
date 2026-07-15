import { useEffect, useRef, useState } from "react";

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
  model: { name?: string };
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

function Sidebar({ bootstrap, dataset, unitId, workspace, runs, runId, onDataset, onUnit, onWorkspace, onRun }: {
  bootstrap: Bootstrap;
  dataset: string;
  unitId: string;
  workspace: "annotate" | "review";
  runs: ComparisonRun[];
  runId: string;
  onDataset: (value: string) => void;
  onUnit: (value: string) => void;
  onWorkspace: (value: "annotate" | "review") => void;
  onRun: (value: string) => void;
}) {
  const units = bootstrap.units.filter((unit) => unit.dataset === dataset);
  const run = runs.find((item) => item.run_id === runId);
  const summary = run?.comparison.summary;
  return (
    <aside className="sidebar">
      <h1>動画イベント</h1>
      <nav className="workspace-switch" aria-label="機能">
        <button className={workspace === "annotate" ? "active" : ""} onClick={() => onWorkspace("annotate")}>アノテーション</button>
        <button className={workspace === "review" ? "active" : ""} onClick={() => onWorkspace("review")}>結果を見る</button>
      </nav>
      <label className="field-label" htmlFor="dataset">データセット</label>
      <select id="dataset" value={dataset} onChange={(event) => onDataset(event.target.value)}>
        {bootstrap.datasets.map((name) => <option key={name}>{name}</option>)}
      </select>
      {workspace === "review" && <>
        <label className="field-label" htmlFor="run">モデル</label>
        <select id="run" value={runId} onChange={(event) => onRun(event.target.value)} disabled={!runs.length}>
          {runs.length ? runs.map((item) => <option key={item.run_id} value={item.run_id}>{item.model.name || item.run_id}</option>) : <option>結果なし</option>}
        </select>
      </>}
      <label className="field-label" htmlFor="unit">動画</label>
      <select id="unit" value={unitId} onChange={(event) => onUnit(event.target.value)}>
        {units.map((unit) => <option key={unit.unit_id} value={unit.unit_id}>{unit.unit_id}</option>)}
      </select>
      {workspace === "review" && summary && <section className="sidebar-summary" aria-label="評価サマリー">
        <h2>評価サマリー</h2>
        <dl>
          <div><dt>mean tIoU</dt><dd>{summary.mean_tiou?.toFixed(3) ?? "—"}</dd></div>
          <div><dt>tIoU@0.5 F1</dt><dd>{summary.thresholds["tiou@0.5"]?.f1?.toFixed(3) ?? "—"}</dd></div>
          <div><dt>人手区間</dt><dd>{summary.gt_occurrences}</dd></div>
          <div><dt>モデル区間</dt><dd>{summary.predicted_occurrences}</dd></div>
        </dl>
      </section>}
    </aside>
  );
}

function VideoPanel({ unit, current, onCurrent }: { unit: UnitData; current: number; onCurrent: (value: number) => void }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [showFrame, setShowFrame] = useState(false);
  const step = 1 / unit.fps;
  useEffect(() => {
    if (videoRef.current && Math.abs(videoRef.current.currentTime - current) > step / 2) videoRef.current.currentTime = current;
  }, [current, step]);
  const seek = (value: number) => {
    const next = clamp(value, 0, Math.max(0, unit.duration_s - step));
    if (videoRef.current) videoRef.current.currentTime = next;
    onCurrent(next);
  };
  const frame = Math.min(unit.n_frames - 1, Math.max(0, Math.round(current * unit.fps)));
  return (
    <section className="video-panel" aria-label="動画">
      <video ref={videoRef} src={unit.media_url} controls preload="metadata" onTimeUpdate={(event) => onCurrent(event.currentTarget.currentTime)} />
      <div className="video-controls">
        <strong>{time(current)}</strong>
        <button onClick={() => seek(current - step)} disabled={current <= 0}>← 1フレーム</button>
        <button onClick={() => seek(current + step)} disabled={current >= unit.duration_s - step}>1フレーム →</button>
        <button className="frame-toggle" aria-expanded={showFrame} onClick={() => setShowFrame(!showFrame)}>{showFrame ? "静止画を閉じる" : "静止画で確認"}</button>
      </div>
      <input className="seek" aria-label="確認位置" type="range" min="0" max={Math.max(0, unit.duration_s - step)} step={step} value={current} onChange={(event) => seek(Number(event.target.value))} />
      {showFrame && <img className="boundary-frame" src={unit.frame_url_template.replace("{index}", String(frame))} alt={`${time(current)}のフレーム`} />}
    </section>
  );
}

type TimelineRow = { id: string; label: string; human?: Span[] | null; model?: Span[] | null };

function Timeline({ rows, duration, current, onSeek, onSelect, onChange }: {
  rows: TimelineRow[];
  duration: number;
  current: number;
  onSeek?: (value: number) => void;
  onSelect?: (id: string) => void;
  onChange?: (id: string, spans: Span[]) => void;
}) {
  const startDrag = (event: React.PointerEvent, row: TimelineRow, index: number, edge: "start" | "end") => {
    if (!onChange || !row.human) return;
    event.preventDefault(); event.stopPropagation();
    const rect = (event.currentTarget.closest(".timeline-track") as HTMLElement).getBoundingClientRect();
    const original = clone(row.human);
    const move = (pointer: PointerEvent) => {
      const value = clamp((pointer.clientX - rect.left) / rect.width * duration, 0, duration);
      const next = clone(original);
      if (edge === "start") next[index].start_s = Math.min(value, next[index].end_s - .001);
      else next[index].end_s = Math.max(value, next[index].start_s + .001);
      onChange(row.id, next.sort((a, b) => a.start_s - b.start_s));
    };
    const up = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); };
    window.addEventListener("pointermove", move); window.addEventListener("pointerup", up, { once: true });
  };
  if (!rows.length) return <div className="empty-state">イベントを追加してください。</div>;
  return (
    <div className="timeline" style={{ "--cursor": `${current / Math.max(duration, .1) * 100}%` } as React.CSSProperties}>
      {rows.map((row) => <div className="timeline-row" key={row.id}>
        <button className="timeline-label" title={row.label} onClick={() => onSelect?.(row.id)}>{row.label}</button>
        <div className="timeline-track" onClick={(event) => {
          if (!onSeek) return;
          const rect = event.currentTarget.getBoundingClientRect();
          onSeek(clamp((event.clientX - rect.left) / rect.width * duration, 0, duration));
        }}>
          <span className="cursor-line" />
          {(row.human || []).map((span, index) => <span className="timeline-span human" key={`h-${index}`} style={{ left: `${span.start_s / duration * 100}%`, width: `${(span.end_s - span.start_s) / duration * 100}%` }} title={`人手 ${time(span.start_s)}–${time(span.end_s)}`}>
            {onChange && <><button className="handle start" aria-label="開始境界" onPointerDown={(event) => startDrag(event, row, index, "start")} /><button className="handle end" aria-label="終了境界" onPointerDown={(event) => startDrag(event, row, index, "end")} /></>}
          </span>)}
          {(row.model || []).map((span, index) => <span className="timeline-span model" key={`m-${index}`} style={{ left: `${span.start_s / duration * 100}%`, width: `${(span.end_s - span.start_s) / duration * 100}%` }} title={`モデル ${time(span.start_s)}–${time(span.end_s)}`} />)}
        </div>
      </div>)}
      <div className="timeline-axis"><span>0秒</span><span>{time(duration / 2)}</span><span>{time(duration)}</span></div>
    </div>
  );
}

function AnnotationWorkspace({ unit, readOnly, saveState, onChange }: {
  unit: UnitData;
  readOnly: boolean;
  saveState: "idle" | "saving" | "saved" | "error";
  onChange: (next: UnitData) => void;
}) {
  const [current, setCurrent] = useState(0);
  const [activeId, setActiveId] = useState(unit.sop.events[0]?.id || "");
  const [creating, setCreating] = useState(unit.sop.events.length === 0);
  const [draftLabel, setDraftLabel] = useState("");
  const [backup, setBackup] = useState<Record<string, Span[]>>({});
  useEffect(() => { setCurrent(0); setActiveId(unit.sop.events[0]?.id || ""); setCreating(unit.sop.events.length === 0); setDraftLabel(""); }, [unit.unit_id]);
  const active = unit.sop.events.find((event) => event.id === activeId);
  const mutate = (fn: (next: UnitData) => void) => { const next = clone(unit); fn(next); onChange(next); };
  const updateSpans = (eventId: string, spans: Span[]) => mutate((next) => {
    const step = 1 / unit.fps;
    next.annotation.events[eventId] = spans.map((span) => {
      const start_s = clamp(Math.round(span.start_s * unit.fps) / unit.fps, 0, Math.max(0, unit.duration_s - step));
      return { start_s, end_s: clamp(Math.round(span.end_s * unit.fps) / unit.fps, start_s + step, unit.duration_s) };
    }).sort((a, b) => a.start_s - b.start_s);
  });
  const addEvent = () => {
    if (!draftLabel.trim()) return;
    const used = new Set(unit.sop.events.map((event) => event.id));
    let index = 1; while (used.has(`event_${String(index).padStart(3, "0")}`)) index += 1;
    const id = `event_${String(index).padStart(3, "0")}`;
    const start = clamp(current, 0, Math.max(0, unit.duration_s - 1 / unit.fps));
    mutate((next) => {
      next.sop.events.push({ id, ask: draftLabel.trim(), values: ["yes", "no"] });
      next.annotation.event_labels[id] = draftLabel.trim();
      next.annotation.events[id] = [{ start_s: start, end_s: Math.min(unit.duration_s, start + Math.max(1, 1 / unit.fps)) }];
    });
    setActiveId(id); setCreating(false); setDraftLabel("");
  };
  const rows = unit.sop.events.map((event) => ({ id: event.id, label: event.ask, human: unit.annotation.events[event.id] }));
  return (
    <div className="workspace annotation-workspace" tabIndex={0} onKeyDown={(event) => {
      if (["INPUT", "TEXTAREA", "SELECT"].includes((event.target as HTMLElement).tagName)) return;
      if (event.key === "ArrowLeft") setCurrent((value) => clamp(value - 1 / unit.fps, 0, unit.duration_s - 1 / unit.fps));
      if (event.key === "ArrowRight") setCurrent((value) => clamp(value + 1 / unit.fps, 0, unit.duration_s - 1 / unit.fps));
    }}>
      <div className="work-grid">
        <VideoPanel unit={unit} current={current} onCurrent={setCurrent} />
        <section>
          <div className="section-heading"><h2>イベント</h2><span className={`save-state ${saveState}`}>{saveState === "saving" ? "保存中…" : saveState === "error" ? "保存エラー" : "保存済み"}</span></div>
          <div className="event-tabs" role="list" aria-label="イベント一覧">
            {unit.sop.events.map((event, index) => <button key={event.id} className={activeId === event.id && !creating ? "active" : ""} onClick={() => { setActiveId(event.id); setCreating(false); }}><span>{index + 1}</span>{event.ask}</button>)}
            {!readOnly && <button className={creating ? "active new" : "new"} onClick={() => setCreating(true)}>＋ 追加</button>}
          </div>
          <div className="editor-card">
            {creating ? <>
              <label className="field-label" htmlFor="new-label">日本語イベント文</label>
              <textarea id="new-label" rows={3} value={draftLabel} onChange={(event) => setDraftLabel(event.target.value)} placeholder="映像で見える動作を具体的に書く" autoFocus />
              <button className="primary wide-button" disabled={!draftLabel.trim()} onClick={addEvent}>現在位置から作成</button>
            </> : active ? <EventEditor unit={unit} event={active} current={current} readOnly={readOnly} backup={backup} setBackup={setBackup} mutate={mutate} onDelete={() => {
              const remaining = unit.sop.events.filter((event) => event.id !== active.id);
              mutate((next) => { next.sop.events = remaining; delete next.annotation.events[active.id]; delete next.annotation.event_labels[active.id]; });
              setActiveId(remaining[0]?.id || ""); setCreating(remaining.length === 0);
            }} /> : <div className="empty-state">イベントを追加してください。</div>}
          </div>
        </section>
      </div>
      <section className="overview"><h2>タイムライン</h2><Timeline rows={rows} duration={unit.duration_s} current={current} onSeek={setCurrent} onSelect={(id) => { setActiveId(id); setCreating(false); }} onChange={readOnly ? undefined : updateSpans} /></section>
    </div>
  );
}

function EventEditor({ unit, event, current, readOnly, backup, setBackup, mutate, onDelete }: {
  unit: UnitData; event: SopEvent; current: number; readOnly: boolean; backup: Record<string, Span[]>;
  setBackup: React.Dispatch<React.SetStateAction<Record<string, Span[]>>>;
  mutate: (fn: (next: UnitData) => void) => void; onDelete: () => void;
}) {
  const spans = unit.annotation.events[event.id];
  const step = 1 / unit.fps;
  const update = (value: Span[] | null) => mutate((next) => { next.annotation.events[event.id] = value; });
  return <>
    <label className="field-label" htmlFor={`label-${event.id}`}>日本語イベント文</label>
    <textarea id={`label-${event.id}`} rows={3} value={event.ask} readOnly={readOnly} onChange={(change) => mutate((next) => {
      const target = next.sop.events.find((item) => item.id === event.id)!; target.ask = change.target.value; next.annotation.event_labels[event.id] = change.target.value;
    })} />
    <label className="absent"><input type="checkbox" checked={spans === null} disabled={readOnly} onChange={(change) => {
      if (change.target.checked) { if (spans) setBackup((value) => ({ ...value, [event.id]: spans })); update(null); }
      else { const start = clamp(current, 0, Math.max(0, unit.duration_s - step)); update(backup[event.id] || [{ start_s: start, end_s: Math.min(unit.duration_s, start + Math.max(1, step)) }]); }
    }} />このイベントは動画内で起きていない</label>
    {spans !== null && <div className="span-list">
      {(spans || []).map((span, index) => <div className="span-editor" key={index}>
        <div className="span-title"><strong>区間 {index + 1}</strong><span>{time(span.start_s)} – {time(span.end_s)}</span></div>
        <div className="span-inputs">
          <label>開始<input type="number" min="0" max={span.end_s - step} step={step} value={span.start_s} readOnly={readOnly} onChange={(change) => { const next = clone(spans || []); next[index].start_s = Number(change.target.value); update(next); }} /></label>
          <button disabled={readOnly} onClick={() => { const next = clone(spans || []); next[index].start_s = Math.min(current, next[index].end_s - step); update(next); }}>現在位置</button>
          <label>終了<input type="number" min={span.start_s + step} max={unit.duration_s} step={step} value={span.end_s} readOnly={readOnly} onChange={(change) => { const next = clone(spans || []); next[index].end_s = Number(change.target.value); update(next); }} /></label>
          <button disabled={readOnly} onClick={() => { const next = clone(spans || []); next[index].end_s = Math.max(current, next[index].start_s + step); update(next); }}>現在位置</button>
          <button className="danger-text" disabled={readOnly || spans?.length === 1} onClick={() => update((spans || []).filter((_, item) => item !== index))}>削除</button>
        </div>
      </div>)}
      {!readOnly && <button className="wide-button" onClick={() => { const start = clamp(current, 0, Math.max(0, unit.duration_s - step)); update([...(spans || []), { start_s: start, end_s: Math.min(unit.duration_s, start + Math.max(1, step)) }].sort((a, b) => a.start_s - b.start_s)); }}>＋ 区間を追加</button>}
    </div>}
    {!readOnly && <details className="danger-zone"><summary>イベントを削除</summary><button className="danger" onClick={() => window.confirm("このイベントを削除しますか？") && onDelete()}>削除する</button></details>}
  </>;
}

function ReviewWorkspace({ unit, run }: { unit: UnitData; run?: ComparisonRun }) {
  const [current, setCurrent] = useState(0);
  useEffect(() => setCurrent(0), [unit.unit_id]);
  if (!run) return <div className="empty-state large">比較可能な推論結果がありません。</div>;
  const rows = run.comparison.events.map((event) => ({ id: event.event_id, label: event.text, human: event.reference_spans, model: event.prediction_spans }));
  const sorted = [...run.comparison.events].sort((a, b) => (a.tiou ?? -1) - (b.tiou ?? -1));
  return <div className="workspace review-workspace">
    <div className="review-grid">
      <VideoPanel unit={unit} current={current} onCurrent={setCurrent} />
      <section className="comparison-panel">
        <div className="comparison-heading"><h2>人手区間とモデル区間</h2><div className="inline-legend"><span><i className="legend-swatch human" />人手</span><span><i className="legend-swatch model" />モデル</span></div></div>
        <Timeline rows={rows} duration={unit.duration_s} current={current} onSeek={setCurrent} />
      </section>
    </div>
    <section className="result-section"><h2>イベント別結果</h2><p>tIoUの低い順。動画と区間を見比べてイベント定義を改善します。</p>
      <div className="result-table-wrap"><table className="result-table"><thead><tr><th>判定</th><th>イベント</th><th>人手区間</th><th>モデル区間</th><th>tIoU</th></tr></thead><tbody>
        {sorted.map((event) => <tr key={event.event_id}><td><span className={`verdict ${(event.tiou ?? 0) < .5 ? "bad" : (event.tiou ?? 0) < .6 ? "check" : "good"}`}>{(event.tiou ?? 0) < .5 ? "要改善" : (event.tiou ?? 0) < .6 ? "境界確認" : "良好"}</span></td><td>{event.text}</td><td>{formatSpans(event.reference_spans)}</td><td>{formatSpans(event.prediction_spans)}</td><td className="score">{event.tiou?.toFixed(3) ?? "—"}</td></tr>)}
      </tbody></table></div>
    </section>
  </div>;
}

function formatSpans(spans: Span[] | null) {
  if (spans === null) return "該当なし";
  if (!spans?.length) return "—";
  return spans.map((span) => `${Number(span.start_s.toFixed(2))}–${Number(span.end_s.toFixed(2))}秒`).join(", ");
}

export default function App() {
  const [bootstrap, setBootstrap] = useState<Bootstrap | null>(null);
  const [dataset, setDataset] = useState("");
  const [unitId, setUnitId] = useState("");
  const [unit, setUnit] = useState<UnitData | null>(null);
  const [workspace, setWorkspace] = useState<"annotate" | "review">("annotate");
  const [runs, setRuns] = useState<ComparisonRun[]>([]);
  const [runId, setRunId] = useState("");
  const [saveState, setSaveState] = useState<"idle" | "saving" | "saved" | "error">("idle");
  const [error, setError] = useState("");
  const saveTimer = useRef<number | undefined>(undefined);
  useEffect(() => { getJson<Bootstrap>("/api/bootstrap").then((value) => {
    setBootstrap(value);
    const initialDataset = value.initial_dataset && value.datasets.includes(value.initial_dataset) ? value.initial_dataset : value.datasets[0] || "";
    const units = value.units.filter((item) => item.dataset === initialDataset);
    setDataset(initialDataset); setUnitId(value.initial_unit && units.some((item) => item.unit_id === value.initial_unit) ? value.initial_unit : units[0]?.unit_id || "");
  }).catch((reason) => setError(String(reason))); }, []);
  useEffect(() => {
    if (!dataset || !unitId) return;
    setUnit(null);
    getJson<UnitData>(`/api/units/${dataset}/${unitId}`).then((value) => { setUnit(value); setSaveState("idle"); setError(""); }).catch((reason) => setError(String(reason)));
    getJson<{ runs: ComparisonRun[] }>(`/api/comparisons/${dataset}/${unitId}`).then((value) => { setRuns(value.runs); setRunId((current) => value.runs.some((item) => item.run_id === current) ? current : value.runs[0]?.run_id || ""); }).catch(() => { setRuns([]); setRunId(""); });
  }, [dataset, unitId]);
  const persist = async (next: UnitData) => {
    setSaveState("saving");
    try { await getJson(`/api/units/${next.dataset}/${next.unit_id}`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ sop: next.sop, annotation: next.annotation }) }); setSaveState("saved"); setError(""); }
    catch (reason) { setSaveState("error"); setError(reason instanceof Error ? reason.message : String(reason)); }
  };
  const changeUnit = (next: UnitData) => { setUnit(next); setSaveState("saving"); window.clearTimeout(saveTimer.current); saveTimer.current = window.setTimeout(() => void persist(next), 450); };
  const changeDataset = (value: string) => { setDataset(value); setUnitId(bootstrap?.units.find((item) => item.dataset === value)?.unit_id || ""); };
  if (error && !bootstrap) return <main className="fatal"><h1>起動できませんでした</h1><p>{error}</p></main>;
  if (!bootstrap || !dataset || !unitId) return <main className="loading">読み込み中…</main>;
  const run = runs.find((item) => item.run_id === runId);
  return <div className="app-shell">
    <Sidebar bootstrap={bootstrap} dataset={dataset} unitId={unitId} workspace={workspace} runs={runs} runId={runId} onDataset={changeDataset} onUnit={setUnitId} onWorkspace={setWorkspace} onRun={setRunId} />
    <main className="main">
      {error && <div className="error-banner" role="alert">{error}<button onClick={() => setError("")}>閉じる</button></div>}
      {unit ? workspace === "annotate" ? <AnnotationWorkspace unit={unit} readOnly={bootstrap.read_only} saveState={saveState} onChange={changeUnit} /> : <ReviewWorkspace unit={unit} run={run} /> : <div className="loading">動画を読み込み中…</div>}
    </main>
  </div>;
}
