"use client";

import { useEffect, useState } from "react";
import { API } from "@/lib/api";

type Camera = {
  camera_id: string; camera_name: string; state: string; connection_status: string;
  violence_score: number | null; last_prediction_time: string | null; error: string | null;
  processed_fps: number; inference_latency_ms: number | null; dropped_windows: number;
  violence_event: { event_start: string; highest_score: number; status: string } | null;
};

const colors: Record<string, string> = {
  NORMAL: "bg-emerald-100 text-emerald-800", VIOLENCE: "bg-red-100 text-red-800",
  OFFLINE: "bg-zinc-200 text-zinc-700", CONNECTING: "bg-amber-100 text-amber-800",
  ERROR: "bg-orange-100 text-orange-800",
  BUFFERING: "bg-blue-100 text-blue-800",
};

function CameraPreview({ camera }: { camera: Camera }) {
  const [attempt, setAttempt] = useState(0);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    if (!failed) return;
    const timer = setTimeout(() => { setFailed(false); setAttempt(n => n + 1); }, 3000);
    return () => clearTimeout(timer);
  }, [failed]);
  return <div className="relative mt-4 aspect-video overflow-hidden rounded-md bg-black text-white">
    {/* Native img is intentional: this is a continuous multipart MJPEG response. */}
    {/* eslint-disable-next-line @next/next/no-img-element */}
    <img
      key={attempt}
      src={`${API}/api/v1/cameras/${encodeURIComponent(camera.camera_id)}/preview?attempt=${attempt}`}
      alt={`Live view of ${camera.camera_name}`}
      onError={() => setFailed(true)}
      className="h-full w-full object-contain"
    />
    {(failed || camera.connection_status !== "ONLINE") && <div className="absolute inset-0 flex items-center justify-center bg-black/80 text-sm">
      {failed ? "Preview unavailable — retrying…" : `${camera.connection_status} — waiting for camera`}
    </div>}
    <span className="absolute bottom-2 left-2 rounded bg-black/70 px-2 py-1 text-xs">Live sampled camera feed</span>
  </div>;
}

export function CameraDashboard() {
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [id, setId] = useState("");
  const [url, setUrl] = useState("");
  const [error, setError] = useState("");
  const [connected, setConnected] = useState(false);
  useEffect(() => {
    let disposed = false;
    let socket: WebSocket | undefined;
    let retry: ReturnType<typeof setTimeout> | undefined;
    let delay = 1000;
    const api = new URL(`${API}/api/v1/cameras/ws`, window.location.href);
    api.protocol = api.protocol === "https:" ? "wss:" : "ws:";
    function connect() {
      if (disposed) return;
      const ws = new WebSocket(api);
      socket = ws;
      ws.onopen = () => {
        if (disposed) return;
        delay = 1000;
        setConnected(true);
      };
      ws.onmessage = (event) => {
        if (disposed) return;
        try {
          const message = JSON.parse(event.data);
          if (Array.isArray(message.cameras)) setCameras(message.cameras);
        } catch { /* Ignore malformed updates. */ }
      };
      ws.onclose = () => {
        if (disposed) return;
        setConnected(false);
        retry = setTimeout(connect, delay);
        delay = Math.min(delay * 2, 10000);
      };
      ws.onerror = () => ws.close();
    }
    connect();
    return () => {
      disposed = true;
      clearTimeout(retry);
      if (socket) {
        socket.onopen = socket.onmessage = socket.onerror = socket.onclose = null;
        socket.close();
      }
    };
  }, []);

  async function add(event: React.FormEvent) {
    event.preventDefault(); setError("");
    try {
      const response = await fetch(`${API}/api/v1/cameras`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ camera_id: id, name: id, url }),
      });
      if (!response.ok) throw new Error(await response.text());
      setId(""); setUrl("");
    } catch (cause) { setError(cause instanceof Error ? cause.message : "Could not add camera."); }
  }

  return <main className="mx-auto max-w-6xl space-y-7 px-5 py-10 sm:px-8">
    <header><p className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">Continuous monitoring</p>
      <h1 className="mt-2 text-3xl font-semibold">Camera dashboard</h1>
      <p className="mt-2 text-sm text-muted-foreground">Scores are uncalibrated model outputs, not probabilities of harm.</p></header>
    <form onSubmit={add} className="grid gap-3 rounded-lg border bg-card p-4 sm:grid-cols-[1fr_2fr_auto]">
      <input aria-label="Camera ID" required value={id} onChange={e => setId(e.target.value)} placeholder="Camera ID" className="rounded-md border bg-background px-3 py-2 text-sm" />
      <input aria-label="Stream URL" required value={url} onChange={e => setUrl(e.target.value)} placeholder="rtsp://… or http://…" className="rounded-md border bg-background px-3 py-2 text-sm" />
      <button className="rounded-md bg-primary px-4 py-2 text-sm text-primary-foreground">Add camera</button>
    </form>
    <p role="status" className="text-sm">{connected ? "Live updates connected" : "Live updates disconnected — reconnecting. Camera statuses below may be stale."}</p>
    {error && <p role="alert" className="text-sm text-red-700">{error}</p>}
    {!cameras.length && <p className="rounded-lg border p-8 text-center text-sm text-muted-foreground">No cameras configured. Add a stream above or configure CAMERAS_JSON before starting the API.</p>}
    <section aria-label="Camera statuses" className="grid gap-4 md:grid-cols-2">
      {cameras.map(camera => <article key={camera.camera_id} className={`rounded-lg border bg-card p-5 ${camera.state === "VIOLENCE" ? "border-red-500 ring-1 ring-red-500" : ""}`}>
        <div className="flex items-start justify-between gap-3"><div><h2 className="font-semibold">{camera.camera_name}</h2><p className="font-mono text-xs text-muted-foreground">{camera.camera_id}</p></div>
          <span className={`rounded-full px-3 py-1 text-xs font-bold ${colors[camera.state] ?? colors.CONNECTING}`}>{camera.state}</span></div>
        <CameraPreview camera={camera} />
        {camera.state === "BUFFERING" && <p className="mt-2 text-xs text-muted-foreground">Camera connected · collecting a temporal window for analysis…</p>}
        {camera.state === "VIOLENCE" && <p className="mt-4 text-lg font-bold text-red-700">VIOLENCE DETECTED</p>}
        <dl className="mt-4 grid grid-cols-2 gap-3 text-sm"><div><dt className="text-muted-foreground">Violence score</dt><dd className="font-mono">{camera.violence_score?.toFixed(3) ?? "—"}</dd></div>
          <div><dt className="text-muted-foreground">Connection</dt><dd>{camera.connection_status}</dd></div>
          <div><dt className="text-muted-foreground">Inference latency</dt><dd>{camera.inference_latency_ms == null ? "—" : `${camera.inference_latency_ms} ms`}</dd></div>
          <div><dt className="text-muted-foreground">Dropped windows</dt><dd>{camera.dropped_windows}</dd></div></dl>
        {camera.violence_event && <p className="mt-3 text-xs text-red-700">Event since {new Date(camera.violence_event.event_start).toLocaleString()} · peak score {camera.violence_event.highest_score.toFixed(3)}</p>}
        {camera.error && <p className="mt-2 text-xs text-orange-700">{camera.error}</p>}
        <button onClick={() => fetch(`${API}/api/v1/cameras/${encodeURIComponent(camera.camera_id)}`, { method: "DELETE" })} className="mt-4 text-xs underline">Remove camera</button>
      </article>)}
    </section>
  </main>;
}
