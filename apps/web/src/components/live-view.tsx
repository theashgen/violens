"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Camera, CameraOff, Radio, ScanEye, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { API, parsePrediction, readErrorDetail, type Prediction } from "@/lib/api";
import { cn } from "@/lib/utils";

type Health = "checking" | "ready" | "unavailable" | "offline";
type LivePhase = "disconnected" | "connecting" | "streaming";
type Source = "webcam" | "rtsp";

type Status = {
  source: Source;
  buffered_frames: number;
  frames_needed: number;
  dropped_frames: number;
  latest_prediction: Prediction | null;
  analyzing: boolean;
  last_analysis_error: string | null;
  connected: boolean | null;
  last_error: string | null;
  seconds_since_frame: number | null;
  interval_seconds: number;
};

const FRAME_UPLOAD_HZ = 5;
const SNAPSHOT_INTERVAL_MS = 1000;
const ANALYSIS_INTERVAL_MS = 5000;
const STATUS_INTERVAL_MS = 2000;
const CAPTURE_WIDTH = 640;
const CAPTURE_HEIGHT = 480;
const RTSP_PATTERN = /^(rtsp|rtmp|http|https|udp):\/\/\S+$/i;

function isStatus(value: unknown): value is Status {
  if (typeof value !== "object" || value === null) return false;
  const body = value as Record<string, unknown>;
  return (
    (body.source === "webcam" || body.source === "rtsp") &&
    typeof body.buffered_frames === "number" &&
    typeof body.frames_needed === "number" &&
    typeof body.dropped_frames === "number" &&
    typeof body.analyzing === "boolean"
  );
}

export function LiveView() {
  const [phase, setPhase] = useState<LivePhase>("disconnected");
  const [source, setSource] = useState<Source | null>(null);
  const [status, setStatus] = useState<Status | null>(null);
  const [error, setError] = useState("");
  const [cameraUrl, setCameraUrl] = useState("");
  const [health, setHealth] = useState<Health>("checking");
  const [mode, setMode] = useState<"choose" | "webcam" | "rtsp">("choose");

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);
  const pushTimer = useRef<number | null>(null);
  const snapshotTimer = useRef<number | null>(null);
  const analysisTimer = useRef<number | null>(null);
  const statusTimer = useRef<number | null>(null);
  const inFlight = useRef(false);
  const sourceRef = useRef<Source | null>(null);
  const healthRef = useRef<Health>("checking");

  useEffect(() => {
    healthRef.current = health;
  }, [health]);

  const stopLoops = useCallback(() => {
    for (const timer of [pushTimer, snapshotTimer, analysisTimer, statusTimer]) {
      if (timer.current !== null) {
        window.clearInterval(timer.current);
        timer.current = null;
      }
    }
    cancelAnimationFrame(rafRef.current);
  }, []);

  // Reads refs only, so a stable identity is unnecessary; hoisted so the
  // self-scheduling requestAnimationFrame call is legal.
  function drawLoop() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (video && canvas && video.readyState >= 2 && video.videoWidth > 0) {
      const context = canvas.getContext("2d");
      if (context) {
        if (canvas.width !== video.videoWidth) canvas.width = video.videoWidth;
        if (canvas.height !== video.videoHeight) canvas.height = video.videoHeight;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
      }
    }
    rafRef.current = requestAnimationFrame(drawLoop);
  }

  const pushFrames = useCallback(async () => {
    if (sourceRef.current !== "webcam" || inFlight.current) return;
    const video = videoRef.current;
    if (!video || video.readyState < 2 || !video.videoWidth) return;
    inFlight.current = true;
    try {
      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const context = canvas.getContext("2d");
      if (!context) return;
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      const blob = await new Promise<Blob | null>((resolve) =>
        canvas.toBlob(resolve, "image/jpeg", 0.7),
      );
      if (!blob) return;
      const form = new FormData();
      form.append("frames", blob, "frame.jpg");
      await fetch(`${API}/api/v1/stream/webcam/frames`, {
        method: "POST",
        body: form,
      });
    } catch {
      // Status polling surfaces connection problems; uploads retry silently.
    } finally {
      inFlight.current = false;
    }
  }, []);

  const runAnalysis = useCallback(async () => {
    if (!sourceRef.current || healthRef.current !== "ready") return;
    try {
      const response = await fetch(
        `${API}/api/v1/stream/${sourceRef.current}/analyze`,
        { method: "POST" },
      );
      if (response.ok) {
        parsePrediction(await response.json());
      }
      // Failures are surfaced through status polling, not popups.
    } catch {
      // Network errors also surface through status polling.
    }
  }, []);

  const refreshStatus = useCallback(() => {
    if (!sourceRef.current) return;
    fetch(`${API}/api/v1/stream/${sourceRef.current}/status`, {
      cache: "no-store",
    })
      .then(async (response) => {
        if (!response.ok) throw new Error(String(response.status));
        const body: unknown = await response.json();
        if (!isStatus(body)) throw new Error("Invalid status");
        setStatus(body);
        setError("");
      })
      .catch(() => {
        if (sourceRef.current) {
          setError("Lost contact with the streaming API.");
        }
      });
  }, []);

  const refreshSnapshot = useCallback(() => {
    if (sourceRef.current !== "rtsp") return;
    fetch(`${API}/api/v1/stream/rtsp/snapshot`, {
      cache: "no-store",
    })
      .then((response) => (response.ok ? response.blob() : null))
      .then((blob) => {
        const canvas = canvasRef.current;
        if (!blob || !canvas) return;
        const image = new Image();
        image.onload = () => {
          const context = canvas.getContext("2d");
          if (!context) return;
          canvas.width = image.naturalWidth;
          canvas.height = image.naturalHeight;
          context.drawImage(image, 0, 0);
          URL.revokeObjectURL(image.src);
        };
        image.src = URL.createObjectURL(blob);
      })
      .catch(() => {});
  }, []);

  const startLoops = useCallback(
    (active: Source) => {
      stopLoops();
      pushTimer.current =
        active === "webcam"
          ? window.setInterval(pushFrames, Math.round(1000 / FRAME_UPLOAD_HZ))
          : null;
      snapshotTimer.current = window.setInterval(
        refreshSnapshot,
        SNAPSHOT_INTERVAL_MS,
      );
      analysisTimer.current = window.setInterval(
        runAnalysis,
        ANALYSIS_INTERVAL_MS,
      );
      statusTimer.current = window.setInterval(refreshStatus, STATUS_INTERVAL_MS);
      rafRef.current = requestAnimationFrame(drawLoop);
      refreshStatus();
      runAnalysis();
    },
    // drawLoop reads refs only and never needs re-subscription.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [stopLoops, pushFrames, refreshSnapshot, runAnalysis, refreshStatus],
  );

  const disconnect = useCallback(() => {
    stopLoops();
    const active = sourceRef.current;
    if (active) {
      fetch(`${API}/api/v1/stream/${active}/stop`, { method: "POST" }).catch(() => {});
    }
    sourceRef.current = null;
    const stream = videoRef.current?.srcObject as MediaStream | null;
    if (stream) {
      for (const track of stream.getTracks()) track.stop();
    }
    if (videoRef.current) videoRef.current.srcObject = null;
    setStatus(null);
    setPhase("disconnected");
    setSource(null);
    setMode("choose");
  }, [stopLoops]);

  useEffect(() => {
    const controller = new AbortController();
    fetch(`${API}/health`, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error("Health check failed");
        const body: unknown = await response.json();
        if (
          typeof body !== "object" ||
          body === null ||
          !("model_ready" in body) ||
          typeof body.model_ready !== "boolean"
        ) {
          throw new Error("Invalid health response");
        }
        setHealth(body.model_ready ? "ready" : "unavailable");
      })
      .catch(() => setHealth("offline"));
    return () => {
      controller.abort();
      disconnect();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function connectWebcam() {
    if (phase !== "disconnected" || health !== "ready") return;
    setError("");
    setPhase("connecting");
    setMode("webcam");
    try {
      const media = await navigator.mediaDevices.getUserMedia({
        video: { width: CAPTURE_WIDTH, height: CAPTURE_HEIGHT },
        audio: false,
      });
      const video = videoRef.current;
      if (!video) throw new Error("Video element unavailable.");
      video.srcObject = media;
      await video.play();
      const response = await fetch(`${API}/api/v1/stream/webcam/connect`, {
        method: "POST",
      });
      if (!response.ok) {
        for (const track of media.getTracks()) track.stop();
        throw new Error(await readErrorDetail(response, "The API rejected the webcam session."));
      }
      sourceRef.current = "webcam";
      setSource("webcam");
      setPhase("streaming");
      startLoops("webcam");
    } catch (cause) {
      disconnect();
      setError(
        cause instanceof Error
          ? cause.name === "NotAllowedError"
            ? "Camera permission was denied. Allow camera access and try again."
            : cause.message
          : "Could not start the webcam.",
      );
    }
  }

  async function connectRtsp(event: React.FormEvent) {
    event.preventDefault();
    if (phase !== "disconnected" || health !== "ready") return;
    if (!RTSP_PATTERN.test(cameraUrl.trim())) {
      setError("Enter an rtsp://, rtmp://, http(s)://, or udp:// camera URL.");
      return;
    }
    setError("");
    setPhase("connecting");
    setMode("rtsp");
    try {
      const response = await fetch(`${API}/api/v1/stream/rtsp/connect`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: cameraUrl.trim() }),
      });
      if (!response.ok) {
        throw new Error(await readErrorDetail(response, "The API rejected the camera URL."));
      }
      sourceRef.current = "rtsp";
      setSource("rtsp");
      setPhase("streaming");
      startLoops("rtsp");
    } catch (cause) {
      disconnect();
      setError(
        cause instanceof Error ? cause.message : "Could not open the camera stream.",
      );
    }
  }

  const buffering = status !== null && status.buffered_frames < status.frames_needed;
  const prediction = status?.latest_prediction ?? null;
  const camConnected =
    status?.connected === null || status?.connected === undefined
      ? source === "webcam"
      : status.connected;

  return (
    <div className="grid items-start gap-5 lg:grid-cols-3">
      <section
        aria-labelledby="live-title"
        className="overflow-hidden rounded-lg border bg-card lg:col-span-2"
      >
        <div className="flex items-center justify-between border-b px-5 py-4">
          <h2 id="live-title" className="text-sm font-semibold">
            Live camera
          </h2>
          <span className="font-mono text-xs text-muted-foreground">
            {source ? `${source.toUpperCase()} / LIVE` : "01 / INPUT"}
          </span>
        </div>
        <div className="p-4 sm:p-5">
          <div
            className={cn(
              "relative flex aspect-video items-center justify-center overflow-hidden rounded-md border",
              phase === "streaming"
                ? "border-border bg-neutral-950"
                : "border-dashed bg-muted/40",
            )}
          >
            <video
              ref={videoRef}
              muted
              playsInline
              className="hidden"
              aria-hidden="true"
            />
            {/* Absolutely positioned so it never competes with the empty-state
                overlay for flex space; visible only while streaming. */}
            <canvas
              ref={canvasRef}
              aria-label="Live camera preview"
              className={cn(
                "absolute inset-0 h-full w-full object-contain",
                phase !== "streaming" && "invisible",
              )}
            />
            {phase !== "streaming" && (
              <div className="z-10 max-w-sm p-5 text-center">
                <div className="mx-auto mb-4 flex size-12 items-center justify-center rounded-lg border bg-card">
                  <ScanEye aria-hidden="true" className="size-5 text-muted-foreground" />
                </div>
                <p className="text-sm font-medium">
                  {phase === "connecting" ? "Connecting…" : "No camera connected"}
                </p>
                <p className="mt-1 text-xs leading-5 text-muted-foreground">
                  {phase === "connecting"
                    ? "Opening the stream and buffering frames."
                    : "Connect your webcam or an RTSP/IP camera to begin."}
                </p>
              </div>
            )}
            {phase === "streaming" && (
              <div className="absolute left-3 top-3 flex items-center gap-2 rounded-md bg-card/90 px-2.5 py-1.5 font-mono text-xs">
                <span
                  aria-hidden="true"
                  className={cn(
                    "size-1.5 rounded-full",
                    buffering ? "bg-yellow-500" : "bg-primary",
                  )}
                />
                {buffering
                  ? `Buffering ${status?.buffered_frames ?? 0}/${status?.frames_needed ?? 16}`
                  : "Analyzing"}
              </div>
            )}
          </div>

          {mode === "choose" && phase === "disconnected" && (
            <div className="mt-4 grid gap-4 sm:grid-cols-2">
              <div className="rounded-md border p-4">
                <div className="flex items-center gap-2">
                  <Camera aria-hidden="true" className="size-4 text-muted-foreground" />
                  <p className="text-sm font-medium">This device</p>
                </div>
                <p className="mt-1.5 text-xs leading-5 text-muted-foreground">
                  Uses your browser camera. Frames stay on your machine except
                  the sampled keyframes sent for analysis.
                </p>
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  className="mt-3"
                  disabled={health !== "ready"}
                  onClick={connectWebcam}
                >
                  Connect webcam
                </Button>
              </div>
              <form className="rounded-md border p-4" onSubmit={connectRtsp}>
                <div className="flex items-center gap-2">
                  <Radio aria-hidden="true" className="size-4 text-muted-foreground" />
                  <p className="text-sm font-medium">IP / CCTV camera</p>
                </div>
                <label
                  htmlFor="camera-url"
                  className="mt-1.5 block text-xs text-muted-foreground"
                >
                  Stream URL (rtsp://, rtmp://, http(s)://, udp://)
                </label>
                <input
                  id="camera-url"
                  type="text"
                  value={cameraUrl}
                  onChange={(event) => setCameraUrl(event.target.value)}
                  placeholder="rtsp://user:pass@camera-ip:554/stream"
                  autoComplete="off"
                  spellCheck={false}
                  className="mt-2 h-9 w-full rounded-md border bg-transparent px-3 font-mono text-xs focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                />
                <Button
                  type="submit"
                  size="sm"
                  variant="outline"
                  className="mt-3"
                  disabled={health !== "ready"}
                >
                  Connect camera
                </Button>
              </form>
            </div>
          )}

          {error && (
            <div
              role="alert"
              className="mt-4 rounded-md border border-destructive/40 bg-destructive/5 p-3 text-sm leading-5 text-destructive"
            >
              {error}
            </div>
          )}
        </div>
        <div className="flex flex-wrap items-center justify-between gap-3 border-t bg-muted/30 px-5 py-4">
          <p className="text-xs text-muted-foreground">
            {phase === "streaming"
              ? "Frames are analyzed continuously in the background."
              : "Model predictions appear while the stream is connected."}
          </p>
          {phase !== "disconnected" && (
            <Button
              type="button"
              size="sm"
              variant="ghost"
              onClick={disconnect}
            >
              <X aria-hidden="true" />
              Disconnect
            </Button>
          )}
        </div>
      </section>

      <section
        aria-labelledby="live-status-title"
        className="rounded-lg border bg-card"
      >
        <div className="flex items-center justify-between border-b px-5 py-4">
          <h2 id="live-status-title" className="text-sm font-semibold">
            Live status
          </h2>
          <span className="font-mono text-xs text-muted-foreground">
            02 / OUTPUT
          </span>
        </div>
        <div className="p-5">
          <div
            role="status"
            aria-live="polite"
            className="flex items-center gap-2 text-xs text-muted-foreground"
          >
            <span
              aria-hidden="true"
              className={cn(
                "size-1.5 rounded-full",
                health === "ready" ? "bg-primary" : "bg-muted-foreground",
              )}
            />
            {health === "checking"
              ? "Checking model"
              : health === "ready"
                ? "Model ready"
                : health === "unavailable"
                  ? "Model not loaded"
                  : "API unreachable"}
          </div>
          <div aria-live="polite" className="min-h-48 py-7">
            {prediction ? (
              <>
                <p className="text-xs text-muted-foreground">Current reading</p>
                <p
                  className={cn(
                    "mt-2 text-3xl font-semibold tracking-tight",
                    prediction.prediction === "violence"
                      ? "text-destructive"
                      : "text-primary",
                  )}
                >
                  {prediction.prediction === "violence" ? "Violence" : "Non-Violence"}
                </p>
                <div className="mt-6 flex items-end justify-between">
                  <span className="text-xs text-muted-foreground">
                    Model confidence
                  </span>
                  <span className="font-mono text-xl">
                    {(prediction.confidence * 100).toFixed(1)}
                    <span className="ml-0.5 text-sm">%</span>
                  </span>
                </div>
                <meter
                  min={0}
                  max={1}
                  value={prediction.confidence}
                  className="mt-2 h-3 w-full accent-primary"
                >
                  {(prediction.confidence * 100).toFixed(1)}%
                </meter>
              </>
            ) : (
              <>
                <p className="text-lg font-medium">
                  {phase === "streaming" ? "Watching the stream…" : "Not connected"}
                </p>
                <p className="mt-3 text-sm leading-6 text-muted-foreground">
                  {phase === "streaming"
                    ? "The model classifies buffered frames every few seconds. The latest reading appears here."
                    : "Connect a camera to see continuous violence status."}
                </p>
              </>
            )}
          </div>
          <dl className="space-y-3 border-y py-4 text-xs">
            <div className="flex justify-between gap-3">
              <dt className="text-muted-foreground">Buffer</dt>
              <dd className="font-mono">
                {status ? `${status.buffered_frames}/${status.frames_needed} frames` : "—"}
              </dd>
            </div>
            <div className="flex justify-between gap-3">
              <dt className="text-muted-foreground">Analysis cadence</dt>
              <dd className="font-mono">
                {status ? `${status.interval_seconds.toFixed(0)} sec` : "—"}
              </dd>
            </div>
            <div className="flex justify-between gap-3">
              <dt className="text-muted-foreground">Dropped frames</dt>
              <dd className="font-mono">{status?.dropped_frames ?? "—"}</dd>
            </div>
            <div className="flex justify-between gap-3">
              <dt className="text-muted-foreground">
                {source === "rtsp" ? "Camera link" : "Capture"}
              </dt>
              <dd>
                {phase !== "streaming"
                  ? "—"
                  : camConnected
                    ? "Connected"
                    : "Reconnecting"}
              </dd>
            </div>
          </dl>
          {status?.last_error && (
            <p role="alert" className="mt-4 text-xs leading-5 text-destructive">
              {status.last_error}
            </p>
          )}
          {status?.last_analysis_error && (
            <p
              role="alert"
              className="mt-2 text-xs leading-5 text-muted-foreground"
            >
              Last analysis: {status.last_analysis_error}
            </p>
          )}
          <p className="mt-4 text-xs leading-5 text-muted-foreground">
            <CameraOff aria-hidden="true" className="mr-1 inline size-3" />
            Model predictions can be incorrect. Confidence is a model score, not
            a guarantee. Human review remains essential.
          </p>
        </div>
      </section>
    </div>
  );
}
