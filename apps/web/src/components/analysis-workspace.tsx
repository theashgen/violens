"use client";

import { useEffect, useRef, useState } from "react";
import { FileVideo, Upload, X } from "lucide-react";
import {
  AnalysisResult,
  type Prediction,
  type Phase,
  type Health,
} from "@/components/analysis-result";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

const API = (
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
).replace(/\/$/, "");
const MAX_BYTES = 100 * 1024 * 1024;
const FORMATS = /\.(mp4|avi|mov|mkv|webm)$/i;

function parsePrediction(value: unknown): Prediction {
  if (
    typeof value !== "object" ||
    value === null ||
    !("prediction" in value) ||
    !("confidence" in value) ||
    (value.prediction !== "violence" && value.prediction !== "non-violence") ||
    typeof value.confidence !== "number" ||
    !Number.isFinite(value.confidence) ||
    value.confidence < 0 ||
    value.confidence > 1
  ) {
    throw new Error(
      "The API returned an invalid prediction. Check the backend version.",
    );
  }
  return { prediction: value.prediction, confidence: value.confidence };
}

export function AnalysisWorkspace() {
  const [file, setFile] = useState<File | null>(null);
  const [url, setUrl] = useState<string | null>(null);
  const [metadata, setMetadata] = useState<{
    duration: number;
    width: number;
    height: number;
  } | null>(null);
  const [previewError, setPreviewError] = useState(false);
  const [phase, setPhase] = useState<Phase>("empty");
  const [health, setHealth] = useState<Health>("checking");
  const [healthAttempt, setHealthAttempt] = useState(0);
  const [result, setResult] = useState<Prediction | null>(null);
  const [error, setError] = useState("");
  const [progress, setProgress] = useState<number | null>(null);
  const [dragging, setDragging] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const request = useRef<XMLHttpRequest | null>(null);
  const busy = phase === "uploading" || phase === "analyzing";
  const tooLong = metadata !== null && metadata.duration > 120;

  useEffect(() => {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 8000);
    fetch(`${API}/health`, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error("Health check failed");
        const body: unknown = await response.json();
        if (
          typeof body !== "object" ||
          body === null ||
          !("model_ready" in body) ||
          typeof body.model_ready !== "boolean"
        )
          throw new Error("Invalid health response");
        setHealth(body.model_ready ? "ready" : "unavailable");
      })
      .catch(() => setHealth("offline"))
      .finally(() => window.clearTimeout(timeout));
    return () => {
      controller.abort();
      window.clearTimeout(timeout);
    };
  }, [healthAttempt]);

  useEffect(
    () => () => {
      if (url) URL.revokeObjectURL(url);
    },
    [url],
  );

  useEffect(() => () => request.current?.abort(), []);

  function choose(next: File | null) {
    if (busy) return;
    setError("");
    if (
      next &&
      (!FORMATS.test(next.name) || !next.size || next.size > MAX_BYTES)
    ) {
      setError(
        !FORMATS.test(next.name)
          ? "Choose an MP4, AVI, MOV, MKV, or WebM video."
          : !next.size
            ? "This file is empty. Choose a video with content."
            : "This video exceeds 100 MiB. Choose a smaller clip.",
      );
      return;
    }
    setFile(next);
    setUrl(next ? URL.createObjectURL(next) : null);
    setMetadata(null);
    setPreviewError(false);
    setResult(null);
    setProgress(null);
    setPhase(next ? "selected" : "empty");
    if (input.current) input.current.value = "";
  }

  function analyze() {
    if (!file || busy || tooLong) return;
    setError("");
    setResult(null);
    setProgress(null);
    setPhase("uploading");
    const xhr = new XMLHttpRequest();
    request.current = xhr;
    xhr.open("POST", `${API}/api/v1/predict`);
    xhr.timeout = 300_000;
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable)
        setProgress(Math.round((event.loaded / event.total) * 100));
    };
    xhr.upload.onload = () => {
      setProgress(100);
      setPhase("analyzing");
    };
    function fail(message: string) {
      setError(message);
      setPhase("failed");
      request.current = null;
    }
    xhr.onload = () => {
      let body: unknown;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        fail(`The API returned an unreadable response (${xhr.status}).`);
        return;
      }
      if (xhr.status < 200 || xhr.status >= 300) {
        const detail =
          typeof body === "object" &&
          body !== null &&
          "detail" in body &&
          typeof body.detail === "string"
            ? body.detail
            : `Analysis failed (${xhr.status}). Please try again.`;
        fail(detail);
        return;
      }
      try {
        setResult(parsePrediction(body));
        setPhase("complete");
        request.current = null;
      } catch (cause) {
        fail(
          cause instanceof Error ? cause.message : "Invalid model response.",
        );
      }
    };
    xhr.onerror = () =>
      fail(
        "Cannot reach the inference API. Check that it is running and allows this web origin.",
      );
    xhr.ontimeout = () =>
      fail(
        "The request timed out after 5 minutes. Try a shorter video; the server may still be finishing this request.",
      );
    const form = new FormData();
    form.append("video", file);
    xhr.send(form);
  }

  const statuses: Record<Health, string> = {
    checking: "Checking model",
    ready: "Model ready",
    unavailable: "Model not loaded",
    offline: "API unreachable",
  };

  return (
    <>
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3 text-xs">
        <div className="flex items-center gap-2">
          <span className="h-4 w-0.5 bg-primary" aria-hidden="true" />
          <span className="font-medium">Single-video analysis</span>
          <span className="text-muted-foreground">/ RGB only</span>
        </div>
        <div className="flex items-center gap-2" role="status">
          <span
            aria-hidden="true"
            className={cn(
              "size-1.5 rounded-full",
              health === "ready" ? "bg-primary" : "bg-muted-foreground",
            )}
          />
          <span>{statuses[health]}</span>
          {(health === "offline" || health === "unavailable") && (
            <button
              type="button"
              className="ml-1 underline underline-offset-4"
              onClick={() => {
                setHealth("checking");
                setHealthAttempt((value) => value + 1);
              }}
            >
              Retry connection
            </button>
          )}
        </div>
      </div>
      <div className="grid items-start gap-5 lg:grid-cols-3">
        <section
          aria-labelledby="source-title"
          className="overflow-hidden rounded-lg border bg-card lg:col-span-2"
        >
          <div className="flex items-center justify-between border-b px-5 py-4">
            <h2 id="source-title" className="text-sm font-semibold">
              Source video
            </h2>
            <span className="font-mono text-xs text-muted-foreground">
              01 / INPUT
            </span>
          </div>
          <div className="p-4 sm:p-5">
            <input
              ref={input}
              id="video-file"
              type="file"
              className="sr-only"
              tabIndex={-1}
              aria-label="Choose video file"
              accept=".mp4,.avi,.mov,.mkv,.webm"
              disabled={busy}
              onChange={(event) => choose(event.target.files?.[0] ?? null)}
            />
            <div
              onDragOver={(event) => {
                event.preventDefault();
                if (!busy) setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={(event) => {
                event.preventDefault();
                setDragging(false);
                if (busy) return;
                if (event.dataTransfer.files.length !== 1) {
                  setError("Choose one video at a time.");
                  return;
                }
                choose(event.dataTransfer.files[0]);
              }}
              className={cn(
                "relative flex aspect-video items-center justify-center overflow-hidden rounded-md border",
                dragging
                  ? "border-primary bg-primary/10"
                  : file
                    ? "border-border bg-neutral-950"
                    : "border-dashed bg-muted/40",
              )}
            >
              {file && url ? (
                <>
                  <video
                    key={url}
                    src={url}
                    controls
                    preload="metadata"
                    aria-label={`Preview of ${file.name}`}
                    className="h-full w-full object-contain"
                    onLoadedMetadata={(event) => {
                      const video = event.currentTarget;
                      if (Number.isFinite(video.duration))
                        setMetadata({
                          duration: video.duration,
                          width: video.videoWidth,
                          height: video.videoHeight,
                        });
                    }}
                    onError={() => setPreviewError(true)}
                  ></video>
                  {previewError && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center bg-muted p-6 text-center">
                      <FileVideo
                        aria-hidden="true"
                        className="mb-3 size-7 text-muted-foreground"
                      />
                      <p className="text-sm font-medium">
                        Preview unavailable in this browser
                      </p>
                      <p className="mt-2 max-w-xs text-xs leading-5 text-muted-foreground">
                        You can still submit this format for server-side
                        decoding and analysis.
                      </p>
                    </div>
                  )}
                </>
              ) : (
                <div className="p-5 text-center">
                  <div className="mx-auto mb-4 flex size-12 items-center justify-center rounded-lg border bg-card">
                    <Upload
                      aria-hidden="true"
                      className="size-5 text-muted-foreground"
                    />
                  </div>
                  <p className="text-sm font-medium">Drop a video here</p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    or select a file from your device
                  </p>
                  <Button
                    type="button"
                    className="mt-5"
                    variant="outline"
                    onClick={() => input.current?.click()}
                  >
                    Choose video
                  </Button>
                </div>
              )}
              {dragging && (
                <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-card/95 text-sm font-medium">
                  Drop to select this video
                </div>
              )}
            </div>
            {file ? (
              <div className="mt-4 flex min-w-0 items-start justify-between gap-3">
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium" title={file.name}>
                    {file.name}
                  </p>
                  <p className="mt-1 font-mono text-xs text-muted-foreground">
                    {(file.size / 1024 / 1024).toFixed(1)} MiB
                    {metadata &&
                      ` · ${metadata.duration.toFixed(1)} sec · ${metadata.width} × ${metadata.height}`}
                  </p>
                </div>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  disabled={busy}
                  onClick={() => choose(null)}
                >
                  <X aria-hidden="true" />
                  Remove
                </Button>
              </div>
            ) : (
              <p className="mt-4 text-xs leading-5 text-muted-foreground">
                MP4, AVI, MOV, MKV, WebM <span aria-hidden="true">·</span> Up to
                100 MiB and 2 minutes
                <br />
                MP4 with H.264 is recommended for browser preview.
              </p>
            )}
            {tooLong && (
              <p role="alert" className="mt-3 text-sm text-destructive">
                This video exceeds 2 minutes. Choose a shorter clip.
              </p>
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
              {file
                ? "Video is sent only when you run analysis."
                : "Your video stays local until you run analysis."}
            </p>
            {file && (
              <Button
                type="button"
                size="sm"
                variant="outline"
                disabled={busy}
                onClick={() => input.current?.click()}
              >
                Replace video
              </Button>
            )}
          </div>
        </section>
        <AnalysisResult
          phase={phase}
          health={health}
          result={result}
          progress={progress}
          disabled={!file || busy || tooLong || health !== "ready"}
          onAnalyze={analyze}
        />
      </div>
    </>
  );
}
