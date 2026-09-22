import { ArrowRight, Check, LoaderCircle, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export type Prediction = {
  prediction: "violence" | "non-violence";
  confidence: number;
};
export type Phase =
  "empty" | "selected" | "uploading" | "analyzing" | "complete" | "failed";
export type Health = "checking" | "ready" | "unavailable" | "offline";

type Props = {
  phase: Phase;
  health: Health;
  result: Prediction | null;
  progress: number | null;
  disabled: boolean;
  onAnalyze: () => void;
};

export function AnalysisResult({
  phase,
  health,
  result,
  progress,
  disabled,
  onAnalyze,
}: Props) {
  const busy = phase === "uploading" || phase === "analyzing";
  const phaseNames: Record<Phase, string> = {
    empty: "Awaiting video",
    selected: "Ready to analyze",
    uploading: "Uploading video",
    analyzing: "Waiting for inference",
    complete: "Analysis complete",
    failed: "Analysis failed",
  };
  return (
    <section
      aria-labelledby="analysis-title"
      aria-busy={busy}
      className="rounded-lg border bg-card"
    >
      <div className="flex items-center justify-between border-b px-5 py-4">
        <h2 id="analysis-title" className="text-sm font-semibold">
          Analysis
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
          {busy ? (
            <LoaderCircle
              aria-hidden="true"
              className="size-3.5 motion-safe:animate-spin"
            />
          ) : phase === "complete" ? (
            <Check aria-hidden="true" className="size-3.5" />
          ) : (
            <span
              className="size-1.5 rounded-full bg-muted-foreground"
              aria-hidden="true"
            />
          )}
          {phaseNames[phase]}
        </div>
        <div aria-live="polite" className="min-h-48 py-7">
          {result ? (
            <>
              <p className="text-xs text-muted-foreground">Prediction</p>
              <p
                className={cn(
                  "mt-2 text-3xl font-semibold tracking-tight",
                  result.prediction === "violence"
                    ? "text-destructive"
                    : "text-primary",
                )}
              >
                {result.prediction === "violence" ? "Violence" : "Non-Violence"}
              </p>
              <div className="mt-6 flex items-end justify-between">
                <label
                  htmlFor="confidence"
                  className="text-xs text-muted-foreground"
                >
                  Model confidence
                </label>
                <span className="font-mono text-xl">
                  {(result.confidence * 100).toFixed(1)}
                  <span className="ml-0.5 text-sm">%</span>
                </span>
              </div>
              <meter
                id="confidence"
                min={0}
                max={1}
                value={result.confidence}
                className="mt-2 h-3 w-full accent-primary"
              >
                {(result.confidence * 100).toFixed(1)}%
              </meter>
            </>
          ) : (
            <>
              <p className="text-lg font-medium">
                {busy
                  ? phase === "uploading"
                    ? "Sending your video…"
                    : "Processing your video…"
                  : phase === "failed"
                    ? "No prediction available"
                    : "Ready when you are"}
              </p>
              <p className="mt-3 text-sm leading-6 text-muted-foreground">
                {phase === "uploading"
                  ? "Transferring the file to the inference API."
                  : phase === "analyzing"
                    ? "Upload complete. Waiting for the server to decode the video and return the model prediction."
                    : phase === "failed"
                      ? "Review the error beside your video, then try again."
                      : "Select a video and run analysis. The model’s prediction and confidence will appear here."}
              </p>
              {phase === "uploading" && (
                <>
                  <label
                    htmlFor="upload-progress"
                    className="mt-4 block text-xs text-muted-foreground"
                  >
                    Upload {progress !== null ? `${progress}%` : "in progress"}
                  </label>
                  <progress
                    id="upload-progress"
                    max={100}
                    value={progress ?? undefined}
                    className="mt-2 h-2 w-full accent-primary"
                  />
                </>
              )}
            </>
          )}
        </div>
        <dl className="space-y-3 border-y py-4 text-xs">
          <div className="flex justify-between gap-3">
            <dt className="text-muted-foreground">Architecture</dt>
            <dd className="font-mono">MC3-18</dd>
          </div>
          <div className="flex justify-between gap-3">
            <dt className="text-muted-foreground">Input</dt>
            <dd>Video / RGB</dd>
          </div>
          <div className="flex justify-between gap-3">
            <dt className="text-muted-foreground">Output</dt>
            <dd>Binary classification</dd>
          </div>
        </dl>
        <Button
          type="button"
          className="mt-5 w-full"
          disabled={disabled}
          onClick={onAnalyze}
        >
          {busy ? (
            <>
              <LoaderCircle
                aria-hidden="true"
                className="motion-safe:animate-spin"
              />
              {phase === "uploading" ? "Uploading…" : "Analyzing…"}
            </>
          ) : (
            <>
              {phase === "complete" || phase === "failed" ? (
                <RotateCcw aria-hidden="true" />
              ) : null}
              {phase === "complete" || phase === "failed"
                ? "Run again"
                : "Run analysis"}
              <ArrowRight aria-hidden="true" className="ml-auto" />
            </>
          )}
        </Button>
        {(health === "unavailable" || health === "offline") && (
          <p className="mt-3 text-xs leading-5 text-muted-foreground">
            {health === "unavailable"
              ? "The API is running but needs a trained checkpoint. Configure it and restart the API, then retry the connection."
              : "Start the FastAPI server, then retry the connection to enable analysis."}
          </p>
        )}
        <p className="mt-4 text-xs leading-5 text-muted-foreground">
          Model-generated predictions can be incorrect. Confidence is a model
          score, not a guarantee. Review the footage before drawing conclusions.
        </p>
      </div>
    </section>
  );
}
