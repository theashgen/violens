import { Aperture, ArrowUpRight, Radio } from "lucide-react";
import Link from "next/link";
import { AnalysisWorkspace } from "@/components/analysis-workspace";

export default function Page() {
  return (
    <>
      <a
        href="#main"
        className="sr-only z-50 bg-card p-3 focus:not-sr-only focus:absolute"
      >
        Skip to analysis
      </a>
      <header className="border-b bg-card">
        <div className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-5 py-5 sm:px-8">
          <div className="flex items-center gap-3">
            <span className="flex size-9 items-center justify-center rounded-md bg-primary text-primary-foreground">
              <Aperture className="size-5" aria-hidden="true" />
            </span>
            <div>
              <p className="text-sm font-semibold tracking-tight">
                Violence Detection
              </p>
              <p className="mt-0.5 text-xs text-muted-foreground">
                Visual intelligence research
              </p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <Link
              href="/monitor"
              className="inline-flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-xs font-medium transition-colors hover:bg-muted"
            >
              Camera monitor
            </Link>
            <Link
              href="/live"
              className="inline-flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-xs font-medium transition-colors hover:bg-muted"
            >
              <Radio aria-hidden="true" className="size-3.5" />
              Live cameras
            </Link>
            <span className="font-mono text-xs text-muted-foreground">
              PHASE 02 <span className="hidden sm:inline">/ VIDEO + LIVE MONITORING</span>
            </span>
          </div>
        </div>
      </header>
      <main id="main" className="mx-auto max-w-7xl px-5 py-8 sm:px-8 sm:py-10">
        <AnalysisWorkspace />
        <section
          aria-labelledby="pipeline-title"
          className="mt-8 border-t pt-6"
        >
          <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-start">
            <div>
              <h2 id="pipeline-title" className="text-sm font-medium">
                A transparent video baseline
              </h2>
              <p className="mt-1 text-xs leading-5 text-muted-foreground">
                One whole-video prediction. No audio or event localization.
              </p>
            </div>
            <a
              href="https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.video.mc3_18.html"
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1 text-xs text-muted-foreground underline-offset-4 hover:underline"
            >
              Model reference{" "}
              <ArrowUpRight className="size-3" aria-hidden="true" />
              <span className="sr-only"> (opens in a new tab)</span>
            </a>
          </div>
          <ol className="mt-5 grid grid-cols-2 gap-4 text-xs sm:grid-cols-4">
            {[
              ["01", "Decode", "Read the video frames"],
              ["02", "Sample", "Select frames across the clip"],
              ["03", "Preprocess", "Apply pretrained transforms"],
              ["04", "Classify", "MC3-18 · two classes"],
            ].map(([number, title, text]) => (
              <li key={number} className="border-l-2 pl-3">
                <span className="font-mono text-muted-foreground">
                  {number}
                </span>
                <p className="mt-1 font-medium">{title}</p>
                <p className="mt-1 leading-5 text-muted-foreground">{text}</p>
              </li>
            ))}
          </ol>
        </section>
      </main>
      <footer className="mx-auto flex max-w-7xl flex-wrap justify-between gap-2 px-5 pb-6 text-xs text-muted-foreground sm:px-8">
        <span>Violence Detection / Research baseline</span>
        <span>Human review remains essential.</span>
      </footer>
    </>
  );
}
