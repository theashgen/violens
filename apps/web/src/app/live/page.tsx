import { ArrowLeft } from "lucide-react";
import Link from "next/link";
import { LiveView } from "@/components/live-view";

export const metadata = {
  title: "Live Cameras · Violence Detection",
  description: "Realtime violence status from a webcam or RTSP/IP camera.",
};

export default function LivePage() {
  return (
    <>
      <header className="border-b bg-card">
        <div className="mx-auto flex max-w-7xl items-center justify-between gap-3 px-4 py-3 sm:px-8 sm:py-5">
          <div className="flex items-center gap-2.5 sm:gap-3 min-w-0">
            <Link
              href="/"
              className="flex size-8 items-center justify-center rounded-md border bg-card text-muted-foreground transition-colors hover:bg-muted sm:size-9 shrink-0"
              aria-label="Back to video analysis"
            >
              <ArrowLeft className="size-4" aria-hidden="true" />
            </Link>
            <div className="min-w-0">
              <p className="truncate text-sm font-semibold tracking-tight">
                Live Cameras
              </p>
              <p className="truncate text-[11px] text-muted-foreground sm:text-xs">
                Realtime status · research baseline
              </p>
            </div>
          </div>
          <Link
            href="/monitor"
            className="inline-flex shrink-0 items-center justify-center rounded-md border px-2.5 py-1.5 text-xs font-medium transition-colors hover:bg-muted sm:px-3 min-h-[34px]"
          >
            Multi-camera monitor
          </Link>
        </div>
      </header>
      <main className="mx-auto max-w-7xl px-4 py-6 sm:px-8 sm:py-10">
        <LiveView />
      </main>
      <footer className="mx-auto flex max-w-7xl flex-col sm:flex-row justify-between gap-2 px-4 pb-6 text-xs text-muted-foreground sm:px-8">
        <span>Violence Detection / Realtime baseline</span>
        <span>Human review remains essential.</span>
      </footer>
    </>
  );
}
