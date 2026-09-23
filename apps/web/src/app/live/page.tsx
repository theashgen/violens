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
        <div className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-5 py-5 sm:px-8">
          <div className="flex items-center gap-3">
            <Link
              href="/"
              className="flex size-9 items-center justify-center rounded-md border bg-card text-muted-foreground transition-colors hover:bg-muted"
              aria-label="Back to video analysis"
            >
              <ArrowLeft className="size-4" aria-hidden="true" />
            </Link>
            <div>
              <p className="text-sm font-semibold tracking-tight">
                Live Cameras
              </p>
              <p className="mt-0.5 text-xs text-muted-foreground">
                Realtime status · research baseline
              </p>
            </div>
          </div>
          <Link href="/monitor" className="text-sm underline">Multi-camera monitor</Link>
        </div>
      </header>
      <main className="mx-auto max-w-7xl px-5 py-8 sm:px-8 sm:py-10">
        <LiveView />
      </main>
      <footer className="mx-auto flex max-w-7xl flex-wrap justify-between gap-2 px-5 pb-6 text-xs text-muted-foreground sm:px-8">
        <span>Violence Detection / Realtime baseline</span>
        <span>Human review remains essential.</span>
      </footer>
    </>
  );
}
