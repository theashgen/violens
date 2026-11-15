import Link from "next/link";
import { CameraDashboard } from "@/components/camera-dashboard";

export const metadata = { title: "Camera Monitoring · Violence Detection" };

export default function MonitorPage() {
  return (
    <>
      <header className="border-b bg-card">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-3 px-4 py-3 sm:px-8 sm:py-4">
          <Link href="/" className="text-sm font-semibold tracking-tight hover:underline">
            Violence Detection
          </Link>
          <Link
            href="/live"
            className="inline-flex shrink-0 items-center justify-center rounded-md border px-2.5 py-1.5 text-xs font-medium transition-colors hover:bg-muted sm:px-3 min-h-[34px]"
          >
            Single-camera live test
          </Link>
        </div>
      </header>
      <CameraDashboard />
      <footer className="mx-auto max-w-6xl px-4 pb-6 text-xs text-muted-foreground sm:px-8">
        Research system only. Human review required; not validated as a safety or law-enforcement tool.
      </footer>
    </>
  );
}
