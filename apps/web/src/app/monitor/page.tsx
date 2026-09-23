import Link from "next/link";
import { CameraDashboard } from "@/components/camera-dashboard";

export const metadata = { title: "Camera Monitoring · Violence Detection" };

export default function MonitorPage() {
  return <><header className="border-b bg-card"><div className="mx-auto flex max-w-6xl items-center justify-between px-5 py-4 sm:px-8"><Link href="/" className="text-sm font-semibold">Violence Detection</Link><Link href="/live" className="text-sm underline">Single-camera live test</Link></div></header><CameraDashboard /><footer className="mx-auto max-w-6xl px-5 pb-6 text-xs text-muted-foreground">Research system only. Human review required; not validated as a safety or law-enforcement tool.</footer></>;
}
