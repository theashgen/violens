/**
 * End-to-end check of the realtime pipeline against the REAL API:
 * fake webcam frames -> /frames push -> ring buffer -> model inference -> status.
 * Run with the API and web dev servers up: node scripts/e2e-live-check.mjs
 */
import { chromium } from "@playwright/test";

const API = "http://localhost:8000";
const WEB = "http://localhost:3000";

const health = await fetch(`${API}/health`).then((r) => r.json());
if (!health.model_ready) {
  console.error("Model not ready; start the API with a checkpoint first.");
  process.exit(1);
}

const browser = await chromium.launch({
  args: [
    "--use-fake-device-for-media-stream",
    "--use-fake-ui-for-media-stream",
    "--autoplay-policy=no-user-gesture-required",
  ],
});
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });

try {
  // Clear any session left over from another client; the API allows one at a time.
  for (const source of ["webcam", "rtsp"]) {
    await fetch(`${API}/api/v1/stream/${source}/stop`, { method: "POST" }).catch(() => {});
  }
  await page.goto(`${WEB}/live`);
  await page.getByRole("button", { name: "Connect webcam" }).click();
  await page
    .getByText("Connecting…")
    .waitFor({ state: "hidden", timeout: 20_000 });

  // Poll the REAL status endpoint until real model inference has produced
  // a prediction from the fake-camera frames pushed through the browser.
  const deadline = Date.now() + 60_000;
  let status = null;
  while (Date.now() < deadline) {
    status = await page.evaluate(async (url) => {
      const response = await fetch(`${url}/api/v1/stream/webcam/status`);
      return response.json();
    }, API);
    if (status?.latest_prediction) break;
    await page.waitForTimeout(2_000);
  }
  if (!status?.latest_prediction) {
    console.error("STATUS DUMP:", JSON.stringify(status, null, 2));
    throw new Error("No prediction produced within 60s.");
  }
  const { prediction, confidence } = status.latest_prediction;
  if (
    !(prediction === "violence" || prediction === "non-violence") ||
    typeof confidence !== "number" ||
    confidence < 0 ||
    confidence > 1
  ) {
    throw new Error(`Malformed prediction: ${JSON.stringify(status.latest_prediction)}`);
  }
  console.log("prediction:", prediction, "confidence:", confidence.toFixed(3));
  console.log("buffered:", status.buffered_frames, "dropped:", status.dropped_frames);
  console.log("E2E PASS: webcam -> frames -> ring buffer -> model -> status all real.");
} finally {
  await page
    .evaluate(() =>
      fetch("http://localhost:8000/api/v1/stream/webcam/stop", { method: "POST" }),
    )
    .catch(() => {});
  await browser.close();
}
