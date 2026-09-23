import { expect, test } from "@playwright/test";

const SNAPSHOT_BYTES = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
  "base64",
);

test.use({
  permissions: ["camera"],
  launchOptions: {
    args: [
      "--use-fake-device-for-media-stream",
      "--use-fake-ui-for-media-stream",
      "--autoplay-policy=no-user-gesture-required",
    ],
  },
});

test.beforeEach(async ({ page }) => {
  await page.route("**/health", (route) =>
    route.fulfill({ json: { model_ready: true } }),
  );
});

test("rejects a malformed camera URL without contacting the API", async ({
  page,
}) => {
  let connectCalled = false;
  await page.route("**/api/v1/stream/rtsp/connect", (route) => {
    connectCalled = true;
    return route.fulfill({ json: { source: "rtsp", frames_needed: 16 } });
  });
  await page.goto("/live");
  await expect(page.getByText("No camera connected")).toBeVisible();
  await page.getByLabel(/Stream URL/).fill("not a url");
  await page.getByRole("button", { name: "Connect camera" }).click();
  await expect(page.getByText("Enter an rtsp://")).toBeVisible();
  expect(connectCalled).toBe(false);
});

test("webcam connect, live status, prediction, and disconnect", async ({
  page,
}) => {
  await page.route("**/api/v1/stream/webcam/connect", (route) =>
    route.fulfill({ json: { source: "webcam", max_fps: 30, frames_needed: 16 } }),
  );
  await page.route("**/api/v1/stream/webcam/frames", (route) =>
    route.fulfill({ json: { accepted: 1, buffered: 5 } }),
  );
  await page.route("**/api/v1/stream/webcam/status", (route) =>
    route.fulfill({
      json: {
        source: "webcam",
        active: true,
        buffered_frames: 16,
        frames_needed: 16,
        dropped_frames: 0,
        latest_prediction: { prediction: "violence", confidence: 0.87 },
        analyzing: true,
        last_analysis_error: null,
        connected: null,
        last_error: null,
        seconds_since_frame: 0.2,
        interval_seconds: 5,
      },
    }),
  );
  await page.route("**/api/v1/stream/webcam/snapshot", (route) =>
    route.fulfill({ body: SNAPSHOT_BYTES, contentType: "image/png" }),
  );
  await page.route("**/api/v1/stream/webcam/stop", (route) =>
    route.fulfill({ json: { stopped: "webcam" } }),
  );
  await page.goto("/live");
  await page.getByRole("button", { name: "Connect webcam" }).click();
  await expect(page.getByText("Analyzing", { exact: true })).toBeVisible({
    timeout: 10_000,
  });
  await expect(page.getByText("Violence", { exact: true })).toBeVisible();
  await expect(page.getByText("16/16 frames")).toBeVisible();
  await page.getByRole("button", { name: "Disconnect" }).click();
  await expect(page.getByText("No camera connected")).toBeVisible();
  await expect(page.getByText("Violence", { exact: true })).not.toBeVisible();
});

test("rtsp reconnecting state and camera error surface in status", async ({
  page,
}) => {
  await page.route("**/api/v1/stream/rtsp/connect", (route) =>
    route.fulfill({ json: { source: "rtsp", frames_needed: 16 } }),
  );
  await page.route("**/api/v1/stream/rtsp/status", (route) =>
    route.fulfill({
      json: {
        source: "rtsp",
        active: true,
        buffered_frames: 0,
        frames_needed: 16,
        dropped_frames: 0,
        latest_prediction: null,
        analyzing: false,
        last_analysis_error: null,
        connected: false,
        last_error: "Camera error: ConnectionRefusedError.",
        seconds_since_frame: null,
        interval_seconds: 5,
      },
    }),
  );
  await page.route("**/api/v1/stream/rtsp/snapshot", (route) =>
    route.fulfill({ status: 404, json: { detail: "No frame captured yet." } }),
  );
  await page.goto("/live");
  await page.getByLabel(/Stream URL/).fill("rtsp://cam.local/stream");
  await page.getByRole("button", { name: "Connect camera" }).click();
  // The connecting phase is transient; assert the steady reconnecting state.
  await expect(page.getByText("Reconnecting")).toBeVisible({ timeout: 10_000 });
  await expect(
    page.getByText("Camera error: ConnectionRefusedError."),
  ).toBeVisible();
});
