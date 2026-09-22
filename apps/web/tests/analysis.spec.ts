import { expect, test } from "@playwright/test";

test.beforeEach(async ({ page }) => {
  await page.route("**/health", (route) =>
    route.fulfill({ json: { model_ready: true } }),
  );
});

test("empty, selected, waiting, success, and removal", async ({ page }) => {
  let finish: (() => void) | undefined;
  const gate = new Promise<void>((resolve) => {
    finish = resolve;
  });
  await page.route("**/api/v1/predict", async (route) => {
    await gate;
    await route.fulfill({
      json: { prediction: "violence", confidence: 0.923 },
    });
  });
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: "Run analysis" }),
  ).toBeDisabled();
  await page
    .getByLabel("Choose video file")
    .setInputFiles({
      name: "sample.mp4",
      mimeType: "video/mp4",
      buffer: Buffer.from("test fixture"),
    });
  await expect(page.getByText("sample.mp4", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Run analysis" }).click();
  await expect(
    page.getByText("Uploading video", { exact: true }),
  ).toBeVisible();
  await expect(page.getByRole("button", { name: "Remove" })).toBeDisabled();
  finish?.();
  await expect(page.getByText("Violence", { exact: true })).toBeVisible();
  await expect(page.getByRole("meter")).toHaveAttribute("value", "0.923");
  await page.getByRole("button", { name: "Remove" }).click();
  await expect(page.getByText("Awaiting video", { exact: true })).toBeVisible();
  await expect(page.getByText("Violence", { exact: true })).not.toBeVisible();
});

test("invalid selection, backend error, and retry", async ({ page }) => {
  await page.route("**/api/v1/predict", (route) =>
    route.fulfill({
      status: 422,
      json: { detail: "Video contains no decodable frames." },
    }),
  );
  await page.goto("/");
  await page
    .getByLabel("Choose video file")
    .setInputFiles({
      name: "notes.txt",
      mimeType: "text/plain",
      buffer: Buffer.from("text"),
    });
  await expect(
    page.getByRole("region", { name: "Source video" }).getByRole("alert"),
  ).toContainText("Choose an MP4");
  await page
    .getByLabel("Choose video file")
    .setInputFiles({
      name: "broken.mp4",
      mimeType: "video/mp4",
      buffer: Buffer.from("bad video"),
    });
  await page.getByRole("button", { name: "Run analysis" }).click();
  await expect(
    page.getByRole("region", { name: "Source video" }).getByRole("alert"),
  ).toContainText("no decodable frames");
  await expect(page.getByText("No prediction available")).toBeVisible();
  await page.route("**/api/v1/predict", (route) =>
    route.fulfill({ json: { prediction: "non-violence", confidence: 0.8 } }),
  );
  await page.getByRole("button", { name: "Run again" }).click();
  await expect(page.getByText("Non-Violence", { exact: true })).toBeVisible();
});

test("unavailable checkpoint, keyboard upload, and mobile layout", async ({
  page,
}) => {
  await page.route("**/health", (route) =>
    route.fulfill({ json: { model_ready: false } }),
  );
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await expect(
    page.getByText("Model not loaded", { exact: true }),
  ).toBeVisible();
  const choose = page.getByRole("button", {
    name: "Choose video",
    exact: true,
  });
  await choose.focus();
  const chooser = page.waitForEvent("filechooser");
  await page.keyboard.press("Enter");
  await (
    await chooser
  ).setFiles({
    name: "clip.mp4",
    mimeType: "video/mp4",
    buffer: Buffer.from("fixture"),
  });
  await expect(
    page.getByRole("button", { name: "Run analysis" }),
  ).toBeDisabled();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
});

test("rejects invalid probability instead of displaying fabricated output", async ({
  page,
}) => {
  await page.route("**/api/v1/predict", (route) =>
    route.fulfill({ json: { prediction: "violence", confidence: 9.2 } }),
  );
  await page.goto("/");
  await page
    .getByLabel("Choose video file")
    .setInputFiles({
      name: "clip.mp4",
      mimeType: "video/mp4",
      buffer: Buffer.from("fixture"),
    });
  await page.getByRole("button", { name: "Run analysis" }).click();
  await expect(
    page.getByRole("region", { name: "Source video" }).getByRole("alert"),
  ).toContainText("invalid prediction");
});
