export const API = (
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
).replace(/\/$/, "");

export type Prediction = {
  prediction: "violence" | "non-violence";
  confidence: number;
};

export function parsePrediction(value: unknown): Prediction {
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

export async function readErrorDetail(
  response: Response,
  fallback: string,
): Promise<string> {
  try {
    const body: unknown = await response.json();
    if (
      typeof body === "object" &&
      body !== null &&
      "detail" in body &&
      typeof body.detail === "string"
    ) {
      return body.detail;
    }
  } catch {
    // Fall through to the generic message.
  }
  return fallback;
}
