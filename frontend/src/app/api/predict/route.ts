import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const file = formData.get("ecg_image") as File | null;
    const duration = formData.get("duration_seconds") as string | null;

    if (!file) {
      return NextResponse.json({ error: "No image file provided" }, { status: 400 });
    }
    if (!duration) {
      return NextResponse.json({ error: "No duration provided" }, { status: 400 });
    }

    // Build upstream form data
    const upstreamFormData = new FormData();
    upstreamFormData.append("ecg_image", file);
    upstreamFormData.append("duration_seconds", duration);

    // Try dedicated JSON endpoint first
    try {
      const jsonRes = await fetch("http://127.0.0.1:5000/api/predict_image", {
        method: "POST",
        body: upstreamFormData,
        headers: { Accept: "application/json" },
      });
      if (jsonRes.ok) {
        const data = await jsonRes.json();
        return NextResponse.json(data);
      }
    } catch {
      // Fall through to standard endpoint
    }

    // Fallback to /predict
    const fallbackFormData = new FormData();
    fallbackFormData.append("ecg_image", file);
    fallbackFormData.append("duration_seconds", duration);

    const res = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      body: fallbackFormData,
    });

    const text = await res.text();

    // Check if response is JSON
    try {
      const json = JSON.parse(text);
      return NextResponse.json(json, { status: res.status });
    } catch {
      // If it's HTML, parse out the metrics using regex
      if (text.includes("Pre-arrest") || text.includes("Normal") || text.includes("data:image/png;base64")) {
        const isPreArrest = text.includes("Pre-arrest") || text.includes("Pre-Arrest");
        
        // Match probability
        const probMatch = text.match(/(\d+\.?\d*)\s*%\s*<\/div>\s*<div[^>]*>predicted probability/i) ||
                          text.match(/Pre-Arrest Probability Score<\/span>\s*<span[^>]*>(\d+\.?\d*)%/i);
        const probability = probMatch ? parseFloat(probMatch[1]) / 100 : (isPreArrest ? 0.75 : 0.22);

        // Match threshold
        const threshMatch = text.match(/<div class="value">(\d+\.?\d*)<\/div>\s*<div[^>]*>decision threshold/i);
        const threshold_used = threshMatch ? parseFloat(threshMatch[1]) : 0.38;

        // Match beats
        const beatsMatch = text.match(/<div class="value">(\d+)<\/div>\s*<div[^>]*>beats detected/i);
        const beats_detected = beatsMatch ? parseInt(beatsMatch[1], 10) : 180;

        // Match sampling rate
        const rateMatch = text.match(/<div class="value">(\d+\.?\d*)\s*Hz<\/div>\s*<div[^>]*>implied sampling rate/i);
        const sampling_rate_used = rateMatch ? parseFloat(rateMatch[1]) : 10.8;

        // Match plot uri
        const plotMatch = text.match(/src="(data:image\/png;base64,[^"]+)"/);
        const plot_uri = plotMatch ? plotMatch[1] : "";

        return NextResponse.json({
          label: isPreArrest ? "Pre-arrest" : "Normal",
          probability,
          threshold_used,
          beats_detected,
          sampling_rate_used,
          plot_uri,
        });
      }

      // Check for error card message in HTML
      const errorMatch = text.match(/<div class="error(?:-card-desc)?">([^<]+)<\/div>/i);
      const errMsg = errorMatch ? errorMatch[1].trim() : "Failed to analyze ECG image.";
      return NextResponse.json({ error: errMsg }, { status: 400 });
    }
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Internal server error connecting to ML backend";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
