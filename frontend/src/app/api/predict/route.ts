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

    const upstreamFormData = new FormData();
    upstreamFormData.append("ecg_image", file);
    upstreamFormData.append("duration_seconds", duration);

    const res = await fetch("http://127.0.0.1:5000/api/predict_image", {
      method: "POST",
      body: upstreamFormData,
      headers: { Accept: "application/json" },
    });

    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Internal server error connecting to ML backend";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
