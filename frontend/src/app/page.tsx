"use client";

import React, { useState, useEffect, useRef } from "react";
import {
  Heartbeat,
  Heart,
  UploadSimple,
  Image as ImageIcon,
  WarningCircle,
  CheckCircle,
  Clock,
  Pulse,
  Cpu,
  ArrowCounterClockwise,
  Printer,
  X,
  Waveform,
  ShieldCheck,
  Sparkle,
  Gauge,
  ArrowsClockwise,
  Sun,
  Moon,
} from "@phosphor-icons/react";

interface PredictionResult {
  label: "Pre-arrest" | "Normal";
  probability: number;
  threshold_used: number;
  beats_detected: number;
  sampling_rate_used: number;
  plot_uri: string;
}

export default function Home() {
  const [theme, setTheme] = useState<"light" | "dark">("light");
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [duration, setDuration] = useState<string>("180");
  const [isDragOver, setIsDragOver] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(false);
  const [loadingStage, setLoadingStage] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [showModal, setShowModal] = useState<boolean>(false);
  const [backendReady, setBackendReady] = useState<boolean>(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Initialize theme (defaults to Light Mode matching mockup)
  useEffect(() => {
    const saved = localStorage.getItem("cardiopulse-theme") as "light" | "dark" | null;
    const initialTheme = saved || "light";
    setTheme(initialTheme);
    document.documentElement.setAttribute("data-theme", initialTheme);
  }, []);

  const toggleTheme = () => {
    const nextTheme = theme === "light" ? "dark" : "light";
    setTheme(nextTheme);
    document.documentElement.setAttribute("data-theme", nextTheme);
    localStorage.setItem("cardiopulse-theme", nextTheme);
  };

  // Check backend health on mount
  useEffect(() => {
    async function checkHealth() {
      try {
        const res = await fetch("/api/backend/health");
        if (res.ok) {
          setBackendReady(true);
        } else {
          const directRes = await fetch("http://127.0.0.1:5000/health");
          if (directRes.ok) setBackendReady(true);
        }
      } catch {
        setBackendReady(false);
      }
    }
    checkHealth();
  }, []);

  // Format file size
  const formatBytes = (bytes?: number) => {
    if (!bytes) return "0 KB";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
  };

  // Set file and preview
  const handleFileSelect = (selectedFile: File) => {
    setError(null);
    setFile(selectedFile);
    const url = URL.createObjectURL(selectedFile);
    setPreviewUrl(url);
  };

  const handleClearFile = (e: React.MouseEvent) => {
    e.stopPropagation();
    setFile(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      setPreviewUrl(null);
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  // Drag & drop handlers
  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const onDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFile = e.dataTransfer.files[0];
      handleFileSelect(droppedFile);
    }
  };

  // Quick Sample Loader
  const handleLoadSample = async (filename: string, sampleDuration: number) => {
    try {
      setLoading(true);
      setLoadingStage(`Loading sample trace (${filename})...`);
      setError(null);

      const res = await fetch(`/samples/${filename}`);
      if (!res.ok) throw new Error("Could not load sample file");
      const blob = await res.blob();
      const sampleFile = new File([blob], filename, { type: blob.type || "image/png" });

      handleFileSelect(sampleFile);
      setDuration(String(sampleDuration));
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to load sample";
      setError(`Sample load failed: ${msg}`);
    } finally {
      setLoading(false);
      setLoadingStage("");
    }
  };

  // Form submission
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) {
      setError("Please select or drop an ECG waveform image.");
      return;
    }
    const durNum = parseFloat(duration);
    if (isNaN(durNum) || durNum < 5 || durNum > 3600) {
      setError("Please specify a duration between 5 and 3600 seconds.");
      return;
    }

    setLoading(true);
    setError(null);

    // Staged progress messaging
    const timer1 = setTimeout(() => setLoadingStage("Filtering vertical pixel trace columns..."), 400);
    const timer2 = setTimeout(() => setLoadingStage("Detecting R-peaks using NeuroKit2..."), 1200);
    const timer3 = setTimeout(() => setLoadingStage("Extracting 90 HRV non-linear biomarkers..."), 2000);
    const timer4 = setTimeout(() => setLoadingStage("Evaluating Random Forest pre-arrest classifier..."), 2800);

    const formData = new FormData();
    formData.append("ecg_image", file);
    formData.append("duration_seconds", duration);

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || "Analysis failed on server.");
      }

      setResult(data);
      setShowModal(true);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Error connecting to model backend.";
      setError(msg);
    } finally {
      clearTimeout(timer1);
      clearTimeout(timer2);
      clearTimeout(timer3);
      clearTimeout(timer4);
      setLoading(false);
      setLoadingStage("");
    }
  };

  return (
    <>
      {/* Ambient ECG Pulse Trace Overlay on the Left (Matching Mockup) */}
      <div className="ecg-ambient-trace" aria-hidden="true">
        <svg viewBox="0 0 540 160" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path
            d="M 0 95 L 90 95 C 102 95 106 82 112 82 C 118 82 122 95 132 95 L 148 95 L 158 20 L 172 145 L 184 75 L 192 95 L 230 95 C 242 95 248 70 260 70 C 272 70 278 95 292 95 L 320 95 L 330 30 L 344 140 L 356 80 L 364 95 L 540 95"
            stroke="url(#ecgLineGradient)"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <defs>
            <linearGradient id="ecgLineGradient" x1="0" y1="0" x2="540" y2="0" gradientUnits="userSpaceOnUse">
              <stop offset="0%" stopColor="#f43f5e" stopOpacity="0.04" />
              <stop offset="28%" stopColor="#f43f5e" stopOpacity="0.5" />
              <stop offset="70%" stopColor="#fb7185" stopOpacity="0.45" />
              <stop offset="100%" stopColor="#fb7185" stopOpacity="0.04" />
            </linearGradient>
          </defs>
        </svg>
      </div>

      {/* Top Apple Nav Header */}
      <header className="top-nav">
        <div className="nav-container">
          <a href="#" className="brand-mark" aria-label="CardioPulse AI Home">
            <img
              src="/logo.png"
              alt="CardioPulse AI"
              className="brand-logo-img"
            />
          </a>

          <div className="nav-right-cluster">
            <button
              type="button"
              className="theme-toggle-btn"
              onClick={toggleTheme}
              title={theme === "light" ? "Switch to Dark Mode" : "Switch to Light Mode"}
              aria-label="Toggle light and dark mode"
            >
              {theme === "light" ? (
                <Sun size={18} weight="bold" />
              ) : (
                <Moon size={18} weight="bold" />
              )}
            </button>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main className="page-main">
        {/* Hero Section */}
        <section className="hero-section">
          <h1 className="hero-title anim-stagger-1">
            Early Warning for
            <br />
            <span className="hero-prearrest-gradient">Pre-Arrest</span> Cardiac Signals
          </h1>
          <p className="hero-desc anim-stagger-2">
            Digitize standard ECG plots into clean 1D time-series, compute 90 non-linear Heart Rate Variability
            biomarkers, and detect pre-arrest patterns with tuned clinical sensitivity.
          </p>
        </section>

        {/* Error Alert */}
        {error && (
          <div className="error-card anim-stagger-3" role="alert">
            <WarningCircle size={22} weight="bold" />
            <div>
              <div className="error-card-title">Processing Notice</div>
              <div className="error-card-desc">{error}</div>
            </div>
          </div>
        )}

        {/* Main Floating Apple Card */}
        <section className="apple-card anim-stagger-4">
          {/* Quick Demo Traces Bar at Top of Card */}
          <div className="card-top-strip">
            <div className="strip-title">
              <Pulse size={18} weight="bold" />
              <span>Quick Demo Traces</span>
            </div>
            <div className="strip-pills">
              <button
                type="button"
                className={`pill-btn ${file?.name === "sample_ecg_180sec.png" ? "active" : ""}`}
                onClick={() => handleLoadSample("sample_ecg_180sec.png", 180)}
                title="Load a verified 180s Normal Sinus Rhythm recording"
              >
                <Pulse size={16} weight="bold" />
                <span>Normal Rhythm</span>
                <span className="pill-badge">180s</span>
              </button>
              <button
                type="button"
                className={`pill-btn ${file?.name === "sample_ecg_prearrest_130sec.png" ? "active" : ""}`}
                onClick={() => handleLoadSample("sample_ecg_prearrest_130sec.png", 130)}
                title="Load a verified 130s Pre-Arrest pattern recording"
              >
                <Pulse size={16} weight="bold" />
                <span>Pre-Arrest Pattern</span>
                <span className="pill-badge">130s</span>
              </button>
            </div>
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit} className="card-form">
            {/* Dropzone matching user mockup */}
            <div
              className={`dropzone-box ${isDragOver ? "is-dragover" : ""} ${file ? "has-file" : ""}`}
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
              onDrop={onDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                id="ecg_file_input"
                type="file"
                accept="image/*"
                style={{ display: "none" }}
                onChange={(e) => {
                  if (e.target.files && e.target.files[0]) {
                    handleFileSelect(e.target.files[0]);
                  }
                }}
              />

              {!file ? (
                <div className="dropzone-empty-state">
                  <div className="dropzone-icon-badge">
                    <UploadSimple size={24} weight="bold" />
                  </div>
                  <div className="dropzone-text-row">
                    <span className="drop-pink-text">Drop your ECG plot here</span>{" "}
                    <span className="drop-slate-text">or click to browse</span>
                  </div>
                  <p className="dropzone-subtext">
                    Supports PNG, JPG, JPEG • Max 10MB
                  </p>
                </div>
              ) : (
                <div className="file-preview-card" onClick={(e) => e.stopPropagation()}>
                  {previewUrl && (
                    <img src={previewUrl} alt="ECG preview thumbnail" className="preview-thumb" />
                  )}
                  <div className="preview-details">
                    <div className="preview-name">{file.name}</div>
                    <div className="preview-meta">{formatBytes(file.size)} • Ready to digitize</div>
                  </div>
                  <button
                    type="button"
                    className="btn-clear-file"
                    onClick={handleClearFile}
                    title="Remove file"
                  >
                    <X size={18} weight="bold" />
                  </button>
                </div>
              )}
            </div>

            {/* Duration Input & Action Strip */}
            <div className="card-controls-row">
              <div className="duration-col">
                <div className="field-label-compact">
                  <Clock size={16} weight="bold" />
                  <span>Duration in Waveform Image</span>
                </div>
                <div className="duration-input-capsule">
                  <input
                    id="duration_input"
                    type="number"
                    min="5"
                    max="3600"
                    step="0.1"
                    placeholder="180"
                    value={duration}
                    onChange={(e) => setDuration(e.target.value)}
                    required
                  />
                  <span className="capsule-unit">seconds</span>
                </div>
                <div className="duration-quick-pills">
                  <button
                    type="button"
                    className={`quick-pill ${duration === "130" ? "active" : ""}`}
                    onClick={() => setDuration("130")}
                  >
                    130s
                  </button>
                  <button
                    type="button"
                    className={`quick-pill ${duration === "180" ? "active" : ""}`}
                    onClick={() => setDuration("180")}
                  >
                    180s
                  </button>
                  <button
                    type="button"
                    className={`quick-pill ${duration === "300" ? "active" : ""}`}
                    onClick={() => setDuration("300")}
                  >
                    300s
                  </button>
                </div>
              </div>

              <div className="submit-col">
                <button type="submit" className="btn-apple-eval" disabled={loading || !file}>
                  <Heart size={20} weight="fill" />
                  <span>{loading ? "Digitizing Waveform..." : "Digitize Waveform & Evaluate Risk"}</span>
                </button>
              </div>
            </div>

            {/* Dynamic Loading Bar */}
            {loading && (
              <div className="loading-state">
                <div className="loading-pulse-bar"></div>
                <div className="loading-text">
                  {loadingStage || "Processing waveform signal..."}
                </div>
              </div>
            )}
          </form>
        </section>

        {/* Result Modal Overlay */}
        {showModal && result && (
          <div className="result-overlay" role="dialog" aria-modal="true">
            <div className="result-modal-card">
              <div className="modal-header-row">
                <div className="modal-title-eyebrow">CardioPulse Diagnostic Summary</div>
                <button
                  type="button"
                  className="btn-modal-close"
                  onClick={() => setShowModal(false)}
                  title="Close summary"
                >
                  <X size={16} weight="bold" />
                </button>
              </div>

              {/* Status Banner */}
              {result.label === "Pre-arrest" ? (
                <div className="status-banner is-prearrest">
                  <div className="status-icon-wrap">
                    <WarningCircle size={28} weight="fill" />
                  </div>
                  <div>
                    <div className="status-text-headline">Pre-Arrest Signal Detected</div>
                    <div className="status-text-sub">
                      Model risk probability exceeds the calibrated clinical cutoff of {result.threshold_used}.
                    </div>
                  </div>
                </div>
              ) : (
                <div className="status-banner is-normal">
                  <div className="status-icon-wrap">
                    <CheckCircle size={28} weight="fill" />
                  </div>
                  <div>
                    <div className="status-text-headline">Normal Sinus Rhythm</div>
                    <div className="status-text-sub">
                      Heart Rate Variability metrics align with normal physiological baseline bounds.
                    </div>
                  </div>
                </div>
              )}

              {/* Risk Gauge */}
              <div className="gauge-card">
                <div className="gauge-header">
                  <span className="gauge-title">Pre-Arrest Probability Score</span>
                  <span className="gauge-value">{(result.probability * 100).toFixed(1)}%</span>
                </div>

                <div className="gauge-track">
                  <div
                    className="gauge-fill"
                    style={{ width: `${Math.min(result.probability * 100, 100)}%` }}
                  ></div>
                  <div
                    className="gauge-threshold-pin"
                    style={{ left: `${result.threshold_used * 100}%` }}
                  ></div>
                </div>

                <div className="gauge-legend">
                  <span>0% (Low Risk)</span>
                  <span>Cutoff: {(result.threshold_used * 100).toFixed(1)}%</span>
                  <span>100% (High Risk)</span>
                </div>
              </div>

              {/* 4-Bento Metrics */}
              <div className="metrics-bento">
                <div className="metric-tile">
                  <div className="metric-tile-header">
                    <span className="metric-tile-label">Confidence</span>
                    <Gauge size={16} weight="bold" className="metric-tile-icon" />
                  </div>
                  <div className="metric-tile-value">{(result.probability * 100).toFixed(1)}%</div>
                  <div className="metric-tile-sub">Model risk score</div>
                </div>

                <div className="metric-tile">
                  <div className="metric-tile-header">
                    <span className="metric-tile-label">Threshold</span>
                    <ShieldCheck size={16} weight="bold" className="metric-tile-icon" />
                  </div>
                  <div className="metric-tile-value">{result.threshold_used}</div>
                  <div className="metric-tile-sub">Decision boundary</div>
                </div>

                <div className="metric-tile">
                  <div className="metric-tile-header">
                    <span className="metric-tile-label">Heartbeats</span>
                    <Heartbeat size={16} weight="bold" className="metric-tile-icon" />
                  </div>
                  <div className="metric-tile-value">{result.beats_detected}</div>
                  <div className="metric-tile-sub">R-peaks detected</div>
                </div>

                <div className="metric-tile">
                  <div className="metric-tile-header">
                    <span className="metric-tile-label">Sampling Rate</span>
                    <Waveform size={16} weight="bold" className="metric-tile-icon" />
                  </div>
                  <div className="metric-tile-value">
                    {result.sampling_rate_used} <span style={{ fontSize: "0.8rem" }}>Hz</span>
                  </div>
                  <div className="metric-tile-sub">Reconstructed</div>
                </div>
              </div>

              {/* Sanity Plot Frame */}
              <div className="plot-container">
                <div className="plot-header">
                  <div className="plot-title">
                    <Waveform size={18} weight="bold" />
                    Digitized Signal &amp; Detected R-Peaks
                  </div>
                  <span className="plot-badge">NeuroKit2 Validated</span>
                </div>
                <img
                  src={result.plot_uri}
                  alt="Digitized signal plot with detected beats in red"
                  className="plot-img"
                />
                <p className="plot-disclaimer">
                  Sanity inspection trace: the slate line is the signal digitized from your image, and the red markers
                  indicate detected R-peaks. Confirm this matches the original trace before trusting predictions.
                </p>
              </div>

              {/* Actions */}
              <div className="modal-actions">
                <button
                  type="button"
                  className="btn-primary-reset"
                  onClick={() => setShowModal(false)}
                >
                  <ArrowsClockwise size={18} weight="bold" />
                  Analyze Another Waveform
                </button>
                <button
                  type="button"
                  className="btn-secondary-action"
                  onClick={() => window.print()}
                >
                  <Printer size={18} weight="bold" />
                  Print Report
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Scientific Pipeline Bento Grid */}
        <section className="pipeline-section anim-stagger-5">
          <div className="section-eyebrow">Scientific Architecture</div>
          <h2 className="section-heading">How the Predictive Pipeline Operates</h2>

          <div className="pipeline-grid">
            <div className="pipeline-card">
              <div className="step-icon-wrapper">
                <Waveform size={20} weight="bold" />
              </div>
              <div className="step-title">1. Trace Digitization</div>
              <div className="step-desc">
                High-contrast pixel column scanning converts 2D static images into continuous 1D voltage series.
              </div>
            </div>

            <div className="pipeline-card">
              <div className="step-icon-wrapper">
                <Heartbeat size={20} weight="bold" />
              </div>
              <div className="step-title">2. QRS Peak Detection</div>
              <div className="step-desc">
                NeuroKit2 QRS filtering pinpoints ventricular depolarizations, measuring exact RR interval dynamics.
              </div>
            </div>

            <div className="pipeline-card">
              <div className="step-icon-wrapper">
                <Pulse size={20} weight="bold" />
              </div>
              <div className="step-title">3. 90 HRV Biomarkers</div>
              <div className="step-desc">
                Extracts time-domain (RMSSD, SDNN), spectral (LF/HF ratio), and non-linear complexity features.
              </div>
            </div>

            <div className="pipeline-card">
              <div className="step-icon-wrapper">
                <Cpu size={20} weight="bold" />
              </div>
              <div className="step-title">4. Random Forest</div>
              <div className="step-desc">
                Ensemble decision trees evaluate the scaled vector against a 0.38 threshold to flag pre-arrest signals.
              </div>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer>
        <div className="footer-content">
          <p>CardioPulse AI • HRV Cardiac Arrest Early Warning System</p>
          <span className="footer-badge">
            Academic Research &amp; Engineering Prototype • Not an FDA/CE cleared diagnostic device
          </span>
        </div>
      </footer>
    </>
  );
}
