import { useEffect, useMemo, useRef, useState } from "react";
import axios from "axios";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";
const HEALTH_CHECK_INTERVAL_MS = 3000;
const HEALTH_CHECK_TIMEOUT_MS = 1500;
const PREDICT_TIMEOUT_MS = 90000;
const FILE_PREVIEW_LIMIT = 10;

function clamp01(value) {
  return Math.max(0, Math.min(1, Number(value) || 0));
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function fileId(file) {
  return `${file.name}:${file.size}:${file.lastModified}`;
}

export default function App() {
  const inputRef = useRef(null);
  const [apiOnline, setApiOnline] = useState(false);
  const [checkingApi, setCheckingApi] = useState(true);
  const [dragActive, setDragActive] = useState(false);
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState([]);
  const [errorMessage, setErrorMessage] = useState("");
  const [copiedKey, setCopiedKey] = useState("");

  useEffect(() => {
    let active = true;

    async function checkHealth() {
      try {
        await axios.get(`${API_BASE}/health`, { timeout: HEALTH_CHECK_TIMEOUT_MS });
        if (active) setApiOnline(true);
      } catch {
        if (active) setApiOnline(false);
      } finally {
        if (active) setCheckingApi(false);
      }
    }

    checkHealth();
    const intervalId = setInterval(checkHealth, HEALTH_CHECK_INTERVAL_MS);
    return () => {
      active = false;
      clearInterval(intervalId);
    };
  }, []);

  const totalBytes = useMemo(() => files.reduce((sum, file) => sum + file.size, 0), [files]);

  const averageConfidence = useMemo(() => {
    if (!results.length) return 0;
    const sum = results.reduce((acc, item) => acc + clamp01(item.confidence), 0);
    return sum / results.length;
  }, [results]);

  function appendFiles(nextFiles) {
    const imageFiles = nextFiles.filter(
      (file) => file.type.startsWith("image/") || /\.(png|jpg|jpeg)$/i.test(file.name)
    );

    setFiles((prev) => {
      const existing = new Set(prev.map(fileId));
      const merged = [...prev];
      for (const file of imageFiles) {
        const key = fileId(file);
        if (!existing.has(key)) {
          existing.add(key);
          merged.push(file);
        }
      }
      return merged;
    });

    setResults([]);
    setErrorMessage("");
  }

  function openFilePicker() {
    inputRef.current?.click();
  }

  function handleChoose(event) {
    const picked = Array.from(event.target.files || []);
    appendFiles(picked);
    if (inputRef.current) inputRef.current.value = "";
  }

  function handleRemoveFile(key) {
    setFiles((prev) => prev.filter((file) => fileId(file) !== key));
    setResults([]);
  }

  function clearAll() {
    setFiles([]);
    setResults([]);
    setErrorMessage("");
    setCopiedKey("");
    if (inputRef.current) inputRef.current.value = "";
  }

  function handleDragEnter(event) {
    event.preventDefault();
    setDragActive(true);
  }

  function handleDragOver(event) {
    event.preventDefault();
    setDragActive(true);
  }

  function handleDragLeave(event) {
    event.preventDefault();
    if (!event.currentTarget.contains(event.relatedTarget)) {
      setDragActive(false);
    }
  }

  function handleDrop(event) {
    event.preventDefault();
    setDragActive(false);
    const dropped = Array.from(event.dataTransfer.files || []);
    appendFiles(dropped);
  }

  async function handleRecognize() {
    if (!files.length || loading || !apiOnline) return;

    setLoading(true);
    setResults([]);
    setErrorMessage("");
    setCopiedKey("");

    try {
      const formData = new FormData();
      for (const file of files) {
        formData.append("files", file, file.name);
      }

      const response = await axios.post(`${API_BASE}/predict`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: PREDICT_TIMEOUT_MS,
      });

      const lines = Array.isArray(response?.data?.lines) ? response.data.lines : [];
      const mapped = lines.map((item, index) => {
        const name = item?.line_id || files[index]?.name || `line-${index + 1}`;
        return {
          key: `${name}-${index}`,
          name,
          text: typeof item?.text === "string" ? item.text : "",
          confidence: clamp01(item?.confidence),
          error: Boolean(item?.error),
        };
      });

      setResults(mapped);
      if (!mapped.length) {
        setErrorMessage("Backend responded, but no predictions were returned.");
      }
    } catch (error) {
      console.error(error);
      setErrorMessage("Prediction failed. Check backend logs and API URL.");
      setResults(
        files.map((file, index) => ({
          key: `${fileId(file)}-${index}`,
          name: file.name,
          text: "",
          confidence: 0,
          error: true,
        }))
      );
    } finally {
      setLoading(false);
    }
  }

  async function copyText(text, key) {
    if (!text || !navigator?.clipboard) return;
    try {
      await navigator.clipboard.writeText(text);
      setCopiedKey(key);
      setTimeout(() => setCopiedKey(""), 1200);
    } catch (error) {
      console.error(error);
    }
  }

  const statusLabel = checkingApi ? "Checking API" : apiOnline ? "API online" : "API offline";
  const statusColor = checkingApi
    ? "bg-amber-300"
    : apiOnline
      ? "bg-emerald-400"
      : "bg-rose-400";

  return (
    <div className="min-h-screen">
      <main className="mx-auto max-w-6xl px-4 py-8 sm:px-6 lg:px-8">
        <section className="glass-card rise-in rounded-3xl p-6 sm:p-8">
          <div className="flex flex-col gap-5 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-200">
                Writer-Independent HTR
              </p>
              <h1 className="mt-2 text-3xl font-semibold text-white sm:text-4xl">
                Sinhala Handwriting Demo
              </h1>
              <p className="mt-3 max-w-2xl text-sm text-slate-300 sm:text-base">
                Upload segmented line images and run recognition in one batch request.
              </p>
            </div>

            <div className="inline-flex items-center gap-3 rounded-full border border-slate-600/70 bg-slate-900/70 px-4 py-2 text-sm">
              <span
                className={`h-2.5 w-2.5 rounded-full ${statusColor} ${
                  checkingApi ? "animate-pulse" : ""
                }`}
              />
              <span className="text-slate-200">{statusLabel}</span>
            </div>
          </div>
        </section>

        <section className="mt-6 grid gap-6 lg:grid-cols-[1.2fr_1fr]">
          <div className="glass-card rise-in-delay rounded-3xl p-5 sm:p-6">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <h2 className="text-lg font-semibold text-white sm:text-xl">Upload Line Images</h2>
              <p className="text-xs text-slate-300 sm:text-sm">
                {files.length ? `${files.length} files | ${formatBytes(totalBytes)}` : "No files selected"}
              </p>
            </div>

            <input
              ref={inputRef}
              type="file"
              multiple
              accept="image/png,image/jpeg,image/jpg"
              onChange={handleChoose}
              className="hidden"
            />

            <div
              onDragEnter={handleDragEnter}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              className={`mt-4 rounded-2xl border-2 border-dashed p-5 transition sm:p-6 ${
                dragActive
                  ? "border-cyan-300 bg-cyan-300/10"
                  : "border-slate-500/60 bg-slate-900/35"
              }`}
            >
              <p className="text-sm text-slate-200 sm:text-base">Drop images here or browse from disk.</p>

              <div className="mt-4 flex flex-wrap gap-3">
                <button
                  onClick={openFilePicker}
                  className="rounded-xl border border-cyan-300/70 bg-cyan-300/15 px-4 py-2 text-sm font-medium text-cyan-100 transition hover:bg-cyan-300/25"
                >
                  Choose files
                </button>

                <button
                  onClick={handleRecognize}
                  disabled={!files.length || loading || !apiOnline}
                  className={`rounded-xl px-4 py-2 text-sm font-medium transition ${
                    !files.length || loading || !apiOnline
                      ? "cursor-not-allowed bg-slate-700/50 text-slate-400"
                      : "bg-emerald-500/90 text-emerald-950 hover:bg-emerald-400"
                  }`}
                >
                  {loading ? "Recognizing..." : "Recognize"}
                </button>

                <button
                  onClick={clearAll}
                  disabled={!files.length && !results.length}
                  className={`rounded-xl border px-4 py-2 text-sm font-medium transition ${
                    !files.length && !results.length
                      ? "cursor-not-allowed border-slate-700/50 text-slate-500"
                      : "border-slate-500/80 text-slate-200 hover:bg-slate-700/40"
                  }`}
                >
                  Clear
                </button>
              </div>

              {files.length > 0 && (
                <div className="mt-5">
                  <p className="text-xs font-semibold uppercase tracking-wider text-slate-300">
                    Selected files
                  </p>
                  <ul className="mt-3 grid gap-2 sm:grid-cols-2">
                    {files.slice(0, FILE_PREVIEW_LIMIT).map((file) => {
                      const key = fileId(file);
                      return (
                        <li
                          key={key}
                          className="flex items-center justify-between rounded-xl border border-slate-700/80 bg-slate-900/50 px-3 py-2"
                        >
                          <div className="min-w-0">
                            <p className="truncate text-sm text-slate-100">{file.name}</p>
                            <p className="text-xs text-slate-400">{formatBytes(file.size)}</p>
                          </div>
                          <button
                            onClick={() => handleRemoveFile(key)}
                            className="ml-3 rounded-lg border border-rose-300/50 px-2 py-1 text-xs text-rose-200 transition hover:bg-rose-400/10"
                          >
                            Remove
                          </button>
                        </li>
                      );
                    })}
                  </ul>
                  {files.length > FILE_PREVIEW_LIMIT && (
                    <p className="mt-2 text-xs text-slate-400">
                      Showing first {FILE_PREVIEW_LIMIT} files.
                    </p>
                  )}
                </div>
              )}
            </div>
          </div>

          <div className="glass-card rise-in-delay rounded-3xl p-5 sm:p-6">
            <div className="flex items-center justify-between gap-3">
              <h2 className="text-lg font-semibold text-white sm:text-xl">Results</h2>
              {results.length > 0 && (
                <p className="text-xs text-slate-300 sm:text-sm">
                  Avg confidence: {(averageConfidence * 100).toFixed(1)}%
                </p>
              )}
            </div>

            {errorMessage && (
              <div className="mt-4 rounded-xl border border-amber-300/50 bg-amber-200/10 p-3 text-sm text-amber-100">
                {errorMessage}
              </div>
            )}

            {results.length === 0 ? (
              <p className="mt-6 text-sm text-slate-300">
                No predictions yet. Upload images and run recognition.
              </p>
            ) : (
              <div className="mt-4 space-y-3">
                {results.map((result) => (
                  <article
                    key={result.key}
                    className="rounded-2xl border border-slate-700/80 bg-slate-900/45 p-4"
                  >
                    <div className="flex items-center justify-between gap-3">
                      <p className="truncate text-xs uppercase tracking-wide text-slate-400">
                        {result.name}
                      </p>
                      <button
                        onClick={() => copyText(result.text, result.key)}
                        disabled={!result.text}
                        className={`rounded-lg border px-2 py-1 text-xs transition ${
                          result.text
                            ? "border-cyan-300/70 text-cyan-100 hover:bg-cyan-300/15"
                            : "cursor-not-allowed border-slate-700 text-slate-500"
                        }`}
                      >
                        {copiedKey === result.key ? "Copied" : "Copy text"}
                      </button>
                    </div>

                    <p className="mt-2 whitespace-pre-wrap break-words text-base text-white">
                      {result.text || <span className="text-slate-500">-</span>}
                    </p>

                    <div className="mt-3">
                      <div className="h-2 w-full overflow-hidden rounded-full bg-slate-700/70">
                        <div
                          className={`confidence-fill h-full ${result.error ? "bg-rose-400" : "bg-cyan-300"}`}
                          style={{ width: `${clamp01(result.confidence) * 100}%` }}
                        />
                      </div>
                      <div className="mt-1 flex items-center justify-between text-xs text-slate-400">
                        <span>Confidence {(clamp01(result.confidence) * 100).toFixed(1)}%</span>
                        {result.error && <span className="text-rose-300">Invalid image</span>}
                      </div>
                    </div>
                  </article>
                ))}
              </div>
            )}
          </div>
        </section>

        <p className="mt-6 text-center text-xs text-slate-400">
          FastAPI + React (Vite) | Sinhala HTR prototype
        </p>
      </main>
    </div>
  );
}
