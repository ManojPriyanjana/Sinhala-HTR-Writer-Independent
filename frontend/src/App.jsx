import { useEffect, useMemo, useRef, useState } from "react";
import axios from "axios";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";
const HEALTH_CHECK_INTERVAL_MS = 3000;
const HEALTH_CHECK_TIMEOUT_MS = 1500;
const PREDICT_TIMEOUT_MS = 90000;
const FILE_PREVIEW_LIMIT = 12;
const THEME_STORAGE_KEY = "htr-theme";
const MIN_CROP_PERCENT = 10;

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

function buildCsv(rows) {
  const esc = (value) => `"${String(value ?? "").replace(/"/g, '""')}"`;
  const header = "line_id,file_name,confidence_percent,text,error,edited";
  const body = rows.map((row) => {
    const confidence = (clamp01(row.confidence) * 100).toFixed(1);
    return [
      esc(row.name),
      esc(row.sourceFileName),
      esc(confidence),
      esc(row.text),
      esc(row.error ? "true" : "false"),
      esc(row.edited ? "true" : "false"),
    ].join(",");
  });
  return [header, ...body].join("\n");
}

function downloadTextFile(filename, content, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

function removeKey(source, key) {
  const copy = { ...source };
  delete copy[key];
  return copy;
}

function getInitialTheme() {
  if (typeof window === "undefined") return "dark";
  const stored = window.localStorage.getItem(THEME_STORAGE_KEY);
  return stored === "light" || stored === "dark" ? stored : "dark";
}

function getLineCount(text) {
  if (!text) return 0;
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean).length;
}

function getWordCount(text) {
  if (!text) return 0;
  return text
    .trim()
    .split(/\s+/)
    .filter(Boolean).length;
}

function isParagraphLikeResult(result) {
  const lines = getLineCount(result?.text || "");
  const longText = (result?.text || "").length >= 220;
  return Number(result?.detectedLines || 0) > 1 || lines >= 3 || longText;
}

function clampNumber(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function normalizeCropMargins(left, right, top, bottom) {
  const maxMargin = 100 - MIN_CROP_PERCENT;
  let nextLeft = clampNumber(left, 0, maxMargin);
  let nextRight = clampNumber(right, 0, maxMargin);
  let nextTop = clampNumber(top, 0, maxMargin);
  let nextBottom = clampNumber(bottom, 0, maxMargin);

  if (nextLeft + nextRight > maxMargin) {
    nextRight = Math.max(0, maxMargin - nextLeft);
  }
  if (nextTop + nextBottom > maxMargin) {
    nextBottom = Math.max(0, maxMargin - nextTop);
  }

  return {
    left: nextLeft,
    right: nextRight,
    top: nextTop,
    bottom: nextBottom,
  };
}

export default function App() {
  const inputRef = useRef(null);
  const [theme, setTheme] = useState(getInitialTheme);
  const [apiOnline, setApiOnline] = useState(false);
  const [checkingApi, setCheckingApi] = useState(true);
  const [modelConnected, setModelConnected] = useState(false);
  const [modelMessage, setModelMessage] = useState("Please connect with model.");
  const [modelPath, setModelPath] = useState("");
  const [dragActive, setDragActive] = useState(false);
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState([]);
  const [errorMessage, setErrorMessage] = useState("");
  const [copiedKey, setCopiedKey] = useState("");
  const [copiedAll, setCopiedAll] = useState(false);
  const [resultQuery, setResultQuery] = useState("");
  const [sortMode, setSortMode] = useState("default");
  const [layoutMode, setLayoutMode] = useState("reader");
  const [textSizeMode, setTextSizeMode] = useState("comfortable");
  const [activePreviewKey, setActivePreviewKey] = useState("");
  const [isPreviewExpanded, setIsPreviewExpanded] = useState(false);
  const [cropDialog, setCropDialog] = useState({
    open: false,
    sourceKey: "",
    sourceName: "",
    previewUrl: "",
    left: 0,
    right: 0,
    top: 0,
    bottom: 0,
    imageWidth: 0,
    imageHeight: 0,
    processing: false,
  });
  const [editingByKey, setEditingByKey] = useState({});
  const [draftByKey, setDraftByKey] = useState({});

  useEffect(() => {
    window.localStorage.setItem(THEME_STORAGE_KEY, theme);
  }, [theme]);

  useEffect(() => {
    let active = true;

    async function checkHealth() {
      try {
        const response = await axios.get(`${API_BASE}/health`, { timeout: HEALTH_CHECK_TIMEOUT_MS });
        if (active) {
          const connected = Boolean(response?.data?.model_connected);
          setApiOnline(true);
          setModelConnected(connected);
          setModelMessage(
            response?.data?.message || (connected ? "Model connected." : "Please connect with model.")
          );
          setModelPath(response?.data?.model_path || "");
        }
      } catch {
        if (active) {
          setApiOnline(false);
          setModelConnected(false);
          setModelMessage("Please connect with model.");
          setModelPath("");
        }
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

  const fileCards = useMemo(
    () => files.map((file) => ({ file, key: fileId(file), previewUrl: URL.createObjectURL(file) })),
    [files]
  );

  useEffect(() => {
    return () => {
      fileCards.forEach((card) => URL.revokeObjectURL(card.previewUrl));
    };
  }, [fileCards]);

  useEffect(() => {
    if (!fileCards.length) {
      setActivePreviewKey("");
      setIsPreviewExpanded(false);
      return;
    }

    const exists = fileCards.some((card) => card.key === activePreviewKey);
    if (!exists) {
      setActivePreviewKey(fileCards[0].key);
    }
  }, [fileCards, activePreviewKey]);

  useEffect(() => {
    if (!isPreviewExpanded) return undefined;

    function handleKeyDown(event) {
      if (event.key === "Escape") {
        setIsPreviewExpanded(false);
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isPreviewExpanded]);

  useEffect(() => {
    if (!cropDialog.open) return undefined;

    function handleKeyDown(event) {
      if (event.key === "Escape" && !cropDialog.processing) {
        setCropDialog((prev) => ({ ...prev, open: false }));
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [cropDialog.open, cropDialog.processing]);

  const previewByKey = useMemo(() => {
    const map = new Map();
    fileCards.forEach((card) => map.set(card.key, card.previewUrl));
    return map;
  }, [fileCards]);

  const activePreviewCard = useMemo(() => {
    if (!fileCards.length) return null;
    return fileCards.find((card) => card.key === activePreviewKey) || fileCards[0];
  }, [fileCards, activePreviewKey]);

  const averageConfidence = useMemo(() => {
    if (!results.length) return 0;
    const sum = results.reduce((acc, item) => acc + clamp01(item.confidence), 0);
    return sum / results.length;
  }, [results]);

  const visibleResults = useMemo(() => {
    const q = resultQuery.trim().toLowerCase();
    let rows = results;

    if (q) {
      rows = rows.filter((row) => {
        return (
          row.name.toLowerCase().includes(q) ||
          row.text.toLowerCase().includes(q) ||
          row.sourceFileName.toLowerCase().includes(q)
        );
      });
    }

    if (sortMode === "confidence") {
      return [...rows].sort((a, b) => clamp01(b.confidence) - clamp01(a.confidence));
    }

    if (sortMode === "name") {
      return [...rows].sort((a, b) => a.name.localeCompare(b.name));
    }

    return rows;
  }, [results, resultQuery, sortMode]);

  const paragraphResultCount = useMemo(
    () => visibleResults.filter((result) => isParagraphLikeResult(result)).length,
    [visibleResults]
  );

  const cropMetrics = useMemo(() => {
    if (!cropDialog.imageWidth || !cropDialog.imageHeight) {
      return {
        x: 0,
        y: 0,
        width: 0,
        height: 0,
        widthPct: 100 - cropDialog.left - cropDialog.right,
        heightPct: 100 - cropDialog.top - cropDialog.bottom,
      };
    }

    const widthPct = Math.max(MIN_CROP_PERCENT, 100 - cropDialog.left - cropDialog.right);
    const heightPct = Math.max(MIN_CROP_PERCENT, 100 - cropDialog.top - cropDialog.bottom);
    const x = Math.round((cropDialog.left / 100) * cropDialog.imageWidth);
    const y = Math.round((cropDialog.top / 100) * cropDialog.imageHeight);
    const width = Math.max(1, Math.round((widthPct / 100) * cropDialog.imageWidth));
    const height = Math.max(1, Math.round((heightPct / 100) * cropDialog.imageHeight));

    return {
      x,
      y,
      width,
      height,
      widthPct,
      heightPct,
    };
  }, [cropDialog]);

  const readerTextClass =
    textSizeMode === "compact"
      ? "text-sm leading-7"
      : textSizeMode === "large"
        ? "text-[1.12rem] leading-9"
        : "text-base leading-8";

  function resetResultViews() {
    setResults([]);
    setErrorMessage("");
    setCopiedKey("");
    setCopiedAll(false);
    setResultQuery("");
    setEditingByKey({});
    setDraftByKey({});
  }

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

    resetResultViews();
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
    resetResultViews();
  }

  function clearAll() {
    setFiles([]);
    setActivePreviewKey("");
    setIsPreviewExpanded(false);
    setCropDialog((prev) => ({ ...prev, open: false, processing: false }));
    resetResultViews();
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

  function openCropDialog() {
    if (!activePreviewCard) return;
    setCropDialog({
      open: true,
      sourceKey: activePreviewCard.key,
      sourceName: activePreviewCard.file.name,
      previewUrl: activePreviewCard.previewUrl,
      left: 0,
      right: 0,
      top: 0,
      bottom: 0,
      imageWidth: 0,
      imageHeight: 0,
      processing: false,
    });
  }

  function updateCropMargin(edge, value) {
    const numeric = Number(value);
    setCropDialog((prev) => {
      const draft = {
        left: prev.left,
        right: prev.right,
        top: prev.top,
        bottom: prev.bottom,
      };
      draft[edge] = Number.isFinite(numeric) ? numeric : draft[edge];
      const normalized = normalizeCropMargins(draft.left, draft.right, draft.top, draft.bottom);
      return { ...prev, ...normalized };
    });
  }

  async function applyCropToFile() {
    if (!cropDialog.open || cropDialog.processing) return;

    const sourceFile = files.find((file) => fileId(file) === cropDialog.sourceKey);
    if (!sourceFile) {
      setCropDialog((prev) => ({ ...prev, open: false, processing: false }));
      return;
    }

    setCropDialog((prev) => ({ ...prev, processing: true }));

    try {
      const img = new Image();
      const loaded = new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = reject;
      });
      img.src = cropDialog.previewUrl;
      await loaded;

      const canvas = document.createElement("canvas");
      canvas.width = cropMetrics.width;
      canvas.height = cropMetrics.height;
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        throw new Error("Could not create crop canvas context.");
      }

      ctx.drawImage(
        img,
        cropMetrics.x,
        cropMetrics.y,
        cropMetrics.width,
        cropMetrics.height,
        0,
        0,
        cropMetrics.width,
        cropMetrics.height
      );

      const outputType =
        sourceFile.type && sourceFile.type.startsWith("image/") ? sourceFile.type : "image/jpeg";
      const blob = await new Promise((resolve, reject) => {
        canvas.toBlob(
          (value) => (value ? resolve(value) : reject(new Error("Image crop failed."))),
          outputType,
          0.95
        );
      });

      const extMatch = sourceFile.name.match(/\.[^.]+$/);
      const ext = extMatch ? extMatch[0] : outputType === "image/png" ? ".png" : ".jpg";
      const stem = sourceFile.name.replace(/\.[^.]+$/, "");
      const croppedFile = new File([blob], `${stem}-cropped${ext}`, {
        type: outputType,
        lastModified: Date.now(),
      });

      setFiles((prev) =>
        prev.map((file) => (fileId(file) === cropDialog.sourceKey ? croppedFile : file))
      );
      setActivePreviewKey(fileId(croppedFile));
      setCropDialog((prev) => ({ ...prev, open: false, processing: false }));
      resetResultViews();
    } catch (error) {
      console.error(error);
      setErrorMessage("Image crop failed. Please try again.");
      setCropDialog((prev) => ({ ...prev, processing: false }));
    }
  }

  async function handleRecognize() {
    if (!files.length || loading || !apiOnline || !modelConnected) return;

    setLoading(true);
    setResults([]);
    setErrorMessage("");
    setCopiedKey("");
    setCopiedAll(false);
    setEditingByKey({});
    setDraftByKey({});

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
        const sourceFile = files[index];
        const sourceKey = sourceFile ? fileId(sourceFile) : "";
        const sourceName = sourceFile?.name || "";
        const text = typeof item?.text === "string" ? item.text : "";
        const name = item?.line_id || sourceName || `line-${index + 1}`;
        return {
          key: `${name}-${index}`,
          name,
          sourceFileKey: sourceKey,
          sourceFileName: sourceName,
          text,
          originalText: text,
          edited: false,
          confidence: clamp01(item?.confidence),
          error: Boolean(item?.error),
          detectedLines: Number(item?.meta?.detected_lines || 0),
          sourceImageWidth: Number(item?.meta?.width || 0),
          sourceImageHeight: Number(item?.meta?.height || 0),
        };
      });

      setResults(mapped);
      if (!mapped.length) {
        setErrorMessage("Backend responded, but no predictions were returned.");
      }
    } catch (error) {
      console.error(error);
      const detail = error?.response?.data?.detail;
      setErrorMessage(typeof detail === "string" ? detail : "Prediction failed. Check backend logs and API URL.");
      setResults(
        files.map((file, index) => ({
          key: `${fileId(file)}-${index}`,
          name: file.name,
          sourceFileKey: fileId(file),
          sourceFileName: file.name,
          text: "",
          originalText: "",
          edited: false,
          confidence: 0,
          error: true,
          detectedLines: 0,
          sourceImageWidth: 0,
          sourceImageHeight: 0,
        }))
      );
    } finally {
      setLoading(false);
    }
  }

  function startEditing(row) {
    setEditingByKey((prev) => ({ ...prev, [row.key]: true }));
    setDraftByKey((prev) => ({ ...prev, [row.key]: row.text || "" }));
  }

  function cancelEditing(key) {
    setEditingByKey((prev) => removeKey(prev, key));
    setDraftByKey((prev) => removeKey(prev, key));
  }

  function saveEditing(key) {
    const nextText = draftByKey[key] ?? "";
    setResults((prev) =>
      prev.map((row) => {
        if (row.key !== key) return row;
        const baseline = row.originalText ?? "";
        return {
          ...row,
          text: nextText,
          edited: nextText !== baseline,
        };
      })
    );
    setEditingByKey((prev) => removeKey(prev, key));
    setDraftByKey((prev) => removeKey(prev, key));
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

  async function copyAllVisible() {
    if (!visibleResults.length || !navigator?.clipboard) return;

    const content = visibleResults
      .map((row, index) => {
        const confidence = (clamp01(row.confidence) * 100).toFixed(1);
        return `${index + 1}. ${row.name}\nConfidence: ${confidence}%\n${row.text || "(empty)"}`;
      })
      .join("\n\n");

    try {
      await navigator.clipboard.writeText(content);
      setCopiedAll(true);
      setTimeout(() => setCopiedAll(false), 1400);
    } catch (error) {
      console.error(error);
    }
  }

  function downloadJson() {
    if (!visibleResults.length) return;
    const payload = {
      generated_at: new Date().toISOString(),
      count: visibleResults.length,
      results: visibleResults,
    };
    downloadTextFile("htr-results.json", JSON.stringify(payload, null, 2), "application/json");
  }

  function downloadCsv() {
    if (!visibleResults.length) return;
    const csv = buildCsv(visibleResults);
    downloadTextFile("htr-results.csv", csv, "text/csv;charset=utf-8");
  }

  const statusLabel = checkingApi
    ? "Checking API"
    : !apiOnline
      ? "API offline"
      : modelConnected
        ? "API online + model connected"
        : "API online | model missing";
  const statusColor = checkingApi
    ? "bg-amber-300"
    : !apiOnline
      ? "bg-rose-400"
      : modelConnected
      ? "bg-emerald-400"
      : "bg-amber-300";

  return (
    <div className={`app-shell min-h-screen ${theme === "light" ? "theme-light" : "theme-dark"}`}>
      <main className="mx-auto max-w-[92rem] px-4 py-8 sm:px-6 lg:px-8">
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
                Upload line or full-page handwriting images and run recognition in one batch request.
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-3">
              <button
                onClick={() => setTheme((prev) => (prev === "dark" ? "light" : "dark"))}
                className="rounded-full border border-cyan-300/70 bg-cyan-300/15 px-4 py-2 text-sm font-medium text-cyan-100 transition hover:bg-cyan-300/25"
              >
                {theme === "dark" ? "Light mode" : "Dark mode"}
              </button>

              <div className="inline-flex items-center gap-3 rounded-full border border-slate-600/70 bg-slate-900/70 px-4 py-2 text-sm">
                <span
                  className={`h-2.5 w-2.5 rounded-full ${statusColor} ${checkingApi ? "animate-pulse" : ""}`}
                />
                <span className="text-slate-200">{statusLabel}</span>
              </div>
            </div>
          </div>

          <div className="mt-5 grid gap-3 sm:grid-cols-3">
            <div className="rounded-2xl border border-slate-700/70 bg-slate-900/45 px-4 py-3">
              <p className="text-xs uppercase tracking-wide text-slate-400">Selected files</p>
              <p className="mt-1 text-xl font-semibold text-white">{files.length}</p>
            </div>
            <div className="rounded-2xl border border-slate-700/70 bg-slate-900/45 px-4 py-3">
              <p className="text-xs uppercase tracking-wide text-slate-400">Upload size</p>
              <p className="mt-1 text-xl font-semibold text-white">{formatBytes(totalBytes)}</p>
            </div>
            <div className="rounded-2xl border border-slate-700/70 bg-slate-900/45 px-4 py-3">
              <p className="text-xs uppercase tracking-wide text-slate-400">Predictions</p>
              <p className="mt-1 text-xl font-semibold text-white">{results.length}</p>
            </div>
          </div>
        </section>

        <section className="mt-6 grid gap-6 lg:grid-cols-[1.35fr_1.05fr]">
          <div className="glass-card rise-in-delay rounded-3xl p-5 sm:p-6">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <h2 className="text-lg font-semibold text-white sm:text-xl">Upload Handwriting Images</h2>
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
                dragActive ? "border-cyan-300 bg-cyan-300/10" : "border-slate-500/60 bg-slate-900/35"
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
                  disabled={!files.length || loading || !apiOnline || !modelConnected}
                  className={`rounded-xl px-4 py-2 text-sm font-medium transition ${
                    !files.length || loading || !apiOnline || !modelConnected
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

              {apiOnline && !modelConnected && (
                <div className="mt-4 rounded-xl border border-amber-300/50 bg-amber-200/10 p-3 text-sm text-amber-100">
                  {modelMessage} Put your model file in <b>backend/models/</b> or set <b>MODEL_PATH</b>.
                </div>
              )}

              {apiOnline && modelConnected && modelPath && (
                <p className="mt-3 text-xs text-slate-300">
                  Active model: <span className="font-medium">{modelPath}</span>
                </p>
              )}

              {loading && (
                <div className="mt-4 overflow-hidden rounded-lg bg-slate-900/50">
                  <div className="scan-loader h-1.5 w-full rounded-lg bg-cyan-300/70" />
                </div>
              )}

              {activePreviewCard && (
                <div className="mt-5">
                  <div className="flex items-center justify-between gap-2">
                    <p className="text-xs font-semibold uppercase tracking-wider text-slate-300">Active preview</p>
                    <p className="truncate text-xs text-slate-400">{activePreviewCard.file.name}</p>
                  </div>
                  <div className="mt-2 overflow-hidden rounded-2xl border border-slate-700/80 bg-slate-900/55">
                    <img
                      src={activePreviewCard.previewUrl}
                      alt={activePreviewCard.file.name}
                      onClick={() => setIsPreviewExpanded(true)}
                      className="h-[26rem] w-full cursor-zoom-in bg-slate-900/60 p-2 object-contain sm:h-[34rem] lg:h-[40rem]"
                      loading="lazy"
                    />
                  </div>
                  <div className="mt-2 flex items-center justify-between gap-3">
                    <p className="text-xs text-slate-400">{formatBytes(activePreviewCard.file.size)}</p>
                    <div className="flex items-center gap-2">
                      <button
                        onClick={openCropDialog}
                        className="rounded-lg border border-emerald-300/70 bg-emerald-300/10 px-2.5 py-1 text-xs text-emerald-100 transition hover:bg-emerald-300/20"
                      >
                        Crop image
                      </button>
                      <button
                        onClick={() => setIsPreviewExpanded(true)}
                        className="rounded-lg border border-cyan-300/70 bg-cyan-300/10 px-2.5 py-1 text-xs text-cyan-100 transition hover:bg-cyan-300/20"
                      >
                        Full view
                      </button>
                    </div>
                  </div>
                </div>
              )}

              {fileCards.length > 0 && (
                <div className="mt-5">
                  <p className="text-xs font-semibold uppercase tracking-wider text-slate-300">Selected files</p>
                  <ul className="mt-3 grid gap-2 sm:grid-cols-2">
                    {fileCards.slice(0, FILE_PREVIEW_LIMIT).map((card) => (
                      <li
                        key={card.key}
                        onClick={() => setActivePreviewKey(card.key)}
                        className={`flex cursor-pointer items-center gap-3 rounded-xl border bg-slate-900/50 px-3 py-2 transition ${
                          activePreviewCard?.key === card.key
                            ? "border-cyan-300/70 ring-1 ring-cyan-300/50"
                            : "border-slate-700/80 hover:border-cyan-300/45"
                        }`}
                      >
                        <img
                          src={card.previewUrl}
                          alt={card.file.name}
                          className="h-16 w-16 flex-none rounded-md border border-slate-700 bg-slate-900/50 p-1 object-contain"
                          loading="lazy"
                        />
                        <div className="min-w-0 flex-1">
                          <p className="truncate text-sm text-slate-100">{card.file.name}</p>
                          <p className="text-xs text-slate-400">{formatBytes(card.file.size)}</p>
                        </div>
                        <button
                          onClick={(event) => {
                            event.stopPropagation();
                            handleRemoveFile(card.key);
                          }}
                          className="rounded-lg border border-rose-300/50 px-2 py-1 text-xs text-rose-200 transition hover:bg-rose-400/10"
                        >
                          Remove
                        </button>
                      </li>
                    ))}
                  </ul>
                  {fileCards.length > FILE_PREVIEW_LIMIT && (
                    <p className="mt-2 text-xs text-slate-400">Showing first {FILE_PREVIEW_LIMIT} files.</p>
                  )}
                </div>
              )}
            </div>
          </div>

          <div className="glass-card rise-in-delay rounded-3xl p-5 sm:p-6">
            <div className="flex items-start justify-between gap-3">
              <div>
                <h2 className="text-lg font-semibold text-white sm:text-xl">Results</h2>
                {results.length > 0 && (
                  <p className="text-xs text-slate-300 sm:text-sm">
                    Avg confidence: {(averageConfidence * 100).toFixed(1)}%
                  </p>
                )}
                {paragraphResultCount > 0 && (
                  <p className="mt-1 text-xs text-cyan-200">
                    Reader-optimized results: {paragraphResultCount}
                  </p>
                )}
              </div>
              <div className="flex flex-wrap gap-2">
                <button
                  onClick={copyAllVisible}
                  disabled={!visibleResults.length}
                  className={`rounded-lg border px-2.5 py-1 text-xs transition ${
                    visibleResults.length
                      ? "border-cyan-300/70 text-cyan-100 hover:bg-cyan-300/15"
                      : "cursor-not-allowed border-slate-700 text-slate-500"
                  }`}
                >
                  {copiedAll ? "Copied all" : "Copy all"}
                </button>
                <button
                  onClick={downloadCsv}
                  disabled={!visibleResults.length}
                  className={`rounded-lg border px-2.5 py-1 text-xs transition ${
                    visibleResults.length
                      ? "border-emerald-300/60 text-emerald-100 hover:bg-emerald-300/15"
                      : "cursor-not-allowed border-slate-700 text-slate-500"
                  }`}
                >
                  CSV
                </button>
                <button
                  onClick={downloadJson}
                  disabled={!visibleResults.length}
                  className={`rounded-lg border px-2.5 py-1 text-xs transition ${
                    visibleResults.length
                      ? "border-emerald-300/60 text-emerald-100 hover:bg-emerald-300/15"
                      : "cursor-not-allowed border-slate-700 text-slate-500"
                  }`}
                >
                  JSON
                </button>
              </div>
            </div>

            {results.length > 0 && (
              <div className="mt-4 grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
                <input
                  value={resultQuery}
                  onChange={(event) => setResultQuery(event.target.value)}
                  placeholder="Filter by file, id, or text"
                  className="w-full rounded-xl border border-slate-600/80 bg-slate-900/55 px-3 py-2 text-sm text-white outline-none ring-cyan-300 placeholder:text-slate-400 focus:ring-1"
                />
                <select
                  value={sortMode}
                  onChange={(event) => setSortMode(event.target.value)}
                  className="w-full rounded-xl border border-slate-600/80 bg-slate-900/55 px-3 py-2 text-sm text-white outline-none ring-cyan-300 focus:ring-1"
                >
                  <option value="default">Sort: Upload order</option>
                  <option value="confidence">Sort: Confidence</option>
                  <option value="name">Sort: Name</option>
                </select>
                <select
                  value={layoutMode}
                  onChange={(event) => setLayoutMode(event.target.value)}
                  className="w-full rounded-xl border border-slate-600/80 bg-slate-900/55 px-3 py-2 text-sm text-white outline-none ring-cyan-300 focus:ring-1"
                >
                  <option value="reader">Layout: Reader</option>
                  <option value="compact">Layout: Compact</option>
                </select>
                <select
                  value={textSizeMode}
                  onChange={(event) => setTextSizeMode(event.target.value)}
                  className="w-full rounded-xl border border-slate-600/80 bg-slate-900/55 px-3 py-2 text-sm text-white outline-none ring-cyan-300 focus:ring-1"
                >
                  <option value="comfortable">Text: Comfortable</option>
                  <option value="large">Text: Large</option>
                  <option value="compact">Text: Compact</option>
                </select>
              </div>
            )}

            {errorMessage && (
              <div className="mt-4 rounded-xl border border-amber-300/50 bg-amber-200/10 p-3 text-sm text-amber-100">
                {errorMessage}
              </div>
            )}

            {results.length === 0 ? (
              <p className="mt-6 text-sm text-slate-300">No predictions yet. Upload images and run recognition.</p>
            ) : visibleResults.length === 0 ? (
              <p className="mt-6 text-sm text-slate-300">No results match your filter.</p>
            ) : (
              <div className="result-scroll mt-4 space-y-3">
                {visibleResults.map((result) => {
                  const isEditing = Boolean(editingByKey[result.key]);
                  const draftText = draftByKey[result.key] ?? result.text;
                  const paragraphLike = isParagraphLikeResult(result);
                  const lineCount = getLineCount(result.text);
                  const wordCount = getWordCount(result.text);
                  const transcriptClass =
                    layoutMode === "reader" && paragraphLike
                      ? `sinhala-reader ${readerTextClass}`
                      : "text-base leading-7";

                  return (
                    <article
                      key={result.key}
                      className={`rounded-2xl border border-slate-700/80 bg-slate-900/45 p-4 ${
                        layoutMode === "reader" && paragraphLike ? "result-card-reader" : ""
                      }`}
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div className="min-w-0">
                          <p className="truncate text-xs uppercase tracking-wide text-slate-400">{result.name}</p>
                          {result.sourceFileName && (
                            <p className="truncate text-xs text-slate-500">{result.sourceFileName}</p>
                          )}
                          <div className="mt-1 flex flex-wrap gap-1.5">
                            {lineCount > 0 && (
                              <span className="rounded-full border border-slate-500/70 px-2 py-0.5 text-[11px] text-slate-300">
                                {lineCount} lines
                              </span>
                            )}
                            {wordCount > 0 && (
                              <span className="rounded-full border border-slate-500/70 px-2 py-0.5 text-[11px] text-slate-300">
                                {wordCount} words
                              </span>
                            )}
                            {result.detectedLines > 1 && (
                              <span className="rounded-full border border-cyan-300/70 bg-cyan-300/10 px-2 py-0.5 text-[11px] text-cyan-100">
                                Segmented: {result.detectedLines}
                              </span>
                            )}
                          </div>
                        </div>

                        <div className="flex flex-wrap items-center gap-2">
                          {result.edited && (
                            <span className="rounded-full border border-amber-300/50 bg-amber-300/15 px-2 py-0.5 text-[11px] text-amber-100">
                              Edited
                            </span>
                          )}

                          {isEditing ? (
                            <>
                              <button
                                onClick={() => saveEditing(result.key)}
                                className="rounded-lg border border-emerald-300/70 px-2 py-1 text-xs text-emerald-100 transition hover:bg-emerald-300/15"
                              >
                                Save
                              </button>
                              <button
                                onClick={() => cancelEditing(result.key)}
                                className="rounded-lg border border-slate-500/80 px-2 py-1 text-xs text-slate-200 transition hover:bg-slate-700/40"
                              >
                                Cancel
                              </button>
                            </>
                          ) : (
                            <button
                              onClick={() => startEditing(result)}
                              className="rounded-lg border border-cyan-300/70 px-2 py-1 text-xs text-cyan-100 transition hover:bg-cyan-300/15"
                            >
                              Edit
                            </button>
                          )}

                          <button
                            onClick={() => copyText(result.text, result.key)}
                            disabled={!result.text || isEditing}
                            className={`rounded-lg border px-2 py-1 text-xs transition ${
                              result.text && !isEditing
                                ? "border-cyan-300/70 text-cyan-100 hover:bg-cyan-300/15"
                                : "cursor-not-allowed border-slate-700 text-slate-500"
                            }`}
                          >
                            {copiedKey === result.key ? "Copied" : "Copy"}
                          </button>
                        </div>
                      </div>

                      <div className={`mt-3 flex items-start gap-3 ${layoutMode === "reader" && paragraphLike ? "md:gap-4" : ""}`}>
                        {result.sourceFileKey && previewByKey.get(result.sourceFileKey) && (
                          <img
                            src={previewByKey.get(result.sourceFileKey)}
                            alt={result.sourceFileName || result.name}
                            className={`rounded-lg border border-slate-700 bg-slate-900/50 p-1 object-contain ${
                              layoutMode === "reader" && paragraphLike ? "h-24 w-24" : "h-20 w-20"
                            }`}
                            loading="lazy"
                          />
                        )}

                        <div className="min-w-0 flex-1">
                          {isEditing ? (
                            <textarea
                              value={draftText}
                              onChange={(event) =>
                                setDraftByKey((prev) => ({ ...prev, [result.key]: event.target.value }))
                              }
                              rows={3}
                              className="w-full resize-y rounded-lg border border-slate-600/80 bg-slate-900/55 px-3 py-2 text-sm text-white outline-none ring-cyan-300 placeholder:text-slate-400 focus:ring-1"
                              placeholder="Edit recognized text"
                            />
                          ) : (
                            <p className={`min-w-0 whitespace-pre-wrap break-words text-white ${transcriptClass}`}>
                              {result.text || <span className="text-slate-500">-</span>}
                            </p>
                          )}
                        </div>
                      </div>

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
                  );
                })}
              </div>
            )}
          </div>
        </section>

        <p className="mt-6 text-center text-xs text-slate-400">FastAPI + React (Vite) | Sinhala HTR prototype</p>
      </main>

      {isPreviewExpanded && activePreviewCard && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 p-3 backdrop-blur-sm"
          onClick={() => setIsPreviewExpanded(false)}
        >
          <div
            className="relative max-h-[94vh] w-[min(96vw,96rem)] overflow-hidden rounded-2xl border border-slate-500/70 bg-slate-900/90 p-3"
            onClick={(event) => event.stopPropagation()}
          >
            <button
              onClick={() => setIsPreviewExpanded(false)}
              className="absolute right-3 top-3 rounded-lg border border-slate-500/70 bg-slate-900/70 px-2.5 py-1 text-xs text-slate-200 transition hover:bg-slate-800"
            >
              Close
            </button>
            <p className="mb-2 truncate pr-16 text-xs text-slate-300">{activePreviewCard.file.name}</p>
            <img
              src={activePreviewCard.previewUrl}
              alt={activePreviewCard.file.name}
              className="max-h-[86vh] w-full rounded-xl bg-slate-900/70 object-contain"
            />
          </div>
        </div>
      )}

      {cropDialog.open && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 p-3 backdrop-blur-sm"
          onClick={() => {
            if (!cropDialog.processing) {
              setCropDialog((prev) => ({ ...prev, open: false }));
            }
          }}
        >
          <div
            className="relative max-h-[96vh] w-[min(96vw,88rem)] overflow-auto rounded-2xl border border-slate-500/70 bg-slate-900/95 p-4"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.16em] text-cyan-200">Crop Image</p>
                <p className="mt-1 max-w-[60ch] text-xs text-slate-300">
                  Remove unwanted borders or regions before recognition. Keep only handwriting area.
                </p>
                <p className="mt-1 truncate text-xs text-slate-400">{cropDialog.sourceName}</p>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() =>
                    setCropDialog((prev) => ({
                      ...prev,
                      ...normalizeCropMargins(0, 0, 0, 0),
                    }))
                  }
                  disabled={cropDialog.processing}
                  className="rounded-lg border border-slate-500/80 px-2.5 py-1 text-xs text-slate-200 transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  Reset
                </button>
                <button
                  onClick={() =>
                    !cropDialog.processing &&
                    setCropDialog((prev) => ({ ...prev, open: false, processing: false }))
                  }
                  disabled={cropDialog.processing}
                  className="rounded-lg border border-slate-500/80 px-2.5 py-1 text-xs text-slate-200 transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  Cancel
                </button>
                <button
                  onClick={applyCropToFile}
                  disabled={cropDialog.processing}
                  className="rounded-lg border border-emerald-300/70 bg-emerald-300/10 px-3 py-1 text-xs text-emerald-100 transition hover:bg-emerald-300/20 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {cropDialog.processing ? "Applying..." : "Apply crop"}
                </button>
              </div>
            </div>

            <div className="mt-4 grid gap-4 lg:grid-cols-[1.45fr_1fr]">
              <div className="overflow-hidden rounded-xl border border-slate-700/80 bg-slate-900/70 p-2">
                <div className="relative mx-auto w-fit">
                  <img
                    src={cropDialog.previewUrl}
                    alt={cropDialog.sourceName}
                    onLoad={(event) => {
                      const { naturalWidth, naturalHeight } = event.currentTarget;
                      setCropDialog((prev) => ({
                        ...prev,
                        imageWidth: naturalWidth || prev.imageWidth,
                        imageHeight: naturalHeight || prev.imageHeight,
                      }));
                    }}
                    className="block max-h-[68vh] w-auto max-w-full rounded-lg object-contain"
                  />

                  <div
                    className="pointer-events-none absolute left-0 top-0 bg-black/45"
                    style={{ width: "100%", height: `${cropDialog.top}%` }}
                  />
                  <div
                    className="pointer-events-none absolute bottom-0 left-0 bg-black/45"
                    style={{ width: "100%", height: `${cropDialog.bottom}%` }}
                  />
                  <div
                    className="pointer-events-none absolute bg-black/45"
                    style={{
                      left: 0,
                      top: `${cropDialog.top}%`,
                      width: `${cropDialog.left}%`,
                      height: `${Math.max(0, 100 - cropDialog.top - cropDialog.bottom)}%`,
                    }}
                  />
                  <div
                    className="pointer-events-none absolute bg-black/45"
                    style={{
                      right: 0,
                      top: `${cropDialog.top}%`,
                      width: `${cropDialog.right}%`,
                      height: `${Math.max(0, 100 - cropDialog.top - cropDialog.bottom)}%`,
                    }}
                  />
                  <div
                    className="pointer-events-none absolute rounded-lg border-2 border-cyan-300/90 shadow-[0_0_0_1px_rgba(15,23,42,0.75)]"
                    style={{
                      left: `${cropDialog.left}%`,
                      top: `${cropDialog.top}%`,
                      width: `${Math.max(MIN_CROP_PERCENT, 100 - cropDialog.left - cropDialog.right)}%`,
                      height: `${Math.max(MIN_CROP_PERCENT, 100 - cropDialog.top - cropDialog.bottom)}%`,
                    }}
                  />
                </div>
              </div>

              <div className="rounded-xl border border-slate-700/80 bg-slate-900/70 p-3">
                <p className="text-xs font-semibold uppercase tracking-wide text-slate-300">Crop Controls</p>
                <div className="mt-3 space-y-3">
                  <label className="block text-xs text-slate-300">
                    Left margin: {cropDialog.left.toFixed(0)}%
                    <input
                      type="range"
                      min="0"
                      max={100 - MIN_CROP_PERCENT}
                      value={cropDialog.left}
                      onChange={(event) => updateCropMargin("left", event.target.value)}
                      className="mt-1 w-full accent-cyan-300"
                    />
                  </label>
                  <label className="block text-xs text-slate-300">
                    Right margin: {cropDialog.right.toFixed(0)}%
                    <input
                      type="range"
                      min="0"
                      max={100 - MIN_CROP_PERCENT}
                      value={cropDialog.right}
                      onChange={(event) => updateCropMargin("right", event.target.value)}
                      className="mt-1 w-full accent-cyan-300"
                    />
                  </label>
                  <label className="block text-xs text-slate-300">
                    Top margin: {cropDialog.top.toFixed(0)}%
                    <input
                      type="range"
                      min="0"
                      max={100 - MIN_CROP_PERCENT}
                      value={cropDialog.top}
                      onChange={(event) => updateCropMargin("top", event.target.value)}
                      className="mt-1 w-full accent-cyan-300"
                    />
                  </label>
                  <label className="block text-xs text-slate-300">
                    Bottom margin: {cropDialog.bottom.toFixed(0)}%
                    <input
                      type="range"
                      min="0"
                      max={100 - MIN_CROP_PERCENT}
                      value={cropDialog.bottom}
                      onChange={(event) => updateCropMargin("bottom", event.target.value)}
                      className="mt-1 w-full accent-cyan-300"
                    />
                  </label>
                </div>

                <div className="mt-4 rounded-lg border border-slate-600/70 bg-slate-900/60 p-2 text-xs text-slate-300">
                  <p>
                    Crop area: {Math.round(cropMetrics.widthPct)}% x {Math.round(cropMetrics.heightPct)}%
                  </p>
                  <p className="mt-1">
                    Output size: {cropMetrics.width} x {cropMetrics.height}px
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
