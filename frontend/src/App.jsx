import React, { useEffect, useMemo, useRef, useState } from "react";
import axios from "axios";

const API_BASE = "http://localhost:8000";

export default function App() {
  const inputRef = useRef(null);

  const [apiOnline, setApiOnline] = useState(false);
  const [files, setFiles] = useState([]); // File[]
  const [loading, setLoading] = useState(false);

  // results: [{ name, text, confidence }]
  const [results, setResults] = useState([]);

  // -------------------- API Health --------------------
  useEffect(() => {
    let alive = true;

    async function check() {
      try {
        await axios.get(`${API_BASE}/health`, { timeout: 1200 });
        if (alive) setApiOnline(true);
      } catch {
        if (alive) setApiOnline(false);
      }
    }

    check();
    const id = setInterval(check, 2000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  // -------------------- Helpers --------------------
  const totalSizeMB = useMemo(() => {
    const sum = files.reduce((a, f) => a + f.size, 0);
    return (sum / (1024 * 1024)).toFixed(2);
  }, [files]);

  function handleChoose(e) {
    const picked = Array.from(e.target.files || []);
    setFiles(picked);
    setResults([]);
  }

  function clearAll() {
    setFiles([]);
    setResults([]);
    if (inputRef.current) inputRef.current.value = "";
  }

  // -------------------- Recognize (batch in ONE request) --------------------
  async function handleRecognize() {
    if (!files.length || loading) return;

    setLoading(true);
    setResults([]);

    try {
      const formData = new FormData();

      // IMPORTANT: backend expects field name = "files"
      // and supports multiple files
      for (const f of files) {
        formData.append("files", f);
      }

      const res = await axios.post(`${API_BASE}/predict`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 60000,
      });

      // backend returns: { lines: [ { line_id, text, confidence, meta } ] }
      const lines = res?.data?.lines || [];

      // Map backend response to UI rows
      const mapped = lines.map((item) => ({
        name: item?.line_id || "unknown",
        text: item?.text ?? "",
        confidence: Number(item?.confidence ?? 0),
      }));

      setResults(mapped);
    } catch (err) {
      console.error(err);
      // Show a readable error result
      setResults(
        files.map((f) => ({
          name: f.name,
          text: "Error (check backend console)",
          confidence: 0,
        }))
      );
    } finally {
      setLoading(false);
    }
  }

  // -------------------- UI --------------------
  return (
    <div className="min-h-screen bg-slate-50 p-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">
              Sinhala HTR System
            </h1>
            <p className="text-slate-600 text-sm mt-1">
              Writer-Independent Line-Level Recognition
            </p>
          </div>

          <div className="flex items-center gap-2">
            <div
              className={`w-3 h-3 rounded-full ${
                apiOnline ? "bg-green-500" : "bg-red-500"
              }`}
            ></div>
            <span className="text-sm text-slate-700">
              {apiOnline ? "API Online" : "API Offline"}
            </span>
          </div>
        </div>

        {/* Upload Section */}
        <div className="bg-white rounded-xl shadow p-6 mb-6">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold">Upload Line Images</h2>
            <div className="text-xs text-slate-600">
              {files.length ? `${files.length} file(s) • ${totalSizeMB} MB` : "No files selected"}
            </div>
          </div>

          <div className="mt-4 flex flex-wrap gap-3 items-center">
            <input
              ref={inputRef}
              type="file"
              multiple
              accept="image/png,image/jpeg,image/jpg"
              onChange={handleChoose}
              className="block"
            />

            <button
              onClick={handleRecognize}
              disabled={!files.length || loading || !apiOnline}
              className={`px-5 py-2 rounded-lg font-medium ${
                !files.length || loading || !apiOnline
                  ? "bg-slate-200 text-slate-500 cursor-not-allowed"
                  : "bg-slate-900 text-white hover:bg-slate-800"
              }`}
            >
              {loading ? "Recognizing..." : "Recognize"}
            </button>

            <button
              onClick={clearAll}
              disabled={!files.length && !results.length}
              className={`px-5 py-2 rounded-lg font-medium ${
                !files.length && !results.length
                  ? "bg-slate-200 text-slate-500 cursor-not-allowed"
                  : "bg-white border border-slate-200 text-slate-800 hover:bg-slate-100"
              }`}
            >
              Clear
            </button>
          </div>

          {/* Quick file list */}
          {files.length > 0 && (
            <div className="mt-4 text-sm text-slate-700">
              <div className="font-medium mb-2">Selected:</div>
              <ul className="list-disc pl-5 space-y-1">
                {files.slice(0, 8).map((f) => (
                  <li key={f.name} className="truncate">
                    {f.name}
                  </li>
                ))}
              </ul>
              {files.length > 8 && (
                <div className="text-xs text-slate-500 mt-2">
                  Showing first 8 files…
                </div>
              )}
            </div>
          )}
        </div>

        {/* Results Section */}
        <div className="bg-white rounded-xl shadow p-6">
          <h2 className="text-lg font-semibold mb-4">Results</h2>

          {results.length === 0 ? (
            <p className="text-slate-500 text-sm">
              No predictions yet. Upload images and click <b>Recognize</b>.
            </p>
          ) : (
            <div className="space-y-4">
              {results.map((r, index) => (
                <div
                  key={index}
                  className="border rounded-lg p-4 bg-slate-50"
                >
                  <p className="text-sm text-slate-500 truncate">{r.name}</p>

                  <div className="mt-2 text-lg font-medium text-slate-900 whitespace-pre-wrap break-words">
                    {r.text || <span className="text-slate-400">—</span>}
                  </div>

                  <div className="mt-3">
                    <div className="h-2 bg-slate-200 rounded-full overflow-hidden">
                      <div
                        className="h-2 bg-slate-900"
                        style={{
                          width: `${Math.max(0, Math.min(1, r.confidence)) * 100}%`,
                        }}
                      />
                    </div>
                    <p className="text-xs text-slate-600 mt-1">
                      Confidence: {(Math.max(0, Math.min(1, r.confidence)) * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="mt-8 text-center text-xs text-slate-500">
          FastAPI + React (Vite) • Sinhala HTR Prototype
        </div>
      </div>
    </div>
  );
}
