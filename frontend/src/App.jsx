import { useEffect, useMemo, useRef, useState } from "react";
import axios from "axios";

const API_BASE = "http://localhost:8000";
const HISTORY_KEY = "sinhala_htr_history_v1";

function nowStamp() {
  const d = new Date();
  return d.toLocaleString();
}

function downloadCSV(rows, filename = "predictions.csv") {
  const header = ["line_id", "text", "confidence"];
  const escape = (v) => `"${String(v ?? "").replaceAll('"', '""')}"`;
  const csv =
    [header.join(","), ...rows.map((r) => [r.line_id, r.text, r.confidence].map(escape).join(","))].join(
      "\n"
    );

  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function loadHistory() {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function saveHistory(items) {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(items));
  } catch {
    // ignore
  }
}

export default function App() {
  const [files, setFiles] = useState([]);
  const [rows, setRows] = useState([]);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");

  // API status
  const [apiOk, setApiOk] = useState(false);
  const [apiMsg, setApiMsg] = useState("Checking API...");
  const apiTimerRef = useRef(null);

  // Drag & Drop UI
  const [dragOver, setDragOver] = useState(false);

  // History
  const [history, setHistory] = useState(() => loadHistory()); // [{id, at, fileCount, rows}]
  const [activeHistoryId, setActiveHistoryId] = useState(null);

  const previews = useMemo(() => {
    return files.map((f) => ({ file: f, url: URL.createObjectURL(f) }));
  }, [files]);

  // Cleanup preview URLs
  useEffect(() => {
    return () => previews.forEach((p) => URL.revokeObjectURL(p.url));
  }, [previews]);

  const checkApi = async () => {
    try {
      const res = await axios.get(`${API_BASE}/health`, { timeout: 1200 });
      if (res?.data?.status === "ok") {
        setApiOk(true);
        setApiMsg("API Online");
        return;
      }
      setApiOk(false);
      setApiMsg("API responded, but unexpected");
    } catch {
      setApiOk(false);
      setApiMsg("API Offline (start backend :8000)");
    }
  };

  useEffect(() => {
    checkApi();
    apiTimerRef.current = setInterval(checkApi, 4000);
    return () => {
      if (apiTimerRef.current) clearInterval(apiTimerRef.current);
    };
  }, []);

  const setPickedFiles = (picked) => {
    setError("");
    setStatus("");
    setFiles(picked);
    setRows([]);
    setActiveHistoryId(null);
  };

  const onPickFiles = (e) => {
    const picked = Array.from(e.target.files || []);
    setPickedFiles(picked);
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const dropped = Array.from(e.dataTransfer.files || []).filter((f) => f.type.startsWith("image/"));
    if (!dropped.length) {
      setError("Drop image files only (PNG/JPG).");
      return;
    }
    setPickedFiles(dropped);
  };

  const clearAll = () => {
    setError("");
    setStatus("");
    setFiles([]);
    setRows([]);
    setActiveHistoryId(null);
  };

  const recognize = async () => {
    setError("");
    setStatus("");

    if (!apiOk) {
      setError("Backend is offline. Start FastAPI on http://localhost:8000");
      return;
    }

    if (!files.length) {
      setError("Please upload at least 1 line image.");
      return;
    }

    setBusy(true);
    setStatus("Uploading & recognizing...");

    try {
      const form = new FormData();
      files.forEach((f) => form.append("files", f));

      const res = await axios.post(`${API_BASE}/predict`, form, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      const out = res.data.lines || [];
      setRows(out);
      setStatus(`Done ✅ (${out.length} lines)`);

      // Save to history
      const entry = {
        id: crypto.randomUUID ? crypto.randomUUID() : String(Date.now()),
        at: nowStamp(),
        fileCount: files.length,
        fileNames: files.map((f) => f.name).slice(0, 30),
        rows: out,
      };

      const next = [entry, ...history].slice(0, 15); // keep last 15 runs
      setHistory(next);
      saveHistory(next);
      setActiveHistoryId(entry.id);
    } catch {
      setError("Prediction call failed. Check backend logs and CORS.");
    } finally {
      setBusy(false);
    }
  };

  const openHistory = (id) => {
    const item = history.find((h) => h.id === id);
    if (!item) return;
    setActiveHistoryId(id);
    setFiles([]); // history doesn't restore images, only results
    setRows(item.rows || []);
    setStatus(`Loaded history ✅ (${item.at})`);
    setError("");
  };

  const deleteHistory = (id) => {
    const next = history.filter((h) => h.id !== id);
    setHistory(next);
    saveHistory(next);
    if (activeHistoryId === id) setActiveHistoryId(null);
  };

  const clearHistory = () => {
    setHistory([]);
    saveHistory([]);
    setActiveHistoryId(null);
  };

  return (
    <div style={styles.page}>
      <div style={styles.shell}>
        {/* Left: main */}
        <div style={styles.card}>
          <div style={styles.headerRow}>
            <div>
              <h1 style={styles.title}>Sinhala HTR Demo</h1>
              <p style={styles.subtitle}>
                Upload segmented line images → Recognize → view predicted Sinhala Unicode text.
              </p>
            </div>

            <div style={styles.rightHeader}>
              <div style={styles.badge}>React + FastAPI</div>
              <div style={styles.apiRow} title={apiMsg}>
                <span style={{ ...styles.dot, background: apiOk ? "#16a34a" : "#ef4444" }} />
                <span style={{ fontSize: 12, opacity: 0.8 }}>{apiMsg}</span>
              </div>
            </div>
          </div>

          {/* Drag & Drop area */}
          <div
            onDragEnter={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={(e) => {
              e.preventDefault();
              setDragOver(false);
            }}
            onDrop={onDrop}
            style={{
              ...styles.dropZone,
              borderColor: dragOver ? "#111827" : "#c7c9d3",
              background: dragOver ? "#f3f4f6" : "white",
            }}
          >
            <div style={{ fontWeight: 800 }}>Drag & drop line images here</div>
            <div style={{ fontSize: 12, opacity: 0.75, marginTop: 6 }}>
              or click “Choose Images” below (PNG/JPG)
            </div>
          </div>

          <div style={styles.controls}>
            <label style={styles.fileBox}>
              <input
                style={{ display: "none" }}
                type="file"
                multiple
                accept="image/*"
                onChange={onPickFiles}
              />
              <div style={{ fontWeight: 700 }}>Choose Images</div>
              <div style={{ fontSize: 12, opacity: 0.75 }}>
                {files.length ? `${files.length} selected` : "No files selected"}
              </div>
            </label>

            <button style={{ ...styles.btn, ...styles.primary }} onClick={recognize} disabled={busy}>
              {busy ? "Recognizing..." : "Recognize"}
            </button>

            <button style={{ ...styles.btn, ...styles.ghost }} onClick={clearAll} disabled={busy}>
              Clear
            </button>

            <button
              style={{ ...styles.btn, ...styles.ghost }}
              onClick={() => downloadCSV(rows, "predictions.csv")}
              disabled={!rows.length || busy}
              title={!rows.length ? "Run recognition first" : "Download predictions.csv"}
            >
              Download CSV
            </button>
          </div>

          {error && <div style={styles.alertError}>{error}</div>}
          {status && !error && <div style={styles.alertOk}>{status}</div>}

          {!!files.length && (
            <>
              <h3 style={styles.sectionTitle}>Preview</h3>
              <div style={styles.grid}>
                {previews.map((p) => (
                  <div key={p.url} style={styles.thumb}>
                    <img src={p.url} alt="" style={styles.thumbImg} />
                    <div style={styles.thumbName}>{p.file.name}</div>
                  </div>
                ))}
              </div>
            </>
          )}

          {!!rows.length && (
            <>
              <h3 style={styles.sectionTitle}>Predictions</h3>
              <div style={styles.tableWrap}>
                <table style={styles.table}>
                  <thead>
                    <tr>
                      <th style={styles.th}>Line ID</th>
                      <th style={styles.th}>Text</th>
                      <th style={styles.th}>Confidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((r, i) => {
                      const conf = Number(r.confidence ?? 0);
                      const pct = Math.max(0, Math.min(100, Math.round(conf * 100)));
                      return (
                        <tr key={i} style={styles.tr}>
                          <td style={styles.tdMono}>{r.line_id}</td>
                          <td style={styles.td}>{r.text}</td>
                          <td style={styles.td}>
                            <div style={styles.confRow}>
                              <div style={styles.barBg}>
                                <div style={{ ...styles.barFill, width: `${pct}%` }} />
                              </div>
                              <div style={styles.pct}>{pct}%</div>
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </>
          )}

          <div style={styles.footer}>
            Tip: Use this UI screenshot in your final report “Prototype / System Implementation” section.
          </div>
        </div>

        {/* Right: history */}
        <div style={styles.side}>
          <div style={styles.sideCard}>
            <div style={styles.sideHeader}>
              <div style={{ fontWeight: 900 }}>History</div>
              <button style={styles.smallBtn} onClick={clearHistory} disabled={!history.length}>
                Clear
              </button>
            </div>

            {!history.length ? (
              <div style={{ fontSize: 13, opacity: 0.75, padding: "10px 0" }}>
                No history yet. Run recognition once and it will appear here.
              </div>
            ) : (
              <div style={styles.historyList}>
                {history.map((h) => (
                  <div
                    key={h.id}
                    style={{
                      ...styles.historyItem,
                      borderColor: h.id === activeHistoryId ? "#111827" : "#eee",
                    }}
                  >
                    <div style={{ display: "flex", justifyContent: "space-between", gap: 10 }}>
                      <div>
                        <div style={{ fontWeight: 800, fontSize: 13 }}>{h.at}</div>
                        <div style={{ fontSize: 12, opacity: 0.75 }}>
                          {h.fileCount} file(s) • {h.rows?.length ?? 0} line(s)
                        </div>
                      </div>

                      <div style={{ display: "flex", gap: 6 }}>
                        <button style={styles.smallBtn} onClick={() => openHistory(h.id)}>
                          Open
                        </button>
                        <button
                          style={styles.smallBtn}
                          onClick={() => downloadCSV(h.rows || [], `predictions_${h.at.replaceAll(":", "-")}.csv`)}
                        >
                          CSV
                        </button>
                        <button style={{ ...styles.smallBtn, color: "#b91c1c" }} onClick={() => deleteHistory(h.id)}>
                          X
                        </button>
                      </div>
                    </div>

                    {!!h.fileNames?.length && (
                      <div style={styles.fileNames}>
                        {h.fileNames.slice(0, 3).join(", ")}
                        {h.fileNames.length > 3 ? ` +${h.fileNames.length - 3} more` : ""}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}

            <div style={{ marginTop: 10, fontSize: 12, opacity: 0.65 }}>
              Stored locally in your browser (localStorage).
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

const styles = {
  page: {
    minHeight: "100vh",
    background: "#f6f7fb",
    padding: "22px 12px",
    fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, Arial",
  },
  shell: {
    maxWidth: 1250,
    margin: "0 auto",
    display: "grid",
    gridTemplateColumns: "1fr 340px",
    gap: 14,
    alignItems: "start",
  },
  card: {
    background: "white",
    borderRadius: 16,
    boxShadow: "0 8px 30px rgba(0,0,0,0.08)",
    padding: 20,
  },
  side: {},
  sideCard: {
    background: "white",
    borderRadius: 16,
    boxShadow: "0 8px 30px rgba(0,0,0,0.08)",
    padding: 16,
    position: "sticky",
    top: 12,
  },
  headerRow: { display: "flex", justifyContent: "space-between", gap: 12, alignItems: "start" },
  rightHeader: { display: "flex", flexDirection: "column", gap: 8, alignItems: "flex-end" },
  title: { margin: 0, fontSize: 34, letterSpacing: -0.5 },
  subtitle: { marginTop: 8, marginBottom: 0, opacity: 0.75 },
  badge: {
    background: "#eef2ff",
    color: "#3730a3",
    padding: "8px 10px",
    borderRadius: 999,
    fontSize: 12,
    fontWeight: 800,
    whiteSpace: "nowrap",
    width: "fit-content",
  },
  apiRow: { display: "flex", alignItems: "center", gap: 8 },
  dot: { width: 10, height: 10, borderRadius: 999, display: "inline-block" },

  dropZone: {
    marginTop: 14,
    padding: "18px 14px",
    border: "2px dashed #c7c9d3",
    borderRadius: 14,
    textAlign: "center",
    transition: "all 0.15s ease",
  },

  controls: {
    display: "flex",
    flexWrap: "wrap",
    gap: 10,
    marginTop: 14,
    alignItems: "center",
  },
  fileBox: {
    border: "1px dashed #c7c9d3",
    borderRadius: 12,
    padding: "10px 12px",
    cursor: "pointer",
    minWidth: 200,
  },
  btn: {
    borderRadius: 12,
    padding: "10px 14px",
    border: "1px solid #d6d7df",
    background: "white",
    cursor: "pointer",
    fontWeight: 800,
  },
  primary: {
    background: "#111827",
    color: "white",
    border: "1px solid #111827",
  },
  ghost: { background: "white" },
  alertError: {
    marginTop: 14,
    background: "#ffe5e5",
    padding: 10,
    borderRadius: 12,
    border: "1px solid #ffb3b3",
  },
  alertOk: {
    marginTop: 14,
    background: "#eaffea",
    padding: 10,
    borderRadius: 12,
    border: "1px solid #b6f2b6",
  },
  sectionTitle: { marginTop: 18, marginBottom: 10 },
  grid: { display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))", gap: 12 },
  thumb: { border: "1px solid #eee", borderRadius: 14, padding: 10 },
  thumbImg: { width: "100%", height: 90, objectFit: "contain", borderRadius: 10, background: "#fafafa" },
  thumbName: { fontSize: 12, marginTop: 8, opacity: 0.8, wordBreak: "break-all" },
  tableWrap: { border: "1px solid #eee", borderRadius: 14, overflow: "hidden" },
  table: { width: "100%", borderCollapse: "collapse" },
  th: { textAlign: "left", padding: 12, background: "#f7f7fb", fontSize: 13 },
  tr: { borderTop: "1px solid #eee" },
  td: { padding: 12, verticalAlign: "top" },
  tdMono: { padding: 12, fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace", fontSize: 13 },
  confRow: { display: "flex", gap: 10, alignItems: "center" },
  barBg: { width: 160, height: 10, background: "#e5e7eb", borderRadius: 999, overflow: "hidden" },
  barFill: { height: "100%", background: "#111827" },
  pct: { fontSize: 12, opacity: 0.8, minWidth: 40 },
  footer: { marginTop: 14, fontSize: 12, opacity: 0.65 },

  sideHeader: { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 },
  smallBtn: {
    borderRadius: 10,
    padding: "6px 10px",
    border: "1px solid #e5e7eb",
    background: "white",
    cursor: "pointer",
    fontWeight: 800,
    fontSize: 12,
  },
  historyList: { display: "flex", flexDirection: "column", gap: 10 },
  historyItem: { border: "1px solid #eee", borderRadius: 14, padding: 10 },
  fileNames: { marginTop: 8, fontSize: 12, opacity: 0.75, wordBreak: "break-word" },
};
