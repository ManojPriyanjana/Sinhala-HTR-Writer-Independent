import { useMemo, useState } from "react";
import axios from "axios";

const API_BASE = "http://localhost:8000";

function downloadCSV(rows) {
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
  a.download = "predictions.csv";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export default function App() {
  const [files, setFiles] = useState([]);
  const [rows, setRows] = useState([]);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");

  const previews = useMemo(() => {
    return files.map((f) => ({ file: f, url: URL.createObjectURL(f) }));
  }, [files]);

  const onPickFiles = (e) => {
    setError("");
    setStatus("");
    const picked = Array.from(e.target.files || []);
    setFiles(picked);
    setRows([]);
  };

  const clearAll = () => {
    setError("");
    setStatus("");
    setFiles([]);
    setRows([]);
  };

  const recognize = async () => {
    setError("");
    setStatus("");
    if (!files.length) return setError("Please upload at least 1 line image.");
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
    } catch (e) {
      setError("Backend call failed. Make sure FastAPI is running on http://localhost:8000");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div style={styles.page}>
      <div style={styles.card}>
        <div style={styles.headerRow}>
          <div>
            <h1 style={styles.title}>Sinhala HTR Demo</h1>
            <p style={styles.subtitle}>
              Upload segmented line images → Recognize → view predicted Sinhala Unicode text.
            </p>
          </div>
          <div style={styles.badge}>React + FastAPI</div>
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
            <div style={{ fontWeight: 600 }}>Choose Images</div>
            <div style={{ fontSize: 12, opacity: 0.75 }}>
              {files.length ? `${files.length} selected` : "PNG/JPG line crops"}
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
            onClick={() => downloadCSV(rows)}
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
          Tip: Use this UI screenshot in your final report “System Implementation / Prototype” section.
        </div>
      </div>
    </div>
  );
}

const styles = {
  page: {
    minHeight: "100vh",
    background: "#f6f7fb",
    padding: "28px 14px",
    fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, Arial",
  },
  card: {
    maxWidth: 1100,
    margin: "0 auto",
    background: "white",
    borderRadius: 16,
    boxShadow: "0 8px 30px rgba(0,0,0,0.08)",
    padding: 22,
  },
  headerRow: { display: "flex", justifyContent: "space-between", gap: 12, alignItems: "start" },
  title: { margin: 0, fontSize: 34, letterSpacing: -0.5 },
  subtitle: { marginTop: 8, marginBottom: 0, opacity: 0.75 },
  badge: {
    background: "#eef2ff",
    color: "#3730a3",
    padding: "8px 10px",
    borderRadius: 999,
    fontSize: 12,
    fontWeight: 700,
    whiteSpace: "nowrap",
  },
  controls: {
    display: "flex",
    flexWrap: "wrap",
    gap: 10,
    marginTop: 18,
    alignItems: "center",
  },
  fileBox: {
    border: "1px dashed #c7c9d3",
    borderRadius: 12,
    padding: "10px 12px",
    cursor: "pointer",
    minWidth: 180,
  },
  btn: {
    borderRadius: 12,
    padding: "10px 14px",
    border: "1px solid #d6d7df",
    background: "white",
    cursor: "pointer",
    fontWeight: 700,
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
  footer: { marginTop: 16, fontSize: 12, opacity: 0.65 },
};
