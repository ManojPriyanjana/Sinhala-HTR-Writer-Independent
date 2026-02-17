import { useMemo, useState } from "react";
import axios from "axios";

const API_BASE = "http://localhost:8000";

export default function App() {
  const [files, setFiles] = useState([]);
  const [rows, setRows] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const previews = useMemo(() => {
    return files.map((f) => ({ file: f, url: URL.createObjectURL(f) }));
  }, [files]);

  const onPickFiles = (e) => {
    setError("");
    const picked = Array.from(e.target.files || []);
    setFiles(picked);
    setRows([]);
  };

  const recognize = async () => {
    setError("");
    if (!files.length) return setError("Upload at least 1 image.");
    setBusy(true);

    try {
      const form = new FormData();
      files.forEach((f) => form.append("files", f));

      const res = await axios.post(`${API_BASE}/predict`, form, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setRows(res.data.lines || []);
    } catch {
      setError("Backend call failed. Is FastAPI running on port 8000?");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div style={{ fontFamily: "system-ui", padding: 20, maxWidth: 1100, margin: "0 auto" }}>
      <h1>Sinhala HTR Demo (React + FastAPI)</h1>
      <p style={{ opacity: 0.8 }}>
        Upload segmented line images → Recognize → view predicted Sinhala Unicode text (mock now).
      </p>

      <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 16 }}>
        <input type="file" multiple accept="image/*" onChange={onPickFiles} />
        <button onClick={recognize} disabled={busy}>
          {busy ? "Recognizing..." : "Recognize"}
        </button>
      </div>

      {error && (
        <div style={{ background: "#ffe5e5", padding: 10, borderRadius: 8, marginBottom: 12 }}>
          {error}
        </div>
      )}

      {!!rows.length && (
        <>
          <h3>Predictions</h3>
          <ul>
            {rows.map((r, i) => (
              <li key={i}>
                <b>{r.line_id}</b> — {r.text} (conf: {r.confidence})
              </li>
            ))}
          </ul>
        </>
      )}

      {!!files.length && !rows.length && (
        <p style={{ opacity: 0.7 }}>Files selected: {files.length}</p>
      )}
    </div>
  );
}
