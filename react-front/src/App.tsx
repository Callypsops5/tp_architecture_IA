import { useState } from "react";
import "./App.css";

type Result = { line: string; score: number };
type SearchResponse = { query: string; cluster_index: string; results: Result[] };

const API_URL = "http://localhost:8000/search";

function App() {
  const [text, setText] = useState("");
  const [data, setData] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSearch = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: text, k: 5 }),
      });
      if (!res.ok) throw new Error(`Erreur HTTP ${res.status}`);
      setData(await res.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : "Erreur inconnue");
      setData(null);
    } finally {
      setLoading(false);
    }
  };

      return (
    <div className="container">
      <h1>Recherche sémantique</h1>

      <div className="search-bar">
        <input
          type="text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSearch()}
          placeholder="Saisis ta recherche"
        />
        <button onClick={handleSearch} disabled={loading}>
          {loading ? "Recherche..." : "Search"}
        </button>
      </div>

      {error && <p className="error">{error}</p>}

      {data && (
        <>
          <p className="cluster">
            Cluster le plus proche : <span className="badge">Cluster {data.cluster_index.replace("data_", "")}</span>
          </p>
          <p className="score-legend">
            Score de pertinence : plus il est élevé, plus la ligne correspond à la recherche.
          </p>
          <ul className="results">
            {data.results.map((r, i) => (
              <li key={i}>
                <span>{r.line}</span>
                <span className="score">{r.score.toFixed(3)}</span>
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}

export default App;