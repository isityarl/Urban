import { useEffect, useMemo, useRef, useState } from "react";

interface Suggestion {
  display_name: string;
  lat: string;
  lon: string;
}

interface Props {
  label: string;
  value: string;
  placeholder?: string;
  disabled?: boolean;
  onChange: (next: string) => void;
  onSelect: (coords: { lat: number; lon: number; label: string }) => void;
}

const DEBOUNCE_MS = 350;

export function GeocoderInput({ label, value, placeholder, disabled, onChange, onSelect }: Props) {
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const controller = useRef<AbortController | null>(null);

  const query = value.trim();

  const isLikelyCoords = useMemo(() => {
    const parts = query.split(",").map((p) => p.trim());
    return parts.length === 2 && parts.every((p) => !Number.isNaN(Number(p)));
  }, [query]);

  useEffect(() => {
    if (!query || query.length < 3 || isLikelyCoords) {
      setSuggestions([]);
      setError(null);
      if (controller.current) {
        controller.current.abort();
      }
      return;
    }

    const id = setTimeout(async () => {
      setLoading(true);
      setError(null);
      controller.current?.abort();
      controller.current = new AbortController();

      try {
        const res = await fetch(
          `https://nominatim.openstreetmap.org/search?format=jsonv2&limit=5&q=${encodeURIComponent(query)}`,
          {
            headers: { Accept: "application/json" },
            signal: controller.current.signal
          }
        );
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`);
        }
        const data: Suggestion[] = await res.json();
        setSuggestions(data);
      } catch (e) {
        if ((e as Error).name === "AbortError") return;
        const msg = e instanceof Error ? e.message : "Ошибка геокодера";
        setError(msg);
      } finally {
        setLoading(false);
      }
    }, DEBOUNCE_MS);

    return () => clearTimeout(id);
  }, [query, isLikelyCoords]);

  return (
    <div className="field">
      <label className="label">{label}</label>
      <input
        className="input"
        placeholder={placeholder}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
      />
      {loading && <div className="status">Ищем адрес...</div>}
      {error && <div className="error" style={{ marginTop: 6 }}>{error}</div>}
      {suggestions.length > 0 && (
        <div className="suggestions">
          {suggestions.map((s, idx) => (
            <button
              key={`${s.lat}-${s.lon}-${idx}`}
              type="button"
              className="suggestion"
              onClick={() => {
                onSelect({ lat: Number(s.lat), lon: Number(s.lon), label: s.display_name });
                setSuggestions([]);
              }}
            >
              <div className="suggestion-title">{s.display_name}</div>
              <div className="suggestion-sub">
                {Number(s.lat).toFixed(5)}, {Number(s.lon).toFixed(5)}
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}











