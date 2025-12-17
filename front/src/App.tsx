import { useMemo, useState } from "react";
import type { LatLngExpression } from "leaflet";
import L from "leaflet";
import { RouteForm } from "./components/RouteForm";
import { MapView } from "./components/MapView";
import { RouteSummary } from "./components/RouteSummary";
import { Toasts, type ToastMessage } from "./components/Toast";
import type { BackendRouteResponse, RouteRequest, RouteResponse } from "./types";
import { useEffect } from "react";
import "leaflet/dist/leaflet.css";
import "./index.css";

import markerIcon2x from "leaflet/dist/images/marker-icon-2x.png";
import markerIcon from "leaflet/dist/images/marker-icon.png";
import markerShadow from "leaflet/dist/images/marker-shadow.png";

L.Icon.Default.mergeOptions({
  iconRetinaUrl: markerIcon2x,
  iconUrl: markerIcon,
  shadowUrl: markerShadow
});

const DEFAULT_API = (import.meta.env.VITE_API_BASE_URL || "").trim() || "/api";

function safeToDate(value?: string) {
  if (!value) return null;
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? null : d;
}

function parseCoord(input: string) {
  const parts = input.split(",").map((p) => p.trim());
  if (parts.length !== 2) {
    throw new Error("Используйте формат 'lat,lon'");
  }
  const [latStr, lonStr] = parts;
  const lat = Number(latStr);
  const lon = Number(lonStr);
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
    throw new Error("Координаты должны быть числами");
  }
  return { lat, lon };
}

type Mode = "sumo" | "stub";

export default function App() {
  const [route, setRoute] = useState<RouteResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastPayload, setLastPayload] = useState<RouteRequest | null>(null);
  const [lastPoints, setLastPoints] = useState<{ origin?: LatLngExpression; destination?: LatLngExpression }>({});
  const [toasts, setToasts] = useState<ToastMessage[]>([]);
  const [mode, setMode] = useState<Mode>("sumo"); // sumo = /simulate_sumo, stub = /simulate
  const [isDark, setIsDark] = useState<boolean>(() => {
    if (typeof window === "undefined") return false;
    const stored = localStorage.getItem("theme");
    if (stored === "dark") return true;
    if (stored === "light") return false;
    return window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? false;
  });

  useEffect(() => {
    const root = document.documentElement;
    if (isDark) {
      root.classList.add("dark");
      localStorage.setItem("theme", "dark");
    } else {
      root.classList.remove("dark");
      localStorage.setItem("theme", "light");
    }
  }, [isDark]);

  const submitPayload = async (payload: RouteRequest) => {
    setLoading(true);
    setError(null);
    try {
      const origin = parseCoord(payload.from);
      const destination = parseCoord(payload.to);
      setLastPayload(payload);
      setLastPoints({ origin: [origin.lat, origin.lon], destination: [destination.lat, destination.lon] });

      const endpoint = mode === "sumo" ? "simulate_sumo_trace" : "simulate";

      const res = await fetch(`${DEFAULT_API}/${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          origin,
          destination,
          preferred_departure_time: payload.departure_time
        })
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `HTTP ${res.status}`);
      }

      const data: BackendRouteResponse = await res.json();

      const polyline = data.optimized_route?.map((p) => [p.lat, p.lon] as [number, number]);
      if (!polyline || polyline.length < 2) {
        throw new Error("Маршрут не найден");
      }
      const etaMinutes = (data.estimated_travel_time_seconds || 0) / 60;
      const avgSpeed = data.meta?.avg_speed_kmh ?? data.meta?.avg_speed_kmph;
      const hints: string[] = [];
      if (data.meta?.note) {
        hints.push(data.meta.note);
      }
      hints.push(mode === "sumo" ? "Режим: SUMO" : "Режим: быстрый прототип");

      setRoute({
        eta_minutes: etaMinutes,
        recommended_departure: data.recommended_departure_time,
        duration_minutes: etaMinutes,
        distance_km: data.meta?.distance_km,
        avg_speed_kmh: avgSpeed,
        polyline,
        hints,
        steps: data.steps
      });
      setToasts((prev) => [
        ...prev,
        { id: crypto.randomUUID(), kind: "success", text: "Маршрут построен" }
      ]);
    } catch (e) {
      const message = e instanceof Error ? e.message : "Неизвестная ошибка";
      setError(`Не удалось получить маршрут: ${message}`);
      setRoute(null);
      setToasts((prev) => [
        ...prev,
        { id: crypto.randomUUID(), kind: "error", text: `Ошибка: ${message}` }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (payload: RouteRequest) => {
    await submitPayload(payload);
  };

  const handleRetry = async () => {
    if (!lastPayload || loading) return;
    await submitPayload(lastPayload);
  };

  const handleReset = () => {
    setRoute(null);
    setError(null);
    setLastPayload(null);
  };

  const removeToast = (id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  };

  const polyline = useMemo<LatLngExpression[] | undefined>(() => {
    if (!route?.polyline) return undefined;
    return route.polyline;
  }, [route]);

  const start = polyline?.[0];
  const end = polyline && polyline.length > 1 ? polyline[polyline.length - 1] : undefined;

  const recommendedTime = safeToDate(route?.recommended_departure);
  const routeSummaryText = route
    ? `~${Math.round(route.distance_km ?? 0)} км · ~${Math.round(route.duration_minutes ?? route.eta_minutes)} мин`
    : undefined;

  return (
    <div className="app-shell">
      <Toasts toasts={toasts} onDismiss={removeToast} />
      <header className="header">
        <div>
          <h1 className="title">UrbanIQ</h1>
        </div>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          <span className="badge">Алматы · GMT+5</span>
          {recommendedTime && <span className="pill">Стартуй: {recommendedTime.toLocaleTimeString("ru-RU", { hour: "2-digit", minute: "2-digit" })}</span>}
          <div className="pill" style={{ display: "flex", gap: 6, alignItems: "center" }}>
            <span>Режим:</span>
            <button
              className={`button ${mode === "sumo" ? "" : "secondary"}`}
              type="button"
              onClick={() => setMode("sumo")}
              disabled={mode === "sumo"}
              style={{ padding: "4px 8px" }}
            >
              SUMO
            </button>
            <button
              className={`button ${mode === "stub" ? "" : "secondary"}`}
              type="button"
              onClick={() => setMode("stub")}
              disabled={mode === "stub"}
              style={{ padding: "4px 8px" }}
            >
              Прототип
            </button>
          </div>
          <button
            className="button secondary"
            type="button"
            onClick={() => setIsDark((v) => !v)}
            style={{ minWidth: 120 }}
          >
            {isDark ? "Светлая" : "Тёмная"}
          </button>
        </div>
      </header>

      <div className="layout">
        <div className="card">
          <RouteForm onSubmit={handleSubmit} loading={loading} onReset={handleReset} />
          <div style={{ marginTop: 10 }}>
            {loading && <div className="status">Считаем трафик и светофоры...</div>}
            {error && <div className="error">{error}</div>}
            {!loading && !error && !route && <div className="status">Введите точки или выберите адреса, затем постройте маршрут.</div>}
          </div>
          <div style={{ marginTop: 10, display: "flex", gap: 8, flexWrap: "wrap" }}>
            <button className="button secondary" type="button" onClick={handleReset} disabled={loading}>
              Очистить результат
            </button>
            <button className="button" type="button" onClick={handleRetry} disabled={loading || !lastPayload}>
              Повторить запрос
            </button>
          </div>
        </div>

        <div className="card map-card">
          <MapView
            polyline={polyline}
            start={start}
            end={end}
            loading={loading}
            fallbackStart={lastPoints.origin}
            fallbackEnd={lastPoints.destination}
            routeSummary={routeSummaryText}
          />
        </div>
      </div>

      <RouteSummary data={route} />
    </div>
  );
}

