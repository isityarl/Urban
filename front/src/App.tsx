import { useMemo, useState } from "react";
import type { LatLngExpression } from "leaflet";
import L from "leaflet";
import axios from "axios";
import { RouteForm } from "./components/RouteForm";
import { MapView } from "./components/MapView";
import { ResultCard } from "./components/ResultCard";
import { Toasts, type ToastMessage } from "./components/Toast";
import type { BackendRouteResponse, RouteRequest, RouteResponse, TraceStep } from "./types";
import { useEffect } from "react";
import "leaflet/dist/leaflet.css";
import "./index.css";
import { api } from "./api/http";

import markerIcon2x from "leaflet/dist/images/marker-icon-2x.png";
import markerIcon from "leaflet/dist/images/marker-icon.png";
import markerShadow from "leaflet/dist/images/marker-shadow.png";

L.Icon.Default.mergeOptions({
  iconRetinaUrl: markerIcon2x,
  iconUrl: markerIcon,
  shadowUrl: markerShadow
});

const TZ_OFFSET = "+05:00";

function safeToDate(value?: string) {
  if (!value) return null;
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? null : d;
}

function isoWithFixedOffset(date: Date, offset: string) {
  // offset like "+05:00" or "-03:00"
  const m = /^([+-])(\d{2}):(\d{2})$/.exec(offset);
  if (!m) {
    // Fallback to ISO string if offset format is wrong
    return date.toISOString();
  }
  const sign = m[1] === "-" ? -1 : 1;
  const hh = Number(m[2]);
  const mm = Number(m[3]);
  const offsetMs = sign * (hh * 60 + mm) * 60 * 1000;

  // Create a "UTC view" of the time in the target offset
  const d = new Date(date.getTime() + offsetMs);
  const y = d.getUTCFullYear();
  const mon = String(d.getUTCMonth() + 1).padStart(2, "0");
  const day = String(d.getUTCDate()).padStart(2, "0");
  const h = String(d.getUTCHours()).padStart(2, "0");
  const min = String(d.getUTCMinutes()).padStart(2, "0");
  const sec = String(d.getUTCSeconds()).padStart(2, "0");
  return `${y}-${mon}-${day}T${h}:${min}:${sec}${offset}`;
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

function computeWaitingSeconds(steps?: TraceStep[]) {
  if (!steps || steps.length < 2) return undefined;
  let total = 0;
  let sawSpeed = false;
  for (let i = 1; i < steps.length; i++) {
    const prev = steps[i - 1];
    const curr = steps[i];
    const dt = curr.t - prev.t;
    if (!Number.isFinite(dt) || dt <= 0) continue;
    const speed = prev.vehicle?.speed_kmh;
    if (typeof speed !== "number") continue;
    sawSpeed = true;
    if (speed <= 1) total += dt;
  }
  return sawSpeed ? total : undefined;
}

export default function App() {
  const [route, setRoute] = useState<RouteResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastPayload, setLastPayload] = useState<RouteRequest | null>(null);
  const [lastPoints, setLastPoints] = useState<{ origin?: LatLngExpression; destination?: LatLngExpression }>({});
  const [pointA, setPointA] = useState<{ lat: number; lon: number } | null>(null);
  const [pointB, setPointB] = useState<{ lat: number; lon: number } | null>(null);
  const [toasts, setToasts] = useState<ToastMessage[]>([]);
  const [routeFormKey, setRouteFormKey] = useState(0);
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

      const endpoint = "simulate";
      const { data } = await api.post<BackendRouteResponse>(endpoint, {
        origin,
        destination,
        preferred_departure_time: payload.departure_time,
        // Task 5 (temporary): backend expects/accepts edge ids in the future.
        // For now we send mock placeholders so the contract is wired end-to-end.
        from_edge: payload.from_edge ?? "mock_from_edge",
        to_edge: payload.to_edge ?? "mock_to_edge"
      });

      const polyline = data.optimized_route?.map((p) => [p.lat, p.lon] as [number, number]);
      if (!polyline || polyline.length < 2) {
        throw new Error("Маршрут не найден");
      }
      const etaMinutes = (data.estimated_travel_time_seconds || 0) / 60;
      const avgSpeed = data.meta?.avg_speed_kmh ?? data.meta?.avg_speed_kmph;
      const travelSeconds = data.estimated_travel_time_seconds;
      const distanceM =
        data.meta?.distance_m ??
        (data.meta?.distance_km !== undefined ? data.meta.distance_km * 1000 : undefined);
      const waitingSeconds = computeWaitingSeconds(data.steps);
      const hints: string[] = [];
      if (data.meta?.note) {
        hints.push(data.meta.note);
      }

      setRoute({
        eta_minutes: etaMinutes,
        travel_time_seconds: travelSeconds,
        recommended_departure: data.recommended_departure_time,
        duration_minutes: etaMinutes,
        distance_m: distanceM,
        distance_km: data.meta?.distance_km,
        avg_speed_kmh: avgSpeed,
        waiting_time_seconds: waitingSeconds,
        polyline,
        hints,
        steps: data.steps
      });
      setToasts((prev) => [
        ...prev,
        { id: crypto.randomUUID(), kind: "success", text: "Маршрут построен" }
      ]);
    } catch (e) {
      let message = e instanceof Error ? e.message : "Неизвестная ошибка";
      if (axios.isAxiosError(e)) {
        const status = e.response?.status;
        const payload = e.response?.data;
        const serverText = typeof payload === "string" ? payload : "";
        if (!e.response) {
          message = "Бэкенд недоступен. Запустите server.py (http://127.0.0.1:5500).";
        } else if (
          status === 500 &&
          (serverText.includes("ECONNREFUSED") ||
            serverText.includes("http proxy error") ||
            serverText.includes("connect ECONNREFUSED"))
        ) {
          message = "Бэкенд недоступен. Проверьте, что server.py запущен на порту 5500.";
        } else {
          message = serverText || (status ? `HTTP ${status}` : e.message);
        }
      }
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
    setLastPoints({});
    setPointA(null);
    setPointB(null);
    setRouteFormKey((k) => k + 1);
  };

  const removeToast = (id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  };

  const handleMapClick = (coords: { lat: number; lon: number }) => {
    if (!pointA) {
      setPointA(coords);
      setToasts((prev) => [
        ...prev,
        { id: crypto.randomUUID(), kind: "info", text: `Точка A: ${coords.lat.toFixed(6)}, ${coords.lon.toFixed(6)}` }
      ]);
      return;
    }
    if (!pointB) {
      setPointB(coords);
      setToasts((prev) => [
        ...prev,
        { id: crypto.randomUUID(), kind: "info", text: `Точка B: ${coords.lat.toFixed(6)}, ${coords.lon.toFixed(6)}` }
      ]);
      return;
    }
    // Third click is disabled in the UI (handler is not passed), but keep a guard just in case.
    setToasts((prev) => [
      ...prev,
      { id: crypto.randomUUID(), kind: "info", text: "Точки A и B уже выбраны. Нажмите «Очистить результат» чтобы выбрать заново." }
    ]);
  };

  const handleSimulateClick = async () => {
    if (!pointA || !pointB || loading) return;
    const departure = isoWithFixedOffset(new Date(Date.now() + 15 * 60 * 1000), TZ_OFFSET);
    await submitPayload({
      from: `${pointA.lat},${pointA.lon}`,
      to: `${pointB.lat},${pointB.lon}`,
      departure_time: departure,
      from_edge: "mock_edge_A",
      to_edge: "mock_edge_B"
    });
  };

  const polyline = useMemo<LatLngExpression[] | undefined>(() => {
    if (!route?.polyline) return undefined;
    return route.polyline;
  }, [route]);

  const start = polyline?.[0];
  const end = polyline && polyline.length > 1 ? polyline[polyline.length - 1] : undefined;

  const recommendedTime = safeToDate(route?.recommended_departure);
  const summaryKm =
    route?.distance_m !== undefined
      ? route.distance_m / 1000
      : route?.distance_km;
  const summaryMinutes =
    route?.travel_time_seconds !== undefined
      ? route.travel_time_seconds / 60
      : route?.duration_minutes ?? route?.eta_minutes;
  const routeSummaryText = route
    ? `~${Math.round(summaryKm ?? 0)} км · ~${Math.round(summaryMinutes ?? 0)} мин`
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
          <RouteForm key={routeFormKey} onSubmit={handleSubmit} loading={loading} onReset={handleReset} />
          <div style={{ marginTop: 10 }}>
            {loading && (
              <div className="status" style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <div className="spinner" />
                <span>Выполняется симуляция трафика…</span>
              </div>
            )}
            {error && (
              <div className="error">
                {error}
                <div style={{ marginTop: 6, opacity: 0.9 }}>
                  Нажмите «Reset», чтобы начать заново.
                </div>
              </div>
            )}
            {!loading && !error && !route && <div className="status">Введите точки или выберите адреса, затем постройте маршрут.</div>}
          </div>
          <div style={{ marginTop: 10, display: "flex", gap: 8, flexWrap: "wrap" }}>
            <button className="button secondary" type="button" onClick={handleReset} disabled={loading}>
              Reset
            </button>
            <button className="button" type="button" onClick={handleRetry} disabled={loading || !lastPayload}>
              Повторить запрос
            </button>
            <button className="button" type="button" onClick={handleSimulateClick} disabled={loading || !pointA || !pointB}>
              {loading ? "Simulating…" : "Simulate"}
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
            onMapClick={pointA && pointB ? undefined : handleMapClick}
            pointA={pointA ? ([pointA.lat, pointA.lon] as LatLngExpression) : undefined}
            pointB={pointB ? ([pointB.lat, pointB.lon] as LatLngExpression) : undefined}
          />
        </div>
      </div>

      <ResultCard data={route} />
    </div>
  );
}

