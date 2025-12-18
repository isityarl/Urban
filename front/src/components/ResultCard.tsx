import type { RouteResponse } from "../types";

interface Props {
  data: RouteResponse | null;
}

function formatMinutes(seconds: number) {
  return Math.max(0, Math.round(seconds / 60));
}

function formatKmFromMeters(meters: number) {
  return Math.max(0, meters / 1000);
}

export function ResultCard({ data }: Props) {
  if (!data) return null;

  const travelSeconds = data.travel_time_seconds ?? Math.round(data.eta_minutes * 60);
  const minutes = formatMinutes(travelSeconds);

  const distanceMeters =
    data.distance_m ??
    (data.distance_km !== undefined ? data.distance_km * 1000 : undefined);
  const distanceKm = distanceMeters !== undefined ? formatKmFromMeters(distanceMeters) : undefined;

  const waitingSeconds = data.waiting_time_seconds;

  return (
    <div className="card">
      <div className="summary">
        <div className="pill">Время в пути: ~{minutes} мин</div>
        {distanceKm !== undefined && <div className="pill">Длина маршрута: {distanceKm.toFixed(1)} км</div>}
        {waitingSeconds !== undefined && <div className="pill">Время ожидания: {Math.round(waitingSeconds)} сек</div>}
        {waitingSeconds === undefined && <div className="pill">Время ожидания: —</div>}
      </div>

      {data.hints && data.hints.length > 0 && (
        <div className="hints">
          {data.hints.map((hint, idx) => (
            <div className="hint" key={idx}>
              {hint}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

