import type { RouteResponse } from "../types";

interface Props {
  data: RouteResponse | null;
}

export function RouteSummary({ data }: Props) {
  if (!data) {
    return null;
  }

  const { eta_minutes, recommended_departure, hints } = data;
  const hasDistance = data.distance_km !== undefined;
  const hasDuration = data.duration_minutes !== undefined;
  const hasSpeed = data.avg_speed_kmh !== undefined;

  return (
    <div className="card">
      <div className="summary">
        <div className="pill">В пути: ~{Math.round(eta_minutes)} мин</div>
        {hasDuration && <div className="pill">Длительность: {Math.round(data.duration_minutes!)} мин</div>}
        {hasDistance && <div className="pill">Дистанция: {data.distance_km?.toFixed(1)} км</div>}
        {hasSpeed && <div className="pill">Ср. скорость: {data.avg_speed_kmh?.toFixed(1)} км/ч</div>}
        {recommended_departure && (
          <div className="pill">
            Стартуй в: {new Date(recommended_departure).toLocaleTimeString("ru-RU", { hour: "2-digit", minute: "2-digit" })}
          </div>
        )}
      </div>

      {hints && hints.length > 0 && (
        <div className="hints">
          {hints.map((hint, idx) => (
            <div className="hint" key={idx}>
              {hint}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

