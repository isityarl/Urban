import { MapContainer, Marker, Polyline, TileLayer, Tooltip } from "react-leaflet";
import type { LatLngExpression } from "leaflet";
import "leaflet/dist/leaflet.css";

interface Props {
  polyline?: LatLngExpression[];
  start?: LatLngExpression;
  end?: LatLngExpression;
  loading?: boolean;
  fallbackStart?: LatLngExpression;
  fallbackEnd?: LatLngExpression;
  routeSummary?: string;
}

// Default to Almaty city center
const DEFAULT_CENTER: LatLngExpression = [43.238949, 76.889709];

export function MapView({ polyline, start, end, loading, fallbackStart, fallbackEnd, routeSummary }: Props) {
  const resolvedStart = start ?? polyline?.[0] ?? fallbackStart;
  const resolvedEnd = end ?? (polyline && polyline.length > 1 ? polyline[polyline.length - 1] : undefined) ?? fallbackEnd;
  const center = resolvedStart ?? polyline?.[0] ?? DEFAULT_CENTER;

  return (
    <div className="map-wrapper">
      {loading && (
        <div className="map-overlay">
          <div className="spinner" />
          <span>Считаем маршрут...</span>
        </div>
      )}
      <MapContainer center={center} zoom={12} scrollWheelZoom className="map-viewport">
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {polyline && (
          <Polyline positions={polyline} color="#2563eb" weight={6} opacity={0.7}>
            {routeSummary && (
              <Tooltip sticky direction="top">
                {routeSummary}
              </Tooltip>
            )}
          </Polyline>
        )}

        {resolvedStart && (
          <Marker position={resolvedStart}>
            <Tooltip direction="top" offset={[0, -12]} opacity={1} permanent>
              Старт
            </Tooltip>
          </Marker>
        )}

        {resolvedEnd && (
          <Marker position={resolvedEnd}>
            <Tooltip direction="top" offset={[0, -12]} opacity={1} permanent>
              Финиш
            </Tooltip>
          </Marker>
        )}
      </MapContainer>
    </div>
  );
}

