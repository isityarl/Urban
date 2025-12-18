import { MapContainer, Marker, Polyline, TileLayer, Tooltip, useMapEvents } from "react-leaflet";
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
  initialCenter?: LatLngExpression;
  initialZoom?: number;
  onMapClick?: (coords: { lat: number; lon: number }) => void;
  pointA?: LatLngExpression;
  pointB?: LatLngExpression;
}

// Default to Almaty city center
const DEFAULT_CENTER: LatLngExpression = [43.238949, 76.889709];
const DEFAULT_ZOOM = 12;

function MapClickHandler({ onMapClick }: { onMapClick?: (coords: { lat: number; lon: number }) => void }) {
  useMapEvents({
    click: (e) => onMapClick?.({ lat: e.latlng.lat, lon: e.latlng.lng })
  });
  return null;
}

export function MapView({
  polyline,
  start,
  end,
  loading,
  fallbackStart,
  fallbackEnd,
  routeSummary,
  initialCenter,
  initialZoom,
  onMapClick,
  pointA,
  pointB
}: Props) {
  const resolvedStart = start ?? polyline?.[0] ?? fallbackStart;
  const resolvedEnd = end ?? (polyline && polyline.length > 1 ? polyline[polyline.length - 1] : undefined) ?? fallbackEnd;
  const center = resolvedStart ?? polyline?.[0] ?? initialCenter ?? DEFAULT_CENTER;
  const zoom = initialZoom ?? DEFAULT_ZOOM;

  return (
    <div className="map-wrapper">
      {loading && (
        <div className="map-overlay">
          <div className="spinner" />
          <span>Выполняется симуляция трафика…</span>
        </div>
      )}
      <MapContainer center={center} zoom={zoom} scrollWheelZoom className="map-viewport">
        <MapClickHandler onMapClick={onMapClick} />
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

        {pointA && (
          <Marker position={pointA}>
            <Tooltip direction="top" offset={[0, -12]} opacity={1} permanent>
              A
            </Tooltip>
          </Marker>
        )}

        {pointB && (
          <Marker position={pointB}>
            <Tooltip direction="top" offset={[0, -12]} opacity={1} permanent>
              B
            </Tooltip>
          </Marker>
        )}
      </MapContainer>
    </div>
  );
}

