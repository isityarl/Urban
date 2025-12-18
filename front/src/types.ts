export interface RouteRequest {
  from: string; // "lat,lon"
  to: string; // "lat,lon"
  departure_time: string; // ISO8601 with offset, e.g. 2025-12-10T15:00:00+05:00
  // Temporary mock fields for backend integration (Task 5)
  from_edge?: string;
  to_edge?: string;
}

export interface BackendRoutePoint {
  lat: number;
  lon: number;
  action?: string;
}

export interface BackendRouteResponse {
  task_id?: string | null;
  status?: string;
  optimized_route?: BackendRoutePoint[];
  recommended_departure_time?: string;
  estimated_travel_time_seconds?: number;
  meta?: {
    note?: string;
    distance_m?: number;
    distance_km?: number;
    avg_speed_kmh?: number;
    avg_speed_kmph?: number; // backward compatibility
  };
  steps?: TraceStep[];
}

export interface RouteResponse {
  eta_minutes: number;
  travel_time_seconds?: number;
  recommended_departure?: string;
  polyline?: [number, number][];
  hints?: string[];
  distance_m?: number;
  distance_km?: number;
  duration_minutes?: number;
  avg_speed_kmh?: number;
  waiting_time_seconds?: number;
  steps?: TraceStep[];
}

export interface TraceStep {
  t: number;
  vehicle?: {
    id: string;
    lat: number;
    lon: number;
    speed_kmh?: number;
  } | null;
  tls?: { id: string; state: string }[];
}

