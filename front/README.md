# Urban Frontend (web)

Single-page app on React + Vite + TypeScript with OSM (Leaflet) to request a route and show ETA / hints from the backend (Flask).

Есть поиск адресов по OSM/Nominatim (дебаунс, подсказки); можно вводить адрес или `lat,lon`.

## Быстрый старт

```bash
cd web
npm install
npm run dev
```

Открой `http://localhost:5173`.

## Настройки

Скопируй `env.sample` → `.env` и укажи URL бэкенда:

```
VITE_API_BASE_URL=http://localhost:5000
```

## API контракт (ожидаемый)

POST `/simulate`

```json
{
  "origin": {"lat": 43.238949, "lon": 76.889709},
  "destination": {"lat": 43.256700, "lon": 76.928600},
  "preferred_departure_time": "2025-12-10T15:00:00+05:00"
}
```

Ответ:

```json
{
  "task_id": null,
  "status": "finished",
  "optimized_route": [
    {"lat": 43.23895, "lon": 76.88971, "action": "start"},
    {"lat": 43.25670, "lon": 76.92860, "action": "arrive"}
  ],
  "recommended_departure_time": "2025-12-10T14:50:00+05:00",
  "estimated_travel_time_seconds": 1620,
  "meta": {
    "note": "sync fallback result from prototype",
    "distance_km": 14.2,
    "avg_speed_kmph": 40
  }
}
```

Фронтенд транслирует `optimized_route` в polyline, ETA в минутах и показывает `meta.note` как подсказку. Время — ISO8601 со смещением GMT+5. Если используешь прокси Vite, API доступен как `/api/...` и идёт на `localhost:5500`.

