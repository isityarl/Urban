# UrbanIQ Backend API Documentation

## 1. Overview

The UrbanIQ Backend provides traffic simulation and comparison of different traffic signal control strategies using the SUMO simulator.

It allows users to:

- Submit geographic coordinates (start + destination)
- Automatically generate a SUMO trip/route from those coordinates
- Run multiple traffic control strategies:
- DQN
- PPO
- Fixed-time baseline
- Retrieve aggregated simulation statistics

The backend is implemented with FastAPI and served with Uvicorn.

## 2. Base URL

`localhost:8000` or custom port


## 3. Authentication

No authentication is required.

All endpoints are publicly accessible (intended for local/academic use).

## 4. Endpoints

### 4.1 Health / Input Format

- Method: `GET`
- Path: `/simulate_compare`

Returns a short description of the expected input format and confirms coordinate-based routing support. This is useful for checking that the backend is running and ready.

#### Response (example)

```text
{
    "message": "Send start and destination coordinates to generate a SUMO route and run simulations.",
    "input_format": {
      "from_coord": "[latitude, longitude]",
      "to_coord": "[latitude, longitude]"
    }
}
```

### 4.2 Simulate and Compare Strategies

- Method: `POST`
- Path: `/simulate_compare`
- Content-Type: `application/json`

Generates a traffic route based on provided coordinates, executes SUMO simulations, and compares multiple traffic control strategies.

#### Request body

```text
{
  "from_coord": [43.238949, 76.889709],
  "to_coord": [43.256670, 76.92861]
}
```

#### Request parameters

| Field | Type | Description |
|---|---|---|
| from_coord | array of float | Start coordinates \([latitude, longitude]\) |
| to_coord | array of float | Destination coordinates \([latitude, longitude]\) |

#### Processing steps

- Resolve start and destination coordinates to the nearest SUMO road edges
- Append a newly generated trip to the SUMO trips file
- Run simulations using:
- DQN-based control
- PPO-based control
- Fixed-time baseline control
- Parse SUMO outputs/logs
- Aggregate metrics and return structured results

#### Response

```text
{
  "from_edge": "edge_12345",
  "to_edge": "edge_67890",
  "results": {
    "DQN_all": {
      "avg_speed": 10.12,
      "waiting_time": 65.4,
      "time_loss": 140.3
    },
  "DQN_spec": {
    "vehicle_id": "specialveh",
    "travel_time": 260.1
    },
  "PPO_all": {
    "avg_speed": 11.03,
    "waiting_time": 58.9,
    "time_loss": 132.7
    },
  "PPO_spec": {
    "vehicle_id": "specialveh",
    "travel_time": 245.6
    },
  "fixed_all": {
    "avg_speed": 9.84,
    "waiting_time": 72.1,
    "time_loss": 155.8
    },
  "fixed_spec": {
    "vehicle_id": "specialveh",
    "travel_time": 280.4
    }
  }
}
```

- Note: Returned metrics depend on SUMO configuration, output files, and parsing logic.

## 5. Request and Response Format

- Format: JSON
- Encoding: UTF-8
- Header: `Content-Type: application/json`

## 6. Error Handling

### Possible error codes

| Code | Meaning | Description |
|---:|---|---|
| 400 | Bad Request | Invalid, malformed, or missing coordinates |
| 500 | Internal Server Error | SUMO execution failure or output parsing error |

## 7. Static Frontend Serving

### UI

- Method: `GET`
- Path: `/ui`

Serves the frontend application.

- Static files are served from: `front/`
- The UI is accessible via a web browser at:

`localhost:8000/ui`


## 8. API Success Criteria

A request is considered successful when:

- Coordinates map correctly to SUMO edges
- A route/trip is generated and reused consistently across strategy runs
- All traffic control strategies run without errors
- Aggregated simulation statistics return in a structured JSON response
- No orphan SUMO processes remain after completion

## 9. Future Extensions

Potential improvements:

- Separate endpoints per algorithm (DQN-only, PPO-only, baseline-only)
- Batch route simulation
- Asynchronous execution with progress tracking (job queue)
- Authentication and access control
