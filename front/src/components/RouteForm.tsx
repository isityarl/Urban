import { useState } from "react";
import type { FormEvent } from "react";
import { GeocoderInput } from "./GeocoderInput";
import type { RouteRequest } from "../types";

interface Props {
  onSubmit: (payload: RouteRequest) => Promise<void> | void;
  loading?: boolean;
  onReset?: () => void;
}

const TZ_OFFSET = "+05:00";

function formatLocalInput(date: Date) {
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, "0");
  const d = String(date.getDate()).padStart(2, "0");
  const h = String(date.getHours()).padStart(2, "0");
  const min = String(date.getMinutes()).padStart(2, "0");
  return `${y}-${m}-${d}T${h}:${min}`;
}

export function RouteForm({ onSubmit, loading, onReset }: Props) {
  const defaultValue = formatLocalInput(new Date(Date.now() + 15 * 60 * 1000));

  // Default points near Almaty for quick testing
  const defaultFrom = "43.238949,76.889709";
  const defaultTo = "43.256700,76.928600";

  const [fromInput, setFromInput] = useState(defaultFrom);
  const [toInput, setToInput] = useState(defaultTo);
  const [fromCoords, setFromCoords] = useState(defaultFrom);
  const [toCoords, setToCoords] = useState(defaultTo);
  const [departure, setDeparture] = useState(defaultValue);

  const ensureCoords = (input: string, coords: string) => {
    const parts = input.split(",").map((p) => p.trim());
    if (parts.length === 2 && parts.every((p) => !Number.isNaN(Number(p)))) {
      return `${Number(parts[0])},${Number(parts[1])}`;
    }
    return coords;
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!fromInput.trim() || !toInput.trim()) {
      return;
    }
    const fromVal = ensureCoords(fromInput, fromCoords);
    const toVal = ensureCoords(toInput, toCoords);
    await onSubmit({
      from: fromVal,
      to: toVal,
      // Send explicit GMT+5 offset
      departure_time: `${departure}:00${TZ_OFFSET}`
    });
  };

  const handleReset = (e: FormEvent) => {
    e.preventDefault();
    setFromInput(defaultFrom);
    setToInput(defaultTo);
    setFromCoords(defaultFrom);
    setToCoords(defaultTo);
    setDeparture(formatLocalInput(new Date(Date.now() + 15 * 60 * 1000)));
    onReset?.();
  };

  return (
    <form className="form-grid" onSubmit={handleSubmit}>
      <GeocoderInput
        label="Откуда (адрес или lat,lon)"
        value={fromInput}
        onChange={setFromInput}
        onSelect={({ lat, lon, label }) => {
          setFromInput(label);
          setFromCoords(`${lat},${lon}`);
        }}
        placeholder="Например, Алматы, Абая 10 или 43.2389,76.8897"
      />

      <GeocoderInput
        label="Куда (адрес или lat,lon)"
        value={toInput}
        onChange={setToInput}
        onSelect={({ lat, lon, label }) => {
          setToInput(label);
          setToCoords(`${lat},${lon}`);
        }}
        placeholder="Например, Байтурсынова 50 или 43.2567,76.9286"
      />

      <div className="field">
        <label className="label" htmlFor="departure">
          Время выезда
        </label>
        <input
          id="departure"
          type="datetime-local"
          className="datetime"
          value={departure}
          onChange={(e) => setDeparture(e.target.value)}
          required
        />
      </div>

      <button className="button" type="submit" disabled={loading}>
        {loading ? "Считаем маршрут..." : "Построить маршрут"}
      </button>
      <button className="button secondary" type="button" onClick={handleReset} disabled={loading}>
        Сбросить
      </button>
    </form>
  );
}

