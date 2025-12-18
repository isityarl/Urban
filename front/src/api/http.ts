import axios from "axios";

export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "").trim() || "/api";

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: { "Content-Type": "application/json" }
});

