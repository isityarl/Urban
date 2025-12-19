const API_URL = "http://127.0.0.1:8000/simulate_compare";

const map = L.map("map").setView([43.2389, 76.8897], 13);

L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  attribution: "© OpenStreetMap",
}).addTo(map);

let points = [];
let markers = [];

map.on("click", (e) => {
  if (points.length >= 2) return;

  points.push(e.latlng);
  const marker = L.marker(e.latlng).addTo(map);
  markers.push(marker);

  marker.bindPopup(points.length === 1 ? "Point A" : "Point B").openPopup();
});

document.getElementById("runBtn").onclick = async () => {
  const output = document.getElementById("output");

  if (points.length < 2) {
    output.textContent = "⚠️ Select Point A and Point B";
    return;
  }

  output.textContent = "⏳ Running simulation...";

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        from_coord: [points[0].lat, points[0].lng],
        to_coord: [points[1].lat, points[1].lng],
      }),
    });

    const data = await res.json();
    output.textContent = JSON.stringify(data, null, 2);

  } catch (err) {
    output.textContent = `❌ Error: ${err.message}`;
  }
};
