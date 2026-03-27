const form = document.getElementById("predictionForm");
const resultCard = document.getElementById("resultCard");
const predictBtn = document.getElementById("predictBtn");

function renderResult(data) {
  const explanationList = data.explanation
    .map((item) => `<li>${item}</li>`)
    .join("");

  resultCard.innerHTML = `
    <h2>Prediction Result</h2>
    <p class="metric">${data.eta_minutes.toFixed(1)} mins</p>
    <p class="meta"><strong>Window:</strong> ${data.eta_window_min.toFixed(1)} - ${data.eta_window_max.toFixed(1)} mins</p>
    <p class="meta"><strong>Distance:</strong> ${data.distance_km.toFixed(1)} km</p>
    <p class="meta"><strong>Estimated delivery (UTC):</strong> ${new Date(data.estimated_delivery_time_utc).toLocaleString()}</p>
    <p class="meta"><strong>Model:</strong> ${data.model_name}</p>
    <ul class="explain">${explanationList}</ul>
  `;
}

function renderError(message) {
  resultCard.innerHTML = `
    <h2>Prediction Result</h2>
    <div class="error">${message}</div>
  `;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  predictBtn.disabled = true;
  predictBtn.textContent = "Predicting...";

  const formData = new FormData(form);
  const payload = Object.fromEntries(formData.entries());

  payload.pickup_lat = Number(payload.pickup_lat);
  payload.pickup_lng = Number(payload.pickup_lng);
  payload.drop_lat = Number(payload.drop_lat);
  payload.drop_lng = Number(payload.drop_lng);
  payload.package_weight_kg = Number(payload.package_weight_kg);
  payload.warehouse_load_pct = Number(payload.warehouse_load_pct);

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const detail = await response.text();
      throw new Error(detail || "Prediction failed.");
    }

    const data = await response.json();
    renderResult(data);
  } catch (error) {
    renderError(error.message || "Something went wrong.");
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = "Predict Delivery Time";
  }
});
