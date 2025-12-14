async function getPrediction() {
  try {
    let data = {};

    const fields = [
      "Chest_Pain", "Shortness_of_Breath", "Fatigue",
      "Palpitations", "Dizziness", "Swelling",
      "Pain_Arms_Jaw_Back", "Cold_Sweats_Nausea",
      "High_BP", "High_Cholesterol", "Diabetes",
      "Smoking", "Obesity", "Sedentary_Lifestyle",
      "Family_History", "Chronic_Stress", "Gender", "Age"
    ];

    // Collect inputs with validation
    for (let f of fields) {
      let value = document.getElementById(f).value;

      if (value === "") {
        alert(`Please fill the field: ${f.replace(/_/g, " ")}`);
        return;
      }

      data[f] = Number(value);
    }

    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data)
    });

    if (!response.ok) {
      throw new Error("Server error");
    }

    const result = await response.json();

    // Clamp score between 0–100
    let score = Math.min(Math.max(result.score, 0), 100);

    document.getElementById("result").innerText =
      `Heart Risk Score: ${score}%`;

    const meter = document.getElementById("meter-fill");
    meter.style.width = score + "%";

    meter.style.background =
      score < 30 ? "#00ff88" :
      score < 60 ? "#ffcc00" :
      "#ff4444";

  } catch (error) {
    alert("Prediction failed. Please try again.");
    console.error(error);
  }
}
