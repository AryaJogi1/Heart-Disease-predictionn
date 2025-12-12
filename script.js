async function getPrediction() {

    let data = {};

    const fields = [
        "Chest_Pain", "Shortness_of_Breath", "Fatigue",
        "Palpitations", "Dizziness", "Swelling",
        "Pain_Arms_Jaw_Back", "Cold_Sweats_Nausea",
        "High_BP", "High_Cholesterol", "Diabetes",
        "Smoking", "Obesity", "Sedentary_Lifestyle",
        "Family_History", "Chronic_Stress", "Gender", "Age"
    ];

    fields.forEach(f => {
        data[f] = Number(document.getElementById(f).value);
    });

    let response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
    });

    let result = await response.json();

    document.getElementById("result").innerText =
        `Heart Risk Score: ${result.score}%`;

    document.getElementById("meter-fill").style.width = result.score + "%";

    if (result.score < 30) {
        document.getElementById("meter-fill").style.background = "#00ff88";
    } else if (result.score < 70) {
        document.getElementById("meter-fill").style.background = "#ffcc00";
    } else {
        document.getElementById("meter-fill").style.background = "#ff4444";
    }
}
