async function startTraining() {
  const fileInput = document.getElementById("csvFile");
  const file = fileInput.files[0];
  if (!file) {
    alert("Please select a CSV file first!");
    return;
  }

  // Show loading
  document.getElementById("loading").style.display = "block";
  document.getElementById("result").innerHTML = "";

  const formData = new FormData();
  formData.append("csv_files", file);

  try {
    const response = await fetch("/api/upload/", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    document.getElementById("loading").style.display = "none";

    if (data.error) {
      document.getElementById("result").innerHTML = `
                <p class="error">Error: ${data.error}</p>
            `;
    } else {
      document.getElementById("result").innerHTML = `
                <h2>Training Complete!</h2>
                <p><strong>Best Model:</strong> ${data.best_model}</p>
                <h3>Results:</h3>
                <pre>${JSON.stringify(data.results, null, 2)}</pre>
                <a href="${
                  data.model_download_url
                }" download class="download-btn">
                    Download Trained Model
                </a>
                ${
                  data.feature_plot_url
                    ? `
                    <h3>Feature Importance</h3>
                    <img src="${data.feature_plot_url}" alt="Feature Importance" style="max-width:100%;">
                `
                    : ""
                }
            `;
    }
  } catch (err) {
    document.getElementById("loading").style.display = "none";
    document.getElementById("result").innerHTML = `
            <p class="error">Network error: ${err.message}</p>
        `;
  }
}
