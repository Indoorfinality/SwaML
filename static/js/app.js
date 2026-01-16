let selectedFile = null;
let detectedTarget = null;
let allColumns = [];

async function startTraining() {
  const fileInput = document.getElementById("csvFile");
  const file = fileInput.files[0];
  if (!file) {
    alert("Please select a CSV file first!");
    return;
  }

  // Store file for later use
  selectedFile = file;

  // Show loading
  document.getElementById("loading").style.display = "block";
  document.getElementById("result").innerHTML = "";
  document.getElementById("targetSelection").style.display = "none";

  const formData = new FormData();
  formData.append("csv_files", file);

  try {
    // First, detect target
    const detectResponse = await fetch("/api/detect-target/", {
      method: "POST",
      body: formData,
    });

    const detectData = await detectResponse.json();
    document.getElementById("loading").style.display = "none";

    if (detectData.error) {
      document.getElementById("result").innerHTML = `
        <p class="error">Error: ${detectData.error}</p>
      `;
    } else {
      detectedTarget = detectData.detected_target;
      allColumns = detectData.all_columns;

      // Show target selection UI
      showTargetSelection(detectData.detected_target, detectData.all_columns);
    }
  } catch (err) {
    document.getElementById("loading").style.display = "none";
    document.getElementById("result").innerHTML = `
      <p class="error">Network error: ${err.message}</p>
    `;
  }
}

function showTargetSelection(detectedTarget, allColumns) {
  const targetSelectionDiv = document.getElementById("targetSelection");
  const selectHtml = allColumns
    .map(
      (col) =>
        `<option value="${col}" ${
          col === detectedTarget ? "selected" : ""
        }>${col}</option>`
    )
    .join("");

  targetSelectionDiv.innerHTML = `
    <div class="target-selection">
      <h3>Target Column Selection</h3>
      <p>Detected target column: <strong>${detectedTarget}</strong></p>
      <label for="targetSelect">Select or confirm target column:</label>
      <select id="targetSelect" style="width: 100%; padding: 8px; margin: 10px 0; font-size: 14px;">
        ${selectHtml}
      </select>
      <button onclick="confirmAndTrain()" style="padding: 10px 20px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 16px;">
        Confirm and Start Training
      </button>
    </div>
  `;
  targetSelectionDiv.style.display = "block";
}

async function confirmAndTrain() {
  const targetSelect = document.getElementById("targetSelect");
  const confirmedTarget = targetSelect.value;

  if (!selectedFile) {
    alert("No file selected!");
    return;
  }

  // Show loading
  document.getElementById("loading").style.display = "block";
  document.getElementById("targetSelection").style.display = "none";
  document.getElementById("result").innerHTML = "";

  const formData = new FormData();
  formData.append("csv_files", selectedFile);
  formData.append("target_column", confirmedTarget);

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
        <a href="${data.model_download_url}" download class="download-btn">
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
