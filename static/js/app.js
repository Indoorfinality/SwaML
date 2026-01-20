document.addEventListener("DOMContentLoaded", function () {
    // State
    let uploadedFilename = null;
    let detectedTarget = null;
  
    // Elements
    const els = {
      startBtn: document.getElementById("startBtn"),
      confirmTrainBtn: document.getElementById("confirmTrainBtn"),
      csvFile: document.getElementById("csvFile"),
      loading: document.getElementById("loading"),
      initialPreview: document.getElementById("initialPreview"),
      targetConfirm: document.getElementById("targetConfirm"),
      uploadInfo: document.getElementById("uploadInfo"),
      uploadedFileName: document.getElementById("uploadedFileName"),
      detectedTargetText: document.getElementById("detectedTargetText"),
      targetSelect: document.getElementById("targetSelect"),
      trainResult: document.getElementById("trainResult"),
      showDatasetBtn: document.getElementById("showDatasetBtn"),
      datasetResult: document.getElementById("datasetResult"),
      rowLimit: document.getElementById("rowLimit"),
      offset: document.getElementById("offset"),
    };
  
    // --- 1. Upload & Analyze Handler ---
    // --- 1. Upload & Analyze Handler ---
    if (els.startBtn) {
      els.startBtn.addEventListener("click", async () => {
        const file = els.csvFile?.files[0];
        if (!file) return alert("Please select a CSV file first!");
  
        // UI Reset
        toggleLoading(true);
        els.initialPreview.innerHTML = "";
        els.trainResult.innerHTML = "";
        hide(els.targetConfirm);
        hide(els.uploadInfo);
  
        const uploadFormData = new FormData();
        uploadFormData.append("csv_files", file); // Must match Django's request.FILES key
  
        try {
          // --- Step A: Upload File ---
          const uploadRes = await fetch("/api/upload/", {
            method: "POST",
            body: uploadFormData,
          });
          
          if (!uploadRes.ok) {
             const errText = await uploadRes.text();
             throw new Error(`Upload failed (${uploadRes.status}): ${errText}`);
          }

          const uploadData = await uploadRes.json();
  
          if (uploadData.status === "error") {
            // Note: Our Upload serializer might not have "status" field, check response structure
            // But assume standard error handling if added
            throw new Error(uploadData.message || "Upload failed");
          }

          // Success: Get filename
          uploadedFilename = uploadData.uploaded_filename;

          // Update UI with file info
          els.uploadedFileName.textContent = uploadedFilename;
          show(els.uploadInfo);
          
          // --- Step B: Analyze Target ---
          const analyzeFormData = new FormData();
          analyzeFormData.append("uploaded_filename", uploadedFilename);

          const analyzeRes = await fetch("/api/analyze-target/", {
            method: "POST",
            body: analyzeFormData,
          });
          const analyzeData = await analyzeRes.json();
  
          toggleLoading(false);
  
          if (analyzeData.status === "error") {
            els.initialPreview.innerHTML = `<div class="error">Analysis Error: ${analyzeData.message}</div>`;
            return;
          }
  
          // Update State
          detectedTarget = analyzeData.detected_target;
  
          // Populate Target Select
          updateTargetSelection(analyzeData.all_columns, detectedTarget);
  
          // Show Preview (Optional, can be parallelized but keeping simple)
          await showPreview();
  
          // Show Confirm Section
          show(els.targetConfirm);
  
        } catch (err) {
          toggleLoading(false);
          els.initialPreview.innerHTML = `<div class="error">Detailed Error: ${err.message}</div>`;
        }
      });
    }
  
    // --- 2. Confirm & Train Handler ---
    // Moved outside to prevent nested listener duplication
    if (els.confirmTrainBtn) {
      els.confirmTrainBtn.addEventListener("click", async () => {
        if (!uploadedFilename) return alert("No file uploaded!");
  
        const selectedTarget = els.targetSelect.value;
        
        // Show local loading state in result area
        els.trainResult.innerHTML = `<div class="loading">Training model... This might take a moment.</div>`;
        
        const trainForm = new FormData();
        trainForm.append("uploaded_filename", uploadedFilename);
        trainForm.append("target_column", selectedTarget);
  
        try {
          const res = await fetch("/api/start-training/", {
            method: "POST",
            body: trainForm,
          });
          const data = await res.json();
  
          if (data.status === "error") {
            els.trainResult.innerHTML = `<div class="error">${data.message}</div>`;
          } else {
            els.trainResult.innerHTML = `
              <div class="success-box">
                <h2>🎉 Training Complete!</h2>
                <p style="font-size: 1.1em; margin: 10px 0;"><strong>Best Model:</strong> ${data.best_model}</p>
                <div style="background: white; padding: 15px; border-radius: 8px; border: 1px solid #eee; margin: 15px 0;">
                    <pre style="overflow: auto; max-height: 300px;">${JSON.stringify(data.results, null, 2)}</pre>
                </div>
                <a href="${data.model_download_url}" download class="download-btn">Download Trained Model</a>
              </div>
            `;
          }
        } catch (err) {
          els.trainResult.innerHTML = `<div class="error">Training failed: ${err.message}</div>`;
        }
      });
    }
  
    // --- 3. Dataset Explorer Handler ---
    if (els.showDatasetBtn) {
      els.showDatasetBtn.addEventListener("click", async () => {
        if (!uploadedFilename) return alert("Please upload a file first to explore the dataset.");
  
        const limit = els.rowLimit.value || 10;
        const offset = els.offset.value || 0;
  
        try {
          const url = `/api/preview/?dataset_name=${encodeURIComponent(uploadedFilename)}&limit=${limit}&offset=${offset}`;
          const res = await fetch(url);
          const data = await res.json();
  
          if (data.status === "error") {
            els.datasetResult.innerHTML = `<div class="error">${data.message}</div>`;
            return;
          }
  
          renderTable(data, els.datasetResult, true);
        } catch (err) {
          alert("Preview error: " + err.message);
        }
      });
    }
  
    // --- Helpers ---
  
    function updateTargetSelection(columns, detected) {
      if (!els.targetSelect) return;
      els.detectedTargetText.textContent = detected;
      
      els.targetSelect.innerHTML = columns
        .map(col => `<option value="${col}" ${col === detected ? "selected" : ""}>${col}</option>`)
        .join("");
    }
  
    async function showPreview() {
      if (!uploadedFilename) return;
      try {
        const url = `/api/preview/?dataset_name=${encodeURIComponent(uploadedFilename)}&limit=5&offset=0`;
        const res = await fetch(url);
        if (!res.ok) {
            throw new Error(`Server returned ${res.status} ${res.statusText}`);
        }
        const data = await res.json();
  
        if (data.status === "error") {
            // Silently handle error or show in specific area, but don't break
            console.warn(data.message);
            return;
        }

        renderTable(data, els.initialPreview);

      } catch (err) {
        console.warn("Auto-preview failed", err);
        els.initialPreview.innerHTML = `<div class="error">Preview failed: ${err.message}</div>`;
      }
    }
  
    function renderTable(data, container, showInfo = false) {
      if (!data || !data.columns || !data.data) {
          container.innerHTML = `<div class="error">Invalid data format for preview</div>`;
          return;
      }

      let infoHtml = "";
      if (showInfo) {
        infoHtml = `<p style="margin-bottom: 15px; color: var(--gray);">Showing rows ${data.offset} - ${data.offset + data.returned_rows} of ${data.total_rows}</p>`;
      }
  
      const html = `
        ${infoHtml}
        <table>
          <thead>
            <tr>${data.columns.map(c => `<th>${c}</th>`).join("")}</tr>
          </thead>
          <tbody>
            ${data.data.map(row => 
              `<tr>${data.columns.map(c => `<td>${row[c] ?? "-"}</td>`).join("")}</tr>`
            ).join("")}
          </tbody>
        </table>
      `;
      container.innerHTML = html;
    }
  
    function show(el) {
      if (el) el.classList.remove("hidden");
    }
  
    function hide(el) {
      if (el) el.classList.add("hidden");
    }
  
    function toggleLoading(isLoading) {
      if (isLoading) show(els.loading);
      else hide(els.loading);
    }
  });