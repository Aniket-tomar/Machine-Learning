// =====================================================================
// Pest Detection — Frontend Logic
// =====================================================================
(() => {
  "use strict";

  // --- DOM refs -------------------------------------------------------
  const dropzone       = document.getElementById("dropzone");
  const dropContent    = document.getElementById("dropzone-content");
  const fileInput      = document.getElementById("file-input");
  const previewImg     = document.getElementById("preview-img");
  const predictBtn     = document.getElementById("predict-btn");
  const btnText        = predictBtn.querySelector(".btn-text");
  const btnLoader      = document.getElementById("btn-loader");
  const resultsCard    = document.getElementById("results-card");
  const resultTop      = document.getElementById("result-top");
  const resultBars     = document.getElementById("result-bars");
  const resetBtn       = document.getElementById("reset-btn");

  let selectedFile = null;

  // Pest emoji map for a bit of fun
  const PEST_EMOJI = {
    "Ants":        "🐜",
    "Bees":        "🐝",
    "Beetle":      "🪲",
    "Caterpillar": "🐛",
    "Earthworms":  "🪱",
    "Earwig":      "🦗",
    "Grasshopper": "🦗",
    "Moth":        "🦋",
    "Slug":        "🐌",
    "Snail":       "🐌",
    "Wasp":        "🐝",
    "Weevil":      "🪲",
  };

  // --- Helpers --------------------------------------------------------
  function showPreview(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      previewImg.classList.remove("hidden");
      dropContent.classList.add("hidden");
    };
    reader.readAsDataURL(file);
    selectedFile = file;
    predictBtn.disabled = false;
  }

  function resetUI() {
    selectedFile = null;
    predictBtn.disabled = true;
    btnText.textContent = "Analyze Image";
    btnLoader.classList.add("hidden");
    previewImg.classList.add("hidden");
    previewImg.src = "";
    dropContent.classList.remove("hidden");
    resultsCard.classList.add("hidden");
    resultTop.innerHTML = "";
    resultBars.innerHTML = "";
  }

  // --- Drag & Drop ----------------------------------------------------
  ["dragenter", "dragover"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      showPreview(file);
    }
  });

  dropzone.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) {
      showPreview(fileInput.files[0]);
    }
  });

  // --- Predict --------------------------------------------------------
  predictBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    // Show loading state
    predictBtn.disabled = true;
    btnText.textContent = "Analyzing…";
    btnLoader.classList.remove("hidden");
    resultsCard.classList.add("hidden");

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const resp = await fetch("/predict", { method: "POST", body: formData });
      const data = await resp.json();

      if (data.error) {
        alert("Error: " + data.error);
        return;
      }

      renderResults(data.predictions);
    } catch (err) {
      alert("Something went wrong. Is the server running?");
      console.error(err);
    } finally {
      btnText.textContent = "Analyze Image";
      btnLoader.classList.add("hidden");
      predictBtn.disabled = false;
    }
  });

  // --- Render results -------------------------------------------------
  function renderResults(predictions) {
    if (!predictions || !predictions.length) return;

    const top = predictions[0];
    const emoji = PEST_EMOJI[top.label] || "🔍";

    // Top prediction card
    resultTop.innerHTML = `
      <div class="result-top-icon">${emoji}</div>
      <div class="result-top-info">
        <div class="result-top-label">${top.label}</div>
        <div class="result-top-confidence">${top.confidence}% confidence</div>
        <div class="result-top-desc">${top.description}</div>
      </div>
    `;

    // Confidence bars
    resultBars.innerHTML = "";
    predictions.forEach((p, i) => {
      const row = document.createElement("div");
      row.className = "bar-row";
      row.innerHTML = `
        <span class="bar-label">${p.label}</span>
        <div class="bar-track">
          <div class="bar-fill" style="width: 0%"></div>
        </div>
        <span class="bar-value">${p.confidence}%</span>
      `;
      resultBars.appendChild(row);

      // Animate bar after a short stagger
      requestAnimationFrame(() => {
        setTimeout(() => {
          row.querySelector(".bar-fill").style.width = p.confidence + "%";
        }, i * 120);
      });
    });

    resultsCard.classList.remove("hidden");

    // Scroll to results
    setTimeout(() => {
      resultsCard.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 200);
  }

  // --- Reset ----------------------------------------------------------
  resetBtn.addEventListener("click", resetUI);
})();
