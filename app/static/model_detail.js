(function () {
  var root = document.getElementById("model-detail-root");
  if (!root) {
    return;
  }

  function formatCount(value) {
    var numeric = Number(value);
    if (!Number.isFinite(numeric)) {
      return "-";
    }
    if (Math.abs(numeric - Math.round(numeric)) < 0.0000001) {
      return String(Math.round(numeric)).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
    }
    return numeric.toFixed(2).replace(/\.?0+$/, "");
  }

  function writeClipboardText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      return navigator.clipboard.writeText(text);
    }
    return new Promise(function (resolve, reject) {
      try {
        var helper = document.createElement("textarea");
        helper.value = text;
        helper.setAttribute("readonly", "readonly");
        helper.style.position = "fixed";
        helper.style.top = "-1000px";
        helper.style.left = "-1000px";
        document.body.appendChild(helper);
        helper.focus();
        helper.select();
        var ok = document.execCommand("copy");
        document.body.removeChild(helper);
        if (ok) {
          resolve();
        } else {
          reject(new Error("Copy command failed"));
        }
      } catch (err) {
        reject(err);
      }
    });
  }

  Array.prototype.slice.call(document.querySelectorAll("[data-copy-target]")).forEach(function (button) {
    button.addEventListener("click", function () {
      var targetId = String(button.getAttribute("data-copy-target") || "");
      if (!targetId) {
        return;
      }
      var target = document.getElementById(targetId);
      if (!target) {
        return;
      }
      var textToCopy = String(target.textContent || "").trim();
      if (!textToCopy || textToCopy === "-") {
        return;
      }
      var originalLabel = button.textContent;
      writeClipboardText(textToCopy)
        .then(function () {
          button.textContent = "Copied";
          window.setTimeout(function () {
            button.textContent = originalLabel;
          }, 1400);
        })
        .catch(function () {
          button.textContent = "Copy failed";
          window.setTimeout(function () {
            button.textContent = originalLabel;
          }, 1400);
        });
    });
  });

  var matrixTable = document.getElementById("confusion-matrix-table");
  var normalizeToggle = document.getElementById("confusion-normalize-toggle");
  if (!matrixTable) {
    return;
  }

  var heatmapCells = Array.prototype.slice.call(matrixTable.querySelectorAll("[data-cm-cell]"));
  function applyConfusionMatrixView() {
    var normalize = Boolean(normalizeToggle && normalizeToggle.checked);
    heatmapCells.forEach(function (cell) {
      var count = Number(cell.getAttribute("data-count")) || 0;
      var rowTotal = Number(cell.getAttribute("data-row-total")) || 0;
      var rawIntensity = Number(cell.getAttribute("data-raw-intensity")) || 0;
      var ratio = rowTotal > 0 ? (count / rowTotal) : 0;
      var intensity = normalize ? ratio : rawIntensity;

      var display = normalize ? (ratio * 100).toFixed(1) + "%" : formatCount(count);
      var alpha = Math.max(0.08, Math.min(0.75, 0.08 + (intensity * 0.67)));
      cell.style.backgroundColor = "rgba(11, 92, 171, " + alpha.toFixed(3) + ")";
      cell.style.color = intensity > 0.5 ? "#ffffff" : "#1f2937";
      cell.textContent = display;

      var trueLabel = String(cell.getAttribute("data-true-label") || "");
      var predLabel = String(cell.getAttribute("data-pred-label") || "");
      var aria = normalize
        ? ("True " + trueLabel + ", predicted " + predLabel + ", " + display)
        : ("True " + trueLabel + ", predicted " + predLabel + ", count " + display);
      cell.setAttribute("aria-label", aria);
    });
  }

  if (normalizeToggle) {
    normalizeToggle.addEventListener("change", applyConfusionMatrixView);
  }
  applyConfusionMatrixView();
})();
