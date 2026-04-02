(function () {
  const config = window.annotationEditorConfig;
  if (!config) {
    return;
  }

  const sourceImage = document.getElementById("annotation-source-image");
  const sourceCanvas = document.getElementById("source-canvas");
  const overlayCanvas = document.getElementById("overlay-canvas");
  const sourceContext = sourceCanvas.getContext("2d");
  const overlayContext = overlayCanvas.getContext("2d");
  const statusElement = document.getElementById("editor-status");
  const selectionElement = document.getElementById("editor-selection");
  const summaryElement = document.getElementById("revision-summary");
  const saveButton = document.getElementById("save-annotation");
  const reloadButton = document.getElementById("reload-annotation");
  const brushRadiusInput = document.getElementById("brush-radius");
  const classButtons = Array.prototype.slice.call(document.querySelectorAll(".class-button"));
  const toolButtons = Array.prototype.slice.call(document.querySelectorAll(".tool-button"));

  const classesByIndex = {};
  config.classes.forEach(function (datasetClass) {
    classesByIndex[datasetClass.class_index] = datasetClass;
  });

  const state = {
    activeClassIndex: config.classes.some(function (item) { return item.class_index === 1; }) ? 1 : 0,
    tool: "brush",
    brushRadius: Number(brushRadiusInput.value || 6),
    dirty: false,
    locked: !!config.locked,
    isPainting: false,
    labelMap: [],
    provenanceMap: [],
    protectionMap: [],
    imageWidth: 0,
    imageHeight: 0
  };

  function getColor(classIndex) {
    const datasetClass = classesByIndex[classIndex];
    return datasetClass ? datasetClass.color : "#000000";
  }

  function hexToRgb(hexColor) {
    const trimmed = String(hexColor || "#000000").replace("#", "");
    const normalized = trimmed.length === 3
      ? trimmed.split("").map(function (part) { return part + part; }).join("")
      : trimmed.padEnd(6, "0");
    return {
      r: parseInt(normalized.slice(0, 2), 16),
      g: parseInt(normalized.slice(2, 4), 16),
      b: parseInt(normalized.slice(4, 6), 16)
    };
  }

  function setStatus(message) {
    statusElement.textContent = message;
  }

  function updateSelectionSummary() {
    const datasetClass = classesByIndex[state.activeClassIndex];
    const className = datasetClass ? datasetClass.name : "unknown";
    selectionElement.textContent =
      "Current class: " + state.activeClassIndex + " (" + className + ") | Tool: " + state.tool + " | Dirty: " + (state.dirty ? "yes" : "no");
  }

  function updateRevisionSummary(revision) {
    summaryElement.textContent =
      "Revision " + revision.id + " | checksum " + revision.revision_checksum + " | updated " + revision.updated_at;
  }

  function updateToolbarState() {
    classButtons.forEach(function (button) {
      const isActive = Number(button.dataset.classIndex) === state.activeClassIndex;
      button.classList.toggle("is-active", isActive);
    });
    toolButtons.forEach(function (button) {
      const isActive = button.dataset.tool === state.tool;
      button.classList.toggle("is-active", isActive);
    });
    saveButton.disabled = state.locked;
    brushRadiusInput.disabled = state.locked;
    overlayCanvas.style.cursor = state.locked ? "not-allowed" : "crosshair";
    updateSelectionSummary();
  }

  function syncCanvasSize(width, height) {
    state.imageWidth = width;
    state.imageHeight = height;
    sourceCanvas.width = width;
    sourceCanvas.height = height;
    overlayCanvas.width = width;
    overlayCanvas.height = height;
    sourceContext.drawImage(sourceImage, 0, 0, width, height);
    drawOverlay();
  }

  function drawOverlay() {
    if (!state.labelMap.length) {
      return;
    }
    const imageData = overlayContext.createImageData(state.imageWidth, state.imageHeight);
    const data = imageData.data;
    for (let y = 0; y < state.imageHeight; y += 1) {
      for (let x = 0; x < state.imageWidth; x += 1) {
        const classIndex = state.labelMap[y][x];
        if (!classIndex) {
          continue;
        }
        const rgba = hexToRgb(getColor(classIndex));
        const offset = (y * state.imageWidth + x) * 4;
        data[offset] = rgba.r;
        data[offset + 1] = rgba.g;
        data[offset + 2] = rgba.b;
        data[offset + 3] = 140;
      }
    }
    overlayContext.putImageData(imageData, 0, 0);
  }

  function getImageCoordinates(event) {
    const rect = overlayCanvas.getBoundingClientRect();
    const scaleX = overlayCanvas.width / rect.width;
    const scaleY = overlayCanvas.height / rect.height;
    const x = Math.max(0, Math.min(state.imageWidth - 1, Math.floor((event.clientX - rect.left) * scaleX)));
    const y = Math.max(0, Math.min(state.imageHeight - 1, Math.floor((event.clientY - rect.top) * scaleY)));
    return { x: x, y: y };
  }

  function markDirty() {
    state.dirty = true;
    updateSelectionSummary();
  }

  function paintBrush(x, y) {
    let changed = false;
    const radius = Math.max(1, state.brushRadius);
    for (let dy = -radius; dy <= radius; dy += 1) {
      for (let dx = -radius; dx <= radius; dx += 1) {
        if ((dx * dx) + (dy * dy) > radius * radius) {
          continue;
        }
        const px = x + dx;
        const py = y + dy;
        if (px < 0 || py < 0 || px >= state.imageWidth || py >= state.imageHeight) {
          continue;
        }
        if (state.labelMap[py][px] !== state.activeClassIndex) {
          state.labelMap[py][px] = state.activeClassIndex;
          state.provenanceMap[py][px] = 1;
          changed = true;
        }
      }
    }
    if (changed) {
      markDirty();
      drawOverlay();
    }
  }

  function floodFill(x, y) {
    const targetClassIndex = state.labelMap[y][x];
    if (targetClassIndex === state.activeClassIndex) {
      return;
    }
    const stack = [[x, y]];
    while (stack.length) {
      const item = stack.pop();
      const px = item[0];
      const py = item[1];
      if (px < 0 || py < 0 || px >= state.imageWidth || py >= state.imageHeight) {
        continue;
      }
      if (state.labelMap[py][px] !== targetClassIndex) {
        continue;
      }
      state.labelMap[py][px] = state.activeClassIndex;
      state.provenanceMap[py][px] = 1;
      stack.push([px + 1, py]);
      stack.push([px - 1, py]);
      stack.push([px, py + 1]);
      stack.push([px, py - 1]);
    }
    markDirty();
    drawOverlay();
  }

  async function loadAnnotation() {
    setStatus("Loading annotation…");
    const response = await fetch(config.apiUrl, { credentials: "same-origin" });
    if (!response.ok) {
      const text = await response.text();
      setStatus("Failed to load annotation: " + text);
      return;
    }
    const payload = await response.json();
    state.labelMap = payload.label_map;
    state.provenanceMap = payload.provenance_map;
    state.protectionMap = payload.protection_map;
    state.dirty = false;
    updateRevisionSummary(payload.revision);
    if (sourceImage.complete) {
      syncCanvasSize(payload.revision.width, payload.revision.height);
    }
    setStatus("Annotation loaded.");
    updateToolbarState();
  }

  async function saveAnnotation() {
    if (state.locked) {
      setStatus("This revision is locked. Fork a new head before saving.");
      return;
    }
    setStatus("Saving annotation…");
    const response = await fetch(config.apiUrl, {
      method: "POST",
      credentials: "same-origin",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        author: "phase2-editor",
        operation_summary: "Saved from Phase 2 minimal editor",
        label_map: state.labelMap,
        provenance_map: state.provenanceMap,
        protection_map: state.protectionMap
      })
    });
    const payload = await response.json();
    if (!response.ok) {
      setStatus(payload.error || "Failed to save annotation.");
      return;
    }
    state.labelMap = payload.label_map;
    state.provenanceMap = payload.provenance_map;
    state.protectionMap = payload.protection_map;
    state.dirty = false;
    updateRevisionSummary(payload.revision);
    updateToolbarState();
    setStatus("Annotation saved.");
  }

  classButtons.forEach(function (button) {
    button.addEventListener("click", function () {
      state.activeClassIndex = Number(button.dataset.classIndex);
      updateToolbarState();
    });
  });

  toolButtons.forEach(function (button) {
    button.addEventListener("click", function () {
      state.tool = button.dataset.tool;
      updateToolbarState();
    });
  });

  brushRadiusInput.addEventListener("input", function () {
    state.brushRadius = Number(brushRadiusInput.value || 6);
    updateToolbarState();
  });

  reloadButton.addEventListener("click", function () {
    loadAnnotation().catch(function (error) {
      setStatus("Failed to reload annotation: " + error.message);
    });
  });

  saveButton.addEventListener("click", function () {
    saveAnnotation().catch(function (error) {
      setStatus("Failed to save annotation: " + error.message);
    });
  });

  overlayCanvas.addEventListener("pointerdown", function (event) {
    if (state.locked || !state.labelMap.length) {
      return;
    }
    const position = getImageCoordinates(event);
    if (state.tool === "fill") {
      floodFill(position.x, position.y);
      return;
    }
    state.isPainting = true;
    paintBrush(position.x, position.y);
  });

  overlayCanvas.addEventListener("pointermove", function (event) {
    if (!state.isPainting || state.locked || state.tool !== "brush") {
      return;
    }
    const position = getImageCoordinates(event);
    paintBrush(position.x, position.y);
  });

  ["pointerup", "pointerleave", "pointercancel"].forEach(function (eventName) {
    overlayCanvas.addEventListener(eventName, function () {
      state.isPainting = false;
    });
  });

  sourceImage.addEventListener("load", function () {
    syncCanvasSize(sourceImage.naturalWidth, sourceImage.naturalHeight);
  });

  if (sourceImage.complete && sourceImage.naturalWidth > 0) {
    syncCanvasSize(sourceImage.naturalWidth, sourceImage.naturalHeight);
  }

  loadAnnotation().catch(function (error) {
    setStatus("Failed to load annotation: " + error.message);
  });
  updateToolbarState();
}());
