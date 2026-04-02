(function () {
  const config = window.APP_CONFIG;
  if (!config) {
    return;
  }

  let apiBase = config.apiBase || "/labeler/api";
  if (Array.isArray(apiBase)) {
    apiBase = apiBase[0] || "/labeler/api";
  }
  apiBase = String(apiBase);

  let imageName = String(config.imageName || "");
  let prevImageUrl = config.prevImageUrl ? String(config.prevImageUrl) : "";
  let nextImageUrl = config.nextImageUrl ? String(config.nextImageUrl) : "";
  const slicDefault = config.slicDefault || {};
  let slicCurrent = config.slicCurrent || {};
  const slicDetailPresets = config.slicDetailPresets || {};
  const simselectDefaults = config.simselectDefaults || {};
  const classes = Array.isArray(config.classes) ? config.classes : [];
  const classNameMap = config.classNameMap || {};
  let currentClassId = Number(config.defaultClassId);
  if (!Number.isInteger(currentClassId) || currentClassId <= 0) {
    if (classes.length > 0 && Number.isInteger(Number(classes[0].id))) {
      currentClassId = Number(classes[0].id);
    } else {
      currentClassId = 1;
    }
  }
  let mode = "add";
  let selectionTool = "single";
  let marqueeRule = "centroid";
  let isDirty = false;
  let isSaving = false;
  let simselectInputMode = "brush";
  let simselectRoiMode = "paint";

  const selectedIds = new Set();

  const MIN_ZOOM = 0.25;
  const MAX_ZOOM = 8;

  const viewport = document.getElementById("canvas-viewport");
  const stage = document.getElementById("canvas-stage");
  const baseImage = document.getElementById("base-image");
  const boundaryImage = document.getElementById("boundary-image");
  const maskImage = document.getElementById("mask-image");
  const selectionImage = document.getElementById("selection-image");
  const simselectRoiCanvas = document.getElementById("simselect-roi-canvas");
  const simselectPreviewImage = document.getElementById("simselect-preview-image");
  const simselectSeedImage = document.getElementById("simselect-seed-image");
  const marqueeRect = document.getElementById("marquee-rect");

  const classSelect = document.getElementById("class-select");
  const slicAlgorithmSelect = document.getElementById("slic-algorithm");
  const slicAlgorithmNote = document.getElementById("slic-algorithm-note");
  const slicPresetModeSelect = document.getElementById("slic-preset-mode");
  const slicDetailWrap = document.getElementById("slic-detail-wrap");
  const slicDetailSelect = document.getElementById("slic-detail");
  const slicColorspaceWrap = document.getElementById("slic-colorspace-wrap");
  const slicColorspaceSelect = document.getElementById("slic-colorspace");
  const slicCustomWrap = document.getElementById("slic-custom-wrap");
  const quickshiftCustomWrap = document.getElementById("quickshift-custom-wrap");
  const felzenszwalbCustomWrap = document.getElementById("felzenszwalb-custom-wrap");
  const slicTextureWrap = document.getElementById("slic-texture-wrap");
  const textureEnabledInput = document.getElementById("texture-enabled");
  const textureLbpEnabledInput = document.getElementById("texture-lbp-enabled");
  const textureLbpPointsInput = document.getElementById("texture-lbp-points");
  const textureLbpRadiiInput = document.getElementById("texture-lbp-radii");
  const textureLbpMethodSelect = document.getElementById("texture-lbp-method");
  const textureLbpNormalizeInput = document.getElementById("texture-lbp-normalize");
  const textureLbpOptions = document.getElementById("texture-lbp-options");
  const textureGaborEnabledInput = document.getElementById("texture-gabor-enabled");
  const textureGaborFrequenciesInput = document.getElementById("texture-gabor-frequencies");
  const textureGaborThetasInput = document.getElementById("texture-gabor-thetas");
  const textureGaborBandwidthInput = document.getElementById("texture-gabor-bandwidth");
  const textureGaborIncludeRealInput = document.getElementById("texture-gabor-include-real");
  const textureGaborIncludeImagInput = document.getElementById("texture-gabor-include-imag");
  const textureGaborIncludeMagnitudeInput = document.getElementById("texture-gabor-include-magnitude");
  const textureGaborNormalizeInput = document.getElementById("texture-gabor-normalize");
  const textureGaborOptions = document.getElementById("texture-gabor-options");
  const textureWeightColorInput = document.getElementById("texture-weight-color");
  const textureWeightLbpInput = document.getElementById("texture-weight-lbp");
  const textureWeightGaborInput = document.getElementById("texture-weight-gabor");
  const textureWeightOptions = document.getElementById("texture-weight-options");
  const slicCompactnessWrap = document.getElementById("slic-compactness-wrap");
  const slicNSegmentsInput = document.getElementById("slic-n-segments");
  const slicCompactnessInput = document.getElementById("slic-compactness");
  const slicSigmaInput = document.getElementById("slic-sigma");
  const slicNSegmentsValue = document.getElementById("slic-n-segments-value");
  const slicCompactnessValue = document.getElementById("slic-compactness-value");
  const slicSigmaValue = document.getElementById("slic-sigma-value");
  const quickshiftRatioInput = document.getElementById("quickshift-ratio");
  const quickshiftKernelSizeInput = document.getElementById("quickshift-kernel-size");
  const quickshiftMaxDistInput = document.getElementById("quickshift-max-dist");
  const quickshiftSigmaInput = document.getElementById("quickshift-sigma");
  const quickshiftRatioValue = document.getElementById("quickshift-ratio-value");
  const quickshiftKernelSizeValue = document.getElementById("quickshift-kernel-size-value");
  const quickshiftMaxDistValue = document.getElementById("quickshift-max-dist-value");
  const quickshiftSigmaValue = document.getElementById("quickshift-sigma-value");
  const felzenszwalbScaleInput = document.getElementById("felzenszwalb-scale");
  const felzenszwalbSigmaInput = document.getElementById("felzenszwalb-sigma");
  const felzenszwalbMinSizeInput = document.getElementById("felzenszwalb-min-size");
  const felzenszwalbScaleValue = document.getElementById("felzenszwalb-scale-value");
  const felzenszwalbSigmaValue = document.getElementById("felzenszwalb-sigma-value");
  const felzenszwalbMinSizeValue = document.getElementById("felzenszwalb-min-size-value");
  const recomputeSlicBtn = document.getElementById("recompute-slic-btn");
  const resetSlicDefaultBtn = document.getElementById("reset-slic-default-btn");
  const applySlicRemainingBtn = document.getElementById("apply-slic-remaining-btn");
  const freezeMasksToggle = document.getElementById("freeze-masks-toggle");

  const addBtn = document.getElementById("mode-add");
  const removeBtn = document.getElementById("mode-remove");
  const selectionToolButtons = Array.from(
    document.querySelectorAll(".labeler-tool-toggle .tool-toggle-btn[data-tool]")
  );
  const marqueeRuleWrap = document.getElementById("marquee-rule-wrap");
  const marqueeRuleSelect = document.getElementById("marquee-rule");
  const selectionCountEl = document.getElementById("selection-count");
  const applySelectionBtn = document.getElementById("apply-selection-btn");
  const clearSelectionBtn = document.getElementById("clear-selection-btn");
  const simselectPanel = document.getElementById("simselect-panel");
  const simselectInputBrushBtn = document.getElementById("simselect-input-brush");
  const simselectInputSeedBtn = document.getElementById("simselect-input-seed");
  const simselectRoiPaintBtn = document.getElementById("simselect-roi-paint");
  const simselectRoiEraseBtn = document.getElementById("simselect-roi-erase");
  const simselectBrushSizeInput = document.getElementById("simselect-brush-size");
  const simselectBrushSizeValue = document.getElementById("simselect-brush-size-value");
  const simselectRoiCountEl = document.getElementById("simselect-roi-count");
  const simselectMatchCountEl = document.getElementById("simselect-match-count");
  const simselectSeedLabelEl = document.getElementById("simselect-seed-label");
  const simselectColorEnabledInput = document.getElementById("simselect-color-enabled");
  const simselectTextureEnabledInput = document.getElementById("simselect-texture-enabled");
  const simselectColorThresholdInput = document.getElementById("simselect-color-threshold");
  const simselectColorThresholdValue = document.getElementById("simselect-color-threshold-value");
  const simselectTextureThresholdInput = document.getElementById("simselect-texture-threshold");
  const simselectTextureThresholdValue = document.getElementById("simselect-texture-threshold-value");
  const simselectLbpPointsInput = document.getElementById("simselect-lbp-points");
  const simselectLbpRadiusInput = document.getElementById("simselect-lbp-radius");
  const simselectLbpMethodSelect = document.getElementById("simselect-lbp-method");
  const simselectPrepareBtn = document.getElementById("simselect-prepare-btn");
  const simselectAddBtn = document.getElementById("simselect-add-btn");
  const simselectSubtractBtn = document.getElementById("simselect-subtract-btn");
  const simselectResetBtn = document.getElementById("simselect-reset-btn");
  const simselectClearRoiBtn = document.getElementById("simselect-clear-roi-btn");
  const boundaryToggle = document.getElementById("toggle-boundary");
  const undoBtn = document.getElementById("undo-btn");
  const redoBtn = document.getElementById("redo-btn");
  const saveBtn = document.getElementById("save-btn");
  const exportBtn = document.getElementById("export-btn");
  const statusEl = document.getElementById("status");
  const dirtyIndicator = document.getElementById("dirty-indicator");
  const zoomOutBtn = document.getElementById("zoom-out-btn");
  const zoomInBtn = document.getElementById("zoom-in-btn");
  const zoomFitBtn = document.getElementById("zoom-fit-btn");
  const zoomLevelEl = document.getElementById("zoom-level");
  const toastStack = document.getElementById("labeler-toast-stack");
  const filmstripEl = document.getElementById("labeler-filmstrip");
  const imageNameEl = document.querySelector(".labeler-image-name code");

  if (!viewport || !stage || !baseImage || !boundaryImage || !maskImage) {
    return;
  }

  let imageWidth = 0;
  let imageHeight = 0;
  let zoom = 1;
  let panX = 0;
  let panY = 0;

  let isPanning = false;
  let panOriginX = 0;
  let panOriginY = 0;
  let suppressNextClick = false;
  let spacePressed = false;

  let isBrushSelecting = false;
  let brushPoints = [];
  let isMarqueeSelecting = false;
  let marqueeStartViewport = null;
  let isImageSwitching = false;
  let isSimselectRoiPainting = false;
  let simselectRoiStrokeMoved = false;
  let simselectLastPoint = null;
  let simselectQueryTimer = null;
  let simselectQueryNonce = 0;
  let simselectPreparePromise = null;
  let simselectRoiMask = null;
  let simselectRoiPixels = 0;
  let simselectMatchedIds = [];
  let simselectSeedId = null;

  let slicAlgorithm = String(slicCurrent.algorithm || "slic");
  let slicPresetMode = String(slicCurrent.preset_mode || "dataset_default");
  let slicDetailLevel = String(slicCurrent.detail_level || "medium");
  const TEXTURE_DEFAULTS = {
    lbpRadii: [1],
    gaborFrequencies: [0.1, 0.2],
    gaborThetas: [0, 45, 90, 135],
  };
  const SIMSELECT_DEFAULTS = {
    colorEnabled: coerceBoolean(simselectDefaults.color_enabled, true),
    textureEnabled: coerceBoolean(simselectDefaults.texture_enabled, true),
    colorThreshold: parseNumber(simselectDefaults.color_threshold, 18.0),
    textureThreshold: parseNumber(simselectDefaults.texture_threshold, 0.35),
    lbpPoints: Math.round(parseNumber(simselectDefaults.lbp_points, 8)),
    lbpRadius: Math.round(parseNumber(simselectDefaults.lbp_radius, 1)),
    lbpMethod: String(simselectDefaults.lbp_method || "uniform"),
  };

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function isFormElement(target) {
    if (!target || !(target instanceof HTMLElement)) {
      return false;
    }
    const tag = target.tagName;
    return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target.isContentEditable;
  }

  function setStatus(message, isError) {
    if (!statusEl) {
      return;
    }
    statusEl.textContent = message;
    statusEl.classList.toggle("error", Boolean(isError));
  }

  function showToast(message, type) {
    if (!toastStack) {
      return;
    }
    const tone = type === "error" ? "error" : "success";
    const item = document.createElement("div");
    item.className = `labeler-toast ${tone}`;
    item.textContent = message;
    toastStack.appendChild(item);
    window.setTimeout(() => {
      item.remove();
    }, 3400);
  }

  function setDirty(nextValue) {
    isDirty = Boolean(nextValue);
    if (!dirtyIndicator) {
      return;
    }
    dirtyIndicator.textContent = isDirty ? "Unsaved changes" : "Saved";
    dirtyIndicator.classList.toggle("dirty", isDirty);
    dirtyIndicator.classList.toggle("clean", !isDirty);
  }

  function setSaving(nextValue) {
    isSaving = Boolean(nextValue);
    if (!saveBtn) {
      return;
    }
    saveBtn.disabled = isSaving;
    saveBtn.textContent = isSaving ? "Saving..." : "Save Masks";
  }

  function setMode(nextMode) {
    mode = nextMode === "remove" ? "remove" : "add";
    if (addBtn) {
      addBtn.classList.toggle("active", mode === "add");
    }
    if (removeBtn) {
      removeBtn.classList.toggle("active", mode === "remove");
    }
    viewport.classList.toggle("mode-remove", mode === "remove");
  }

  function updateSelectionToolUI() {
    viewport.classList.toggle("tool-brush", selectionTool === "brush");
    viewport.classList.toggle("tool-marquee", selectionTool === "marquee");
    viewport.classList.toggle("tool-similarity", selectionTool === "similarity");
    selectionToolButtons.forEach((button) => {
      const tool = String(button.dataset.tool || "");
      const isActive = tool === selectionTool;
      button.classList.toggle("active", isActive);
      button.setAttribute("aria-pressed", isActive ? "true" : "false");
    });
    if (marqueeRuleWrap) {
      marqueeRuleWrap.style.display = selectionTool === "marquee" ? "block" : "none";
    }
    if (simselectPanel) {
      const showSimselect = selectionTool === "similarity";
      simselectPanel.hidden = !showSimselect;
      simselectPanel.style.display = showSimselect ? "grid" : "none";
    }
    if (selectionTool !== "similarity") {
      isSimselectRoiPainting = false;
      simselectLastPoint = null;
      if (simselectQueryTimer) {
        window.clearTimeout(simselectQueryTimer);
        simselectQueryTimer = null;
      }
      simselectQueryNonce += 1;
    }
    setSimselectInputMode(simselectInputMode);
    updateSimselectOverlayVisibility();
    normalizeToolGridLayout();
  }

  function setSelectionTool(nextTool) {
    selectionTool =
      nextTool === "brush" || nextTool === "marquee" || nextTool === "similarity" ? nextTool : "single";
    updateSelectionToolUI();
  }

  function setMaskFromBase64(maskPngBase64) {
    maskImage.src = `data:image/png;base64,${maskPngBase64}`;
  }

  function setSelectionOverlay(maskPngBase64) {
    if (!selectionImage) {
      return;
    }
    if (!maskPngBase64) {
      selectionImage.style.display = "none";
      selectionImage.src = "";
      return;
    }
    selectionImage.src = `data:image/png;base64,${maskPngBase64}`;
    selectionImage.style.display = "block";
  }

  function setOverlayImage(imgEl, maskPngBase64) {
    if (!imgEl) {
      return;
    }
    if (!maskPngBase64) {
      imgEl.style.display = "none";
      imgEl.src = "";
      return;
    }
    imgEl.src = `data:image/png;base64,${maskPngBase64}`;
    imgEl.style.display = "block";
  }

  function setSimselectPreviewOverlay(maskPngBase64) {
    setOverlayImage(simselectPreviewImage, maskPngBase64);
    updateSimselectOverlayVisibility();
  }

  function setSimselectSeedOverlay(maskPngBase64) {
    setOverlayImage(simselectSeedImage, maskPngBase64);
    updateSimselectOverlayVisibility();
  }

  function updateSimselectStats() {
    if (simselectRoiCountEl) {
      simselectRoiCountEl.textContent = `ROI pixels: ${simselectRoiPixels}`;
    }
    if (simselectMatchCountEl) {
      simselectMatchCountEl.textContent = `Matches: ${simselectMatchedIds.length}`;
    }
    if (simselectSeedLabelEl) {
      const seedText =
        simselectSeedId === null || simselectSeedId === undefined ? "none" : String(simselectSeedId);
      simselectSeedLabelEl.textContent = `Seed superpixel: ${seedText}`;
    }
  }

  function clearSimselectPreview(clearSeedOverlay) {
    simselectMatchedIds = [];
    setSimselectPreviewOverlay("");
    if (clearSeedOverlay) {
      setSimselectSeedOverlay("");
    }
    updateSimselectStats();
  }

  function normalizeToolGridLayout() {
    const fullWidthGroups = [
      ".labeler-mode-toggle",
      ".labeler-tool-toggle",
      ".labeler-selection-row",
      ".labeler-action-grid",
      ".simselect-row",
      ".simselect-action-grid",
    ];
    fullWidthGroups.forEach((selector) => {
      const groups = Array.from(document.querySelectorAll(selector));
      groups.forEach((groupEl) => {
        if (!(groupEl instanceof HTMLElement)) {
          return;
        }
        groupEl.style.width = "100%";
        const buttons = Array.from(groupEl.querySelectorAll("button"));
        buttons.forEach((btn) => {
          if (!(btn instanceof HTMLButtonElement)) {
            return;
          }
          btn.style.width = "100%";
          btn.style.minWidth = "0";
          btn.style.maxWidth = "100%";
          btn.style.marginRight = "0";
        });
      });
    });
  }

  function updateSimselectOverlayVisibility() {
    const showSimselect = selectionTool === "similarity";
    if (simselectRoiCanvas) {
      if (!showSimselect) {
        simselectRoiCanvas.style.display = "none";
      } else if (imageWidth > 0 && imageHeight > 0) {
        simselectRoiCanvas.style.display = "block";
      }
    }
    if (simselectPreviewImage) {
      const hasPreview = Boolean(simselectPreviewImage.getAttribute("src"));
      simselectPreviewImage.style.display = showSimselect && hasPreview ? "block" : "none";
    }
    if (simselectSeedImage) {
      const hasSeed = Boolean(simselectSeedImage.getAttribute("src"));
      simselectSeedImage.style.display = showSimselect && hasSeed ? "block" : "none";
    }
  }

  function classNameForId(classId) {
    var key = String(Number(classId));
    if (classNameMap && Object.prototype.hasOwnProperty.call(classNameMap, key)) {
      return String(classNameMap[key] || "");
    }
    var match = classes.find(function (entry) {
      return Number(entry.id) === Number(classId);
    });
    if (match && match.name) {
      return String(match.name);
    }
    return `Class ${Number(classId)}`;
  }

  function updateSelectionCount() {
    if (selectionCountEl) {
      selectionCountEl.textContent = `${selectedIds.size} selected`;
    }
  }

  function parseNumber(value, fallback) {
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) {
      return fallback;
    }
    return parsed;
  }

  function normalizeSlicAlgorithm(value, fallback) {
    var candidate = String(value || "").trim().toLowerCase();
    if (candidate === "slic" || candidate === "slico" || candidate === "quickshift" || candidate === "felzenszwalb") {
      return candidate;
    }
    var fallbackCandidate = String(fallback || "slic").trim().toLowerCase();
    if (
      fallbackCandidate === "slic" ||
      fallbackCandidate === "slico" ||
      fallbackCandidate === "quickshift" ||
      fallbackCandidate === "felzenszwalb"
    ) {
      return fallbackCandidate;
    }
    return "slic";
  }

  function coerceBoolean(value, fallback) {
    if (typeof value === "boolean") {
      return value;
    }
    if (value === null || value === undefined) {
      return Boolean(fallback);
    }
    if (typeof value === "number") {
      return value !== 0;
    }
    var normalized = String(value).trim().toLowerCase();
    if (normalized === "1" || normalized === "true" || normalized === "yes" || normalized === "on") {
      return true;
    }
    if (normalized === "0" || normalized === "false" || normalized === "no" || normalized === "off") {
      return false;
    }
    return Boolean(fallback);
  }

  function normalizeIntegerList(value, fallback, minValue, maxValue) {
    var rawItems = [];
    if (Array.isArray(value)) {
      rawItems = value.slice();
    } else if (typeof value === "string") {
      rawItems = value
        .split(",")
        .map((item) => item.trim())
        .filter((item) => item.length > 0);
    }
    var seen = new Set();
    var normalized = [];
    rawItems.forEach((item) => {
      var parsed = Math.round(parseNumber(item, NaN));
      if (!Number.isFinite(parsed)) {
        return;
      }
      parsed = Math.max(minValue, Math.min(maxValue, parsed));
      if (seen.has(parsed)) {
        return;
      }
      seen.add(parsed);
      normalized.push(parsed);
    });
    if (normalized.length > 0) {
      return normalized;
    }
    return Array.isArray(fallback) ? fallback.slice() : [];
  }

  function normalizeFloatList(value, fallback, minValue, maxValue) {
    var rawItems = [];
    if (Array.isArray(value)) {
      rawItems = value.slice();
    } else if (typeof value === "string") {
      rawItems = value
        .split(",")
        .map((item) => item.trim())
        .filter((item) => item.length > 0);
    }
    var normalized = [];
    rawItems.forEach((item) => {
      var parsed = parseNumber(item, NaN);
      if (!Number.isFinite(parsed)) {
        return;
      }
      parsed = Math.max(minValue, Math.min(maxValue, parsed));
      normalized.push(parsed);
    });
    if (normalized.length > 0) {
      return normalized;
    }
    return Array.isArray(fallback) ? fallback.slice() : [];
  }

  function formatNumberList(values, fallback, fractionDigits) {
    var source = Array.isArray(values) && values.length > 0 ? values : Array.isArray(fallback) ? fallback : [];
    return source
      .map((value) => {
        if (fractionDigits === null || fractionDigits === undefined) {
          return String(value);
        }
        var numeric = parseNumber(value, NaN);
        if (!Number.isFinite(numeric)) {
          return String(value);
        }
        return numeric.toFixed(fractionDigits);
      })
      .join(",");
  }

  function updateSlicSliderLabels() {
    if (slicNSegmentsInput && slicNSegmentsValue) {
      slicNSegmentsValue.textContent = String(Math.round(parseNumber(slicNSegmentsInput.value, 1200)));
    }
    if (slicCompactnessInput && slicCompactnessValue) {
      slicCompactnessValue.textContent = parseNumber(slicCompactnessInput.value, 10).toFixed(2);
    }
    if (slicSigmaInput && slicSigmaValue) {
      slicSigmaValue.textContent = parseNumber(slicSigmaInput.value, 1).toFixed(2);
    }
    if (quickshiftRatioInput && quickshiftRatioValue) {
      quickshiftRatioValue.textContent = parseNumber(quickshiftRatioInput.value, 1).toFixed(2);
    }
    if (quickshiftKernelSizeInput && quickshiftKernelSizeValue) {
      quickshiftKernelSizeValue.textContent = String(Math.round(parseNumber(quickshiftKernelSizeInput.value, 5)));
    }
    if (quickshiftMaxDistInput && quickshiftMaxDistValue) {
      quickshiftMaxDistValue.textContent = parseNumber(quickshiftMaxDistInput.value, 10).toFixed(1);
    }
    if (quickshiftSigmaInput && quickshiftSigmaValue) {
      quickshiftSigmaValue.textContent = parseNumber(quickshiftSigmaInput.value, 0).toFixed(2);
    }
    if (felzenszwalbScaleInput && felzenszwalbScaleValue) {
      felzenszwalbScaleValue.textContent = parseNumber(felzenszwalbScaleInput.value, 100).toFixed(1);
    }
    if (felzenszwalbSigmaInput && felzenszwalbSigmaValue) {
      felzenszwalbSigmaValue.textContent = parseNumber(felzenszwalbSigmaInput.value, 0.8).toFixed(2);
    }
    if (felzenszwalbMinSizeInput && felzenszwalbMinSizeValue) {
      felzenszwalbMinSizeValue.textContent = String(Math.round(parseNumber(felzenszwalbMinSizeInput.value, 50)));
    }
  }

  function setSlicInputsFromValues(values) {
    if (slicNSegmentsInput && values.n_segments !== undefined) {
      slicNSegmentsInput.value = String(values.n_segments);
    }
    if (slicCompactnessInput && values.compactness !== undefined) {
      slicCompactnessInput.value = String(values.compactness);
    }
    if (slicSigmaInput && values.sigma !== undefined) {
      slicSigmaInput.value = String(values.sigma);
    }
    if (slicColorspaceSelect && values.colorspace) {
      slicColorspaceSelect.value = String(values.colorspace);
    }
    if (quickshiftRatioInput && values.quickshift_ratio !== undefined) {
      quickshiftRatioInput.value = String(values.quickshift_ratio);
    }
    if (quickshiftKernelSizeInput && values.quickshift_kernel_size !== undefined) {
      quickshiftKernelSizeInput.value = String(values.quickshift_kernel_size);
    }
    if (quickshiftMaxDistInput && values.quickshift_max_dist !== undefined) {
      quickshiftMaxDistInput.value = String(values.quickshift_max_dist);
    }
    if (quickshiftSigmaInput && values.quickshift_sigma !== undefined) {
      quickshiftSigmaInput.value = String(values.quickshift_sigma);
    }
    if (felzenszwalbScaleInput && values.felzenszwalb_scale !== undefined) {
      felzenszwalbScaleInput.value = String(values.felzenszwalb_scale);
    }
    if (felzenszwalbSigmaInput && values.felzenszwalb_sigma !== undefined) {
      felzenszwalbSigmaInput.value = String(values.felzenszwalb_sigma);
    }
    if (felzenszwalbMinSizeInput && values.felzenszwalb_min_size !== undefined) {
      felzenszwalbMinSizeInput.value = String(values.felzenszwalb_min_size);
    }
    if (textureEnabledInput && values.texture_enabled !== undefined) {
      textureEnabledInput.checked = coerceBoolean(values.texture_enabled, false);
    }
    if (textureLbpEnabledInput && values.texture_lbp_enabled !== undefined) {
      textureLbpEnabledInput.checked = coerceBoolean(values.texture_lbp_enabled, false);
    }
    if (textureLbpPointsInput && values.texture_lbp_points !== undefined) {
      textureLbpPointsInput.value = String(Math.round(parseNumber(values.texture_lbp_points, 8)));
    }
    if (textureLbpRadiiInput) {
      var lbpRadiiValue = values.texture_lbp_radii !== undefined ? values.texture_lbp_radii : values.texture_lbp_radii_json;
      var lbpRadii = normalizeIntegerList(lbpRadiiValue, TEXTURE_DEFAULTS.lbpRadii, 1, 64);
      textureLbpRadiiInput.value = formatNumberList(lbpRadii, TEXTURE_DEFAULTS.lbpRadii, null);
    }
    if (textureLbpMethodSelect && values.texture_lbp_method) {
      textureLbpMethodSelect.value = String(values.texture_lbp_method);
    }
    if (textureLbpNormalizeInput && values.texture_lbp_normalize !== undefined) {
      textureLbpNormalizeInput.checked = coerceBoolean(values.texture_lbp_normalize, true);
    }
    if (textureGaborEnabledInput && values.texture_gabor_enabled !== undefined) {
      textureGaborEnabledInput.checked = coerceBoolean(values.texture_gabor_enabled, false);
    }
    if (textureGaborFrequenciesInput) {
      var gaborFrequenciesValue =
        values.texture_gabor_frequencies !== undefined
          ? values.texture_gabor_frequencies
          : values.texture_gabor_frequencies_json;
      var gaborFrequencies = normalizeFloatList(gaborFrequenciesValue, TEXTURE_DEFAULTS.gaborFrequencies, 0.001, 1.0);
      textureGaborFrequenciesInput.value = formatNumberList(gaborFrequencies, TEXTURE_DEFAULTS.gaborFrequencies, 3);
    }
    if (textureGaborThetasInput) {
      var gaborThetasValue =
        values.texture_gabor_thetas !== undefined ? values.texture_gabor_thetas : values.texture_gabor_thetas_json;
      var gaborThetas = normalizeFloatList(gaborThetasValue, TEXTURE_DEFAULTS.gaborThetas, 0.0, 179.999);
      textureGaborThetasInput.value = formatNumberList(gaborThetas, TEXTURE_DEFAULTS.gaborThetas, 3);
    }
    if (textureGaborBandwidthInput && values.texture_gabor_bandwidth !== undefined) {
      textureGaborBandwidthInput.value = parseNumber(values.texture_gabor_bandwidth, 1.0).toFixed(2);
    }
    if (textureGaborIncludeRealInput && values.texture_gabor_include_real !== undefined) {
      textureGaborIncludeRealInput.checked = coerceBoolean(values.texture_gabor_include_real, false);
    }
    if (textureGaborIncludeImagInput && values.texture_gabor_include_imag !== undefined) {
      textureGaborIncludeImagInput.checked = coerceBoolean(values.texture_gabor_include_imag, false);
    }
    if (textureGaborIncludeMagnitudeInput && values.texture_gabor_include_magnitude !== undefined) {
      textureGaborIncludeMagnitudeInput.checked = coerceBoolean(values.texture_gabor_include_magnitude, true);
    }
    if (textureGaborNormalizeInput && values.texture_gabor_normalize !== undefined) {
      textureGaborNormalizeInput.checked = coerceBoolean(values.texture_gabor_normalize, true);
    }
    if (textureWeightColorInput && values.texture_weight_color !== undefined) {
      textureWeightColorInput.value = parseNumber(values.texture_weight_color, 1.0).toFixed(2);
    }
    if (textureWeightLbpInput && values.texture_weight_lbp !== undefined) {
      textureWeightLbpInput.value = parseNumber(values.texture_weight_lbp, 0.25).toFixed(2);
    }
    if (textureWeightGaborInput && values.texture_weight_gabor !== undefined) {
      textureWeightGaborInput.value = parseNumber(values.texture_weight_gabor, 0.25).toFixed(2);
    }
    updateSlicSliderLabels();
  }

  function applyDetailPresetInputs(detailLevel) {
    const preset = slicDetailPresets[detailLevel];
    if (!preset) {
      return;
    }
    setSlicInputsFromValues({
      n_segments: preset.n_segments,
      compactness: preset.compactness,
      sigma: preset.sigma,
    });
  }

  function updateAlgorithmCustomWraps(isCustomMode) {
    var showSlic = isCustomMode && (slicAlgorithm === "slic" || slicAlgorithm === "slico");
    var showQuickshift = isCustomMode && slicAlgorithm === "quickshift";
    var showFelzenszwalb = isCustomMode && slicAlgorithm === "felzenszwalb";
    if (slicCustomWrap) {
      slicCustomWrap.style.display = showSlic ? "grid" : "none";
    }
    if (quickshiftCustomWrap) {
      quickshiftCustomWrap.style.display = showQuickshift ? "grid" : "none";
    }
    if (felzenszwalbCustomWrap) {
      felzenszwalbCustomWrap.style.display = showFelzenszwalb ? "grid" : "none";
    }
  }

  function updateTextureControlsUI() {
    var isSlicFamily = slicAlgorithm === "slic" || slicAlgorithm === "slico";
    var allowSlicOverrides = slicPresetMode === "custom" || slicPresetMode === "detail";
    var showSlicFamilyControls = allowSlicOverrides && isSlicFamily;
    if (slicColorspaceWrap) {
      slicColorspaceWrap.style.display = showSlicFamilyControls ? "" : "none";
    }
    if (slicTextureWrap) {
      slicTextureWrap.style.display = showSlicFamilyControls ? "grid" : "none";
    }
    var textureEnabled = showSlicFamilyControls && textureEnabledInput && textureEnabledInput.checked;
    var textureLbpEnabled = textureEnabled && textureLbpEnabledInput && textureLbpEnabledInput.checked;
    var textureGaborEnabled = textureEnabled && textureGaborEnabledInput && textureGaborEnabledInput.checked;
    if (textureLbpOptions) {
      textureLbpOptions.style.display = textureLbpEnabled ? "grid" : "none";
    }
    if (textureGaborOptions) {
      textureGaborOptions.style.display = textureGaborEnabled ? "grid" : "none";
    }
    if (textureWeightOptions) {
      textureWeightOptions.style.display = textureEnabled ? "grid" : "none";
    }
  }

  function updateSlicAlgorithmUI() {
    if (slicAlgorithmSelect) {
      slicAlgorithm = normalizeSlicAlgorithm(slicAlgorithmSelect.value, "slic");
      slicAlgorithmSelect.value = slicAlgorithm;
    } else {
      slicAlgorithm = normalizeSlicAlgorithm(slicAlgorithm, "slic");
    }
    var usesSlico = slicAlgorithm === "slico";
    var isSlic = slicAlgorithm === "slic";
    var isQuickshift = slicAlgorithm === "quickshift";
    var isFelzenszwalb = slicAlgorithm === "felzenszwalb";
    if (slicCompactnessWrap) {
      slicCompactnessWrap.style.display = usesSlico ? "none" : "";
    }
    if (slicAlgorithmNote) {
      if (usesSlico) {
        slicAlgorithmNote.textContent =
          "SLICO adapts compactness automatically and ignores manual compactness tuning.";
      } else if (isSlic) {
        slicAlgorithmNote.textContent = "SLIC uses the configured compactness value.";
      } else if (isQuickshift) {
        slicAlgorithmNote.textContent =
          "Quickshift clusters pixels by local density using ratio, kernel size, and max distance.";
      } else if (isFelzenszwalb) {
        slicAlgorithmNote.textContent =
          "Felzenszwalb graph segmentation uses scale, sigma, and minimum segment size.";
      } else {
        slicAlgorithmNote.textContent = "";
      }
    }
    updateAlgorithmCustomWraps(slicPresetMode === "custom");
  }

  function updateSlicPresetUI() {
    if (slicPresetModeSelect) {
      slicPresetMode = String(slicPresetModeSelect.value || "dataset_default");
    }
    if (slicDetailSelect) {
      slicDetailLevel = String(slicDetailSelect.value || "medium");
    }
    if (slicAlgorithmSelect) {
      slicAlgorithm = normalizeSlicAlgorithm(slicAlgorithmSelect.value, slicAlgorithm);
    }
    var isSlicFamily = slicAlgorithm === "slic" || slicAlgorithm === "slico";
    if (!isSlicFamily && slicPresetMode === "detail") {
      slicPresetMode = "custom";
      if (slicPresetModeSelect) {
        slicPresetModeSelect.value = "custom";
      }
    }

    if (slicPresetMode === "dataset_default") {
      if (slicDetailWrap) {
        slicDetailWrap.style.display = "none";
      }
      updateAlgorithmCustomWraps(false);
      setSlicInputsFromValues(slicDefault);
      updateSlicAlgorithmUI();
      updateTextureControlsUI();
      return;
    }

    if (slicPresetMode === "detail") {
      if (slicDetailWrap) {
        slicDetailWrap.style.display = "block";
      }
      updateAlgorithmCustomWraps(false);
      applyDetailPresetInputs(slicDetailLevel);
      updateSlicAlgorithmUI();
      updateTextureControlsUI();
      return;
    }

    if (slicDetailWrap) {
      slicDetailWrap.style.display = "none";
    }
    updateAlgorithmCustomWraps(true);
    updateSlicAlgorithmUI();
    updateTextureControlsUI();
    updateSlicSliderLabels();
  }

  function currentSlicPayload(applyRemaining, forceOverwrite) {
    var textureLbpRadii = normalizeIntegerList(
      textureLbpRadiiInput ? textureLbpRadiiInput.value : "",
      TEXTURE_DEFAULTS.lbpRadii,
      1,
      64
    );
    var textureGaborFrequencies = normalizeFloatList(
      textureGaborFrequenciesInput ? textureGaborFrequenciesInput.value : "",
      TEXTURE_DEFAULTS.gaborFrequencies,
      0.001,
      1.0
    );
    var textureGaborThetas = normalizeFloatList(
      textureGaborThetasInput ? textureGaborThetasInput.value : "",
      TEXTURE_DEFAULTS.gaborThetas,
      0.0,
      179.999
    );
    const payload = {
      image_name: imageName,
      algorithm: normalizeSlicAlgorithm(slicAlgorithm, "slic"),
      preset_mode: slicPresetMode,
      detail_level: slicDetailLevel,
      apply_remaining: Boolean(applyRemaining),
      force_overwrite: Boolean(forceOverwrite),
      freeze_masks: freezeMasksToggle ? Boolean(freezeMasksToggle.checked) : false,
      colorspace: slicColorspaceSelect ? String(slicColorspaceSelect.value || "lab") : "lab",
      texture_enabled: textureEnabledInput ? Boolean(textureEnabledInput.checked) : false,
      texture_mode: "append_to_color",
      texture_lbp_enabled: textureLbpEnabledInput ? Boolean(textureLbpEnabledInput.checked) : false,
      texture_lbp_points: Math.round(parseNumber(textureLbpPointsInput && textureLbpPointsInput.value, 8)),
      texture_lbp_radii: textureLbpRadii,
      texture_lbp_method: textureLbpMethodSelect ? String(textureLbpMethodSelect.value || "uniform") : "uniform",
      texture_lbp_normalize: textureLbpNormalizeInput ? Boolean(textureLbpNormalizeInput.checked) : true,
      texture_gabor_enabled: textureGaborEnabledInput ? Boolean(textureGaborEnabledInput.checked) : false,
      texture_gabor_frequencies: textureGaborFrequencies,
      texture_gabor_thetas: textureGaborThetas,
      texture_gabor_bandwidth: parseNumber(textureGaborBandwidthInput && textureGaborBandwidthInput.value, 1.0),
      texture_gabor_include_real: textureGaborIncludeRealInput ? Boolean(textureGaborIncludeRealInput.checked) : false,
      texture_gabor_include_imag: textureGaborIncludeImagInput ? Boolean(textureGaborIncludeImagInput.checked) : false,
      texture_gabor_include_magnitude: textureGaborIncludeMagnitudeInput
        ? Boolean(textureGaborIncludeMagnitudeInput.checked)
        : true,
      texture_gabor_normalize: textureGaborNormalizeInput ? Boolean(textureGaborNormalizeInput.checked) : true,
      texture_weight_color: parseNumber(textureWeightColorInput && textureWeightColorInput.value, 1.0),
      texture_weight_lbp: parseNumber(textureWeightLbpInput && textureWeightLbpInput.value, 0.25),
      texture_weight_gabor: parseNumber(textureWeightGaborInput && textureWeightGaborInput.value, 0.25),
    };
    if (textureLbpRadiiInput) {
      textureLbpRadiiInput.value = formatNumberList(textureLbpRadii, TEXTURE_DEFAULTS.lbpRadii, null);
    }
    if (textureGaborFrequenciesInput) {
      textureGaborFrequenciesInput.value = formatNumberList(
        textureGaborFrequencies,
        TEXTURE_DEFAULTS.gaborFrequencies,
        3
      );
    }
    if (textureGaborThetasInput) {
      textureGaborThetasInput.value = formatNumberList(textureGaborThetas, TEXTURE_DEFAULTS.gaborThetas, 3);
    }

    if (
      payload.texture_gabor_enabled &&
      !payload.texture_gabor_include_real &&
      !payload.texture_gabor_include_imag &&
      !payload.texture_gabor_include_magnitude
    ) {
      payload.texture_gabor_include_magnitude = true;
      if (textureGaborIncludeMagnitudeInput) {
        textureGaborIncludeMagnitudeInput.checked = true;
      }
    }
    if (payload.texture_enabled && !payload.texture_lbp_enabled && !payload.texture_gabor_enabled) {
      payload.texture_enabled = false;
    }
    if (payload.algorithm !== "slic" && payload.algorithm !== "slico") {
      payload.texture_enabled = false;
    }

    if (slicPresetMode === "custom") {
      payload.n_segments = Math.round(parseNumber(slicNSegmentsInput && slicNSegmentsInput.value, 1200));
      payload.compactness = parseNumber(slicCompactnessInput && slicCompactnessInput.value, 10);
      payload.sigma = parseNumber(slicSigmaInput && slicSigmaInput.value, 1);
      payload.quickshift_ratio = parseNumber(quickshiftRatioInput && quickshiftRatioInput.value, 1.0);
      payload.quickshift_kernel_size = Math.round(
        parseNumber(quickshiftKernelSizeInput && quickshiftKernelSizeInput.value, 5)
      );
      payload.quickshift_max_dist = parseNumber(quickshiftMaxDistInput && quickshiftMaxDistInput.value, 10.0);
      payload.quickshift_sigma = parseNumber(quickshiftSigmaInput && quickshiftSigmaInput.value, 0.0);
      payload.felzenszwalb_scale = parseNumber(felzenszwalbScaleInput && felzenszwalbScaleInput.value, 100.0);
      payload.felzenszwalb_sigma = parseNumber(felzenszwalbSigmaInput && felzenszwalbSigmaInput.value, 0.8);
      payload.felzenszwalb_min_size = Math.round(
        parseNumber(felzenszwalbMinSizeInput && felzenszwalbMinSizeInput.value, 50)
      );
    }
    return payload;
  }

  async function parseJsonResponse(response) {
    const rawText = await response.text();
    let payload = {};
    if (rawText) {
      try {
        payload = JSON.parse(rawText);
      } catch (err) {
        if (!response.ok) {
          throw new Error(`Server returned ${response.status} ${response.statusText}`);
        }
        throw new Error("Invalid JSON response from server.");
      }
    }
    if (!response.ok) {
      throw new Error(payload.error || payload.message || `Request failed (${response.status}).`);
    }
    return payload;
  }

  async function postJSON(url, body) {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return parseJsonResponse(response);
  }

  async function getJSON(url) {
    const response = await fetch(url);
    return parseJsonResponse(response);
  }

  function sortedSelectedIds() {
    return Array.from(selectedIds).sort((a, b) => a - b);
  }

  function addSelectedIds(ids) {
    let added = 0;
    (ids || []).forEach((value) => {
      const spId = Number(value);
      if (!Number.isInteger(spId) || spId < 0) {
        return;
      }
      if (!selectedIds.has(spId)) {
        selectedIds.add(spId);
        added += 1;
      }
    });
    updateSelectionCount();
    return added;
  }

  function removeSelectedIds(ids) {
    let removed = 0;
    (ids || []).forEach((value) => {
      const spId = Number(value);
      if (!Number.isInteger(spId) || spId < 0) {
        return;
      }
      if (selectedIds.has(spId)) {
        selectedIds.delete(spId);
        removed += 1;
      }
    });
    updateSelectionCount();
    return removed;
  }

  async function refreshSelectionOverlay() {
    if (!selectionImage) {
      return;
    }
    if (selectedIds.size === 0) {
      setSelectionOverlay("");
      return;
    }
    try {
      const payload = await postJSON(`${apiBase}/selection_preview`, {
        image_name: imageName,
        selected_ids: sortedSelectedIds(),
      });
      setSelectionOverlay(payload.mask_png_base64);
    } catch (err) {
      setStatus(err.message, true);
      showToast(err.message, "error");
    }
  }

  async function clearSelection() {
    selectedIds.clear();
    updateSelectionCount();
    setSelectionOverlay("");
  }

  function normalizeSimselectLbpMethod(value) {
    const candidate = String(value || "").trim().toLowerCase();
    if (candidate === "uniform" || candidate === "ror" || candidate === "default") {
      return candidate;
    }
    return "uniform";
  }

  function simselectFeatureConfigPayload() {
    const points = Math.round(parseNumber(simselectLbpPointsInput && simselectLbpPointsInput.value, SIMSELECT_DEFAULTS.lbpPoints));
    const radius = Math.round(parseNumber(simselectLbpRadiusInput && simselectLbpRadiusInput.value, SIMSELECT_DEFAULTS.lbpRadius));
    const method = normalizeSimselectLbpMethod(simselectLbpMethodSelect ? simselectLbpMethodSelect.value : SIMSELECT_DEFAULTS.lbpMethod);
    if (simselectLbpPointsInput) {
      simselectLbpPointsInput.value = String(clamp(points, 4, 64));
    }
    if (simselectLbpRadiusInput) {
      simselectLbpRadiusInput.value = String(clamp(radius, 1, 16));
    }
    if (simselectLbpMethodSelect) {
      simselectLbpMethodSelect.value = method;
    }
    return {
      lbp_points: clamp(points, 4, 64),
      lbp_radius: clamp(radius, 1, 16),
      lbp_method: method,
    };
  }

  function simselectQueryConfigPayload() {
    const colorEnabled = simselectColorEnabledInput ? Boolean(simselectColorEnabledInput.checked) : true;
    const textureEnabled = simselectTextureEnabledInput ? Boolean(simselectTextureEnabledInput.checked) : true;
    const colorThreshold = parseNumber(
      simselectColorThresholdInput && simselectColorThresholdInput.value,
      SIMSELECT_DEFAULTS.colorThreshold
    );
    const textureThreshold = parseNumber(
      simselectTextureThresholdInput && simselectTextureThresholdInput.value,
      SIMSELECT_DEFAULTS.textureThreshold
    );
    return {
      color_enabled: colorEnabled,
      texture_enabled: textureEnabled,
      color_threshold: clamp(colorThreshold, 0, 200),
      texture_threshold: clamp(textureThreshold, 0, 10),
    };
  }

  function setSimselectInputMode(nextMode) {
    simselectInputMode = nextMode === "seed" ? "seed" : "brush";
    if (simselectInputBrushBtn) {
      simselectInputBrushBtn.classList.toggle("active", simselectInputMode === "brush");
    }
    if (simselectInputSeedBtn) {
      simselectInputSeedBtn.classList.toggle("active", simselectInputMode === "seed");
    }
    if (viewport) {
      viewport.classList.toggle("simselect-seed", selectionTool === "similarity" && simselectInputMode === "seed");
    }
  }

  function setSimselectRoiMode(nextMode) {
    simselectRoiMode = nextMode === "erase" ? "erase" : "paint";
    if (simselectRoiPaintBtn) {
      simselectRoiPaintBtn.classList.toggle("active", simselectRoiMode === "paint");
    }
    if (simselectRoiEraseBtn) {
      simselectRoiEraseBtn.classList.toggle("active", simselectRoiMode === "erase");
    }
  }

  function updateSimselectThresholdLabels() {
    if (simselectBrushSizeValue) {
      const brushSize = Math.round(parseNumber(simselectBrushSizeInput && simselectBrushSizeInput.value, 24));
      simselectBrushSizeValue.textContent = `${brushSize} px`;
    }
    if (simselectColorThresholdValue) {
      const colorThreshold = parseNumber(
        simselectColorThresholdInput && simselectColorThresholdInput.value,
        SIMSELECT_DEFAULTS.colorThreshold
      );
      simselectColorThresholdValue.textContent = colorThreshold.toFixed(1);
    }
    if (simselectTextureThresholdValue) {
      const textureThreshold = parseNumber(
        simselectTextureThresholdInput && simselectTextureThresholdInput.value,
        SIMSELECT_DEFAULTS.textureThreshold
      );
      simselectTextureThresholdValue.textContent = textureThreshold.toFixed(2);
    }
  }

  function resetSimselectRoiCanvas() {
    if (!simselectRoiCanvas) {
      return;
    }
    if (imageWidth > 0 && imageHeight > 0) {
      simselectRoiCanvas.width = imageWidth;
      simselectRoiCanvas.height = imageHeight;
    }
    const ctx = simselectRoiCanvas.getContext("2d");
    if (!ctx) {
      return;
    }
    ctx.clearRect(0, 0, simselectRoiCanvas.width, simselectRoiCanvas.height);
    ctx.globalCompositeOperation = "source-over";
    updateSimselectOverlayVisibility();
  }

  function ensureSimselectRoiMask() {
    if (!imageWidth || !imageHeight) {
      return false;
    }
    const requiredLength = imageWidth * imageHeight;
    if (!(simselectRoiMask instanceof Uint8Array) || simselectRoiMask.length !== requiredLength) {
      simselectRoiMask = new Uint8Array(requiredLength);
      simselectRoiPixels = 0;
      resetSimselectRoiCanvas();
      updateSimselectStats();
    }
    return true;
  }

  function clearSimselectRoiMask(options) {
    const opts = options && typeof options === "object" ? options : {};
    if (!ensureSimselectRoiMask()) {
      return;
    }
    simselectRoiMask.fill(0);
    simselectRoiPixels = 0;
    resetSimselectRoiCanvas();
    clearSimselectPreview(true);
    updateSimselectStats();
    if (!opts.silent && selectionTool === "similarity") {
      setStatus("ROI cleared.", false);
    }
  }

  function clearSimselectStateForImage() {
    if (simselectQueryTimer) {
      window.clearTimeout(simselectQueryTimer);
      simselectQueryTimer = null;
    }
    simselectQueryNonce += 1;
    simselectPreparePromise = null;
    simselectSeedId = null;
    simselectMatchedIds = [];
    simselectRoiMask = null;
    simselectRoiPixels = 0;
    isSimselectRoiPainting = false;
    simselectRoiStrokeMoved = false;
    simselectLastPoint = null;
    setSimselectPreviewOverlay("");
    setSimselectSeedOverlay("");
    resetSimselectRoiCanvas();
    updateSimselectStats();
  }

  function encodeRoiMaskRle(maskArray) {
    if (!(maskArray instanceof Uint8Array) || maskArray.length <= 0) {
      return [];
    }
    const runs = [];
    let runStart = -1;
    for (let i = 0; i < maskArray.length; i += 1) {
      const value = maskArray[i] === 1;
      if (value) {
        if (runStart < 0) {
          runStart = i;
        }
        continue;
      }
      if (runStart >= 0) {
        runs.push([runStart, i - runStart]);
        runStart = -1;
      }
    }
    if (runStart >= 0) {
      runs.push([runStart, maskArray.length - runStart]);
    }
    return runs;
  }

  function drawRoiBrushPoint(x, y, radius, mode) {
    if (!ensureSimselectRoiMask()) {
      return;
    }
    const cx = clamp(Math.round(x), 0, imageWidth - 1);
    const cy = clamp(Math.round(y), 0, imageHeight - 1);
    const r = Math.max(1, Math.round(radius));
    const minX = Math.max(0, cx - r);
    const maxX = Math.min(imageWidth - 1, cx + r);
    const minY = Math.max(0, cy - r);
    const maxY = Math.min(imageHeight - 1, cy + r);
    const paint = mode !== "erase";
    const rSquared = r * r;

    for (let yy = minY; yy <= maxY; yy += 1) {
      for (let xx = minX; xx <= maxX; xx += 1) {
        const dx = xx - cx;
        const dy = yy - cy;
        if ((dx * dx) + (dy * dy) > rSquared) {
          continue;
        }
        const idx = (yy * imageWidth) + xx;
        const prev = simselectRoiMask[idx];
        if (paint) {
          if (prev === 0) {
            simselectRoiMask[idx] = 1;
            simselectRoiPixels += 1;
          }
        } else if (prev === 1) {
          simselectRoiMask[idx] = 0;
          simselectRoiPixels = Math.max(0, simselectRoiPixels - 1);
        }
      }
    }

    if (!simselectRoiCanvas) {
      return;
    }
    const ctx = simselectRoiCanvas.getContext("2d");
    if (!ctx) {
      return;
    }
    if (paint) {
      ctx.globalCompositeOperation = "source-over";
      ctx.fillStyle = "rgba(0, 178, 255, 0.45)";
    } else {
      ctx.globalCompositeOperation = "destination-out";
      ctx.fillStyle = "rgba(0, 0, 0, 1)";
    }
    ctx.beginPath();
    ctx.arc(cx + 0.5, cy + 0.5, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalCompositeOperation = "source-over";
  }

  function drawRoiBrushStroke(fromPoint, toPoint) {
    if (!fromPoint || !toPoint) {
      return;
    }
    const radius = Math.round(parseNumber(simselectBrushSizeInput && simselectBrushSizeInput.value, 24));
    const modeName = simselectRoiMode === "erase" ? "erase" : "paint";
    const dx = toPoint.x - fromPoint.x;
    const dy = toPoint.y - fromPoint.y;
    const steps = Math.max(Math.abs(dx), Math.abs(dy), 1);
    for (let step = 0; step <= steps; step += 1) {
      const t = step / steps;
      const x = fromPoint.x + (dx * t);
      const y = fromPoint.y + (dy * t);
      drawRoiBrushPoint(x, y, radius, modeName);
    }
    updateSimselectStats();
  }

  function scheduleSimselectQuery(delayMs) {
    if (simselectQueryTimer) {
      window.clearTimeout(simselectQueryTimer);
      simselectQueryTimer = null;
    }
    if (selectionTool !== "similarity") {
      return;
    }
    const delay = Number.isFinite(delayMs) ? Math.max(0, Number(delayMs)) : 130;
    simselectQueryTimer = window.setTimeout(() => {
      simselectQueryTimer = null;
      runSimselectQuery();
    }, delay);
  }

  async function prepareSimselectFeatures() {
    if (simselectPreparePromise) {
      return simselectPreparePromise;
    }
    const body = {
      image_name: imageName,
      feature_config: simselectFeatureConfigPayload(),
    };
    simselectPreparePromise = postJSON(`${apiBase}/simselect/prepare`, body)
      .then((payload) => {
        if (payload && payload.ok) {
          setStatus(
            `Similarity features ready for ${payload.num_segments || 0} segments (${payload.cache_hit ? "cache hit" : "computed"}).`,
            false
          );
        }
        return payload;
      })
      .finally(() => {
        simselectPreparePromise = null;
      });
    return simselectPreparePromise;
  }

  async function runSimselectQuery() {
    if (selectionTool !== "similarity") {
      return;
    }
    if (simselectSeedId === null || simselectSeedId === undefined) {
      clearSimselectPreview(true);
      return;
    }
    if (!ensureSimselectRoiMask() || simselectRoiPixels <= 0) {
      clearSimselectPreview(true);
      return;
    }
    const queryConfig = simselectQueryConfigPayload();
    if (!queryConfig.color_enabled && !queryConfig.texture_enabled) {
      clearSimselectPreview(true);
      setStatus("Enable color and/or texture similarity before querying.", true);
      return;
    }

    const nonce = simselectQueryNonce + 1;
    simselectQueryNonce = nonce;
    const payloadBody = {
      image_name: imageName,
      seed_superpixel_id: simselectSeedId,
      roi_mask: {
        shape: [imageHeight, imageWidth],
        runs: encodeRoiMaskRle(simselectRoiMask),
      },
      feature_config: simselectFeatureConfigPayload(),
      ...queryConfig,
    };
    try {
      const payload = await postJSON(`${apiBase}/simselect/query`, payloadBody);
      if (nonce !== simselectQueryNonce) {
        return;
      }
      simselectMatchedIds = Array.isArray(payload.matched_superpixel_ids)
        ? payload.matched_superpixel_ids
            .map((value) => Number(value))
            .filter((value) => Number.isInteger(value) && value >= 0)
        : [];
      setSimselectPreviewOverlay(payload.matched_mask_png_base64 || "");
      setSimselectSeedOverlay(payload.seed_mask_png_base64 || "");
      updateSimselectStats();
      setStatus(
        `Similarity matched ${simselectMatchedIds.length} / ${Number(payload.candidate_count || 0)} candidate superpixels.`,
        false
      );
    } catch (err) {
      if (nonce !== simselectQueryNonce) {
        return;
      }
      clearSimselectPreview(false);
      setStatus(err.message, true);
      showToast(err.message, "error");
    }
  }

  async function setSimselectSeedAtEvent(event) {
    const coords = imageCoordinatesFromEvent(event, false);
    if (!coords) {
      return;
    }
    try {
      const payload = await postJSON(`${apiBase}/select_ids`, {
        image_name: imageName,
        tool: "brush",
        points: [[coords.x, coords.y]],
      });
      const ids = Array.isArray(payload.selected_ids) ? payload.selected_ids : [];
      if (!ids.length) {
        setStatus("No superpixel available at that location.", true);
        return;
      }
      const seed = Number(ids[0]);
      if (!Number.isInteger(seed) || seed < 0) {
        setStatus("Failed to set similarity seed.", true);
        return;
      }
      simselectSeedId = seed;
      updateSimselectStats();
      setStatus(`Seed superpixel set to ${seed}.`, false);
      scheduleSimselectQuery(0);
    } catch (err) {
      setStatus(err.message, true);
      showToast(err.message, "error");
    }
  }

  async function commitSimselectMatches(modeName) {
    const action = modeName === "subtract" ? "subtract" : "add";
    if (!Array.isArray(simselectMatchedIds) || simselectMatchedIds.length <= 0) {
      setStatus("No similarity matches to commit.", true);
      return;
    }
    let changed = 0;
    if (action === "add") {
      changed = addSelectedIds(simselectMatchedIds);
    } else {
      changed = removeSelectedIds(simselectMatchedIds);
    }
    await refreshSelectionOverlay();
    const message =
      action === "add"
        ? `Added ${changed} matched superpixels to selection.`
        : `Removed ${changed} matched superpixels from selection.`;
    setStatus(message, false);
  }

  function updateZoomLabel() {
    if (!zoomLevelEl) {
      return;
    }
    zoomLevelEl.textContent = `${Math.round(zoom * 100)}%`;
  }

  function applyTransform() {
    stage.style.transform = `translate(${panX}px, ${panY}px) scale(${zoom})`;
    updateZoomLabel();
  }

  function clampPanToViewport() {
    if (!imageWidth || !imageHeight) {
      return;
    }
    const viewportW = viewport.clientWidth;
    const viewportH = viewport.clientHeight;
    const scaledW = imageWidth * zoom;
    const scaledH = imageHeight * zoom;

    if (scaledW <= viewportW) {
      panX = (viewportW - scaledW) / 2;
    } else {
      panX = clamp(panX, viewportW - scaledW, 0);
    }

    if (scaledH <= viewportH) {
      panY = (viewportH - scaledH) / 2;
    } else {
      panY = clamp(panY, viewportH - scaledH, 0);
    }
  }

  function fitToViewport() {
    if (!imageWidth || !imageHeight) {
      return;
    }
    const viewportW = Math.max(1, viewport.clientWidth);
    const viewportH = Math.max(1, viewport.clientHeight);
    zoom = clamp(Math.min(viewportW / imageWidth, viewportH / imageHeight), MIN_ZOOM, MAX_ZOOM);
    panX = (viewportW - imageWidth * zoom) / 2;
    panY = (viewportH - imageHeight * zoom) / 2;
    applyTransform();
  }

  function zoomAt(clientX, clientY, factor) {
    const nextZoom = clamp(zoom * factor, MIN_ZOOM, MAX_ZOOM);
    if (Math.abs(nextZoom - zoom) < 1e-6) {
      return;
    }
    const rect = viewport.getBoundingClientRect();
    const vx = clientX - rect.left;
    const vy = clientY - rect.top;
    const imageX = (vx - panX) / zoom;
    const imageY = (vy - panY) / zoom;

    zoom = nextZoom;
    panX = vx - imageX * zoom;
    panY = vy - imageY * zoom;
    clampPanToViewport();
    applyTransform();
  }

  function viewportCoordsFromEvent(event) {
    const rect = viewport.getBoundingClientRect();
    return {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
      width: rect.width,
      height: rect.height,
    };
  }

  function viewportToImage(vx, vy, clipToBounds) {
    if (!imageWidth || !imageHeight) {
      return null;
    }
    let x = Math.floor((vx - panX) / zoom);
    let y = Math.floor((vy - panY) / zoom);
    if (clipToBounds) {
      x = clamp(x, 0, imageWidth - 1);
      y = clamp(y, 0, imageHeight - 1);
      return { x, y };
    }
    if (x < 0 || y < 0 || x >= imageWidth || y >= imageHeight) {
      return null;
    }
    return { x, y };
  }

  function imageCoordinatesFromEvent(event, clipToBounds) {
    const viewportPoint = viewportCoordsFromEvent(event);
    return viewportToImage(viewportPoint.x, viewportPoint.y, clipToBounds);
  }

  function hideMarqueeRect() {
    if (!marqueeRect) {
      return;
    }
    marqueeRect.hidden = true;
    marqueeRect.style.left = "0px";
    marqueeRect.style.top = "0px";
    marqueeRect.style.width = "0px";
    marqueeRect.style.height = "0px";
  }

  function renderMarqueeRect(start, current) {
    if (!marqueeRect) {
      return;
    }
    const left = Math.min(start.x, current.x);
    const top = Math.min(start.y, current.y);
    const width = Math.abs(start.x - current.x);
    const height = Math.abs(start.y - current.y);
    marqueeRect.hidden = false;
    marqueeRect.style.left = `${left}px`;
    marqueeRect.style.top = `${top}px`;
    marqueeRect.style.width = `${width}px`;
    marqueeRect.style.height = `${height}px`;
  }

  async function refreshMaskForClass() {
    try {
      const payload = await getJSON(
        `${apiBase}/mask/${encodeURIComponent(imageName)}?class_id=${encodeURIComponent(String(currentClassId))}`
      );
      setMaskFromBase64(payload.mask_png_base64);
    } catch (err) {
      setStatus(err.message, true);
      showToast(err.message, "error");
    }
  }

  async function applyLabelAction(event, modeOverride) {
    const coords = imageCoordinatesFromEvent(event, false);
    if (!coords) {
      return;
    }
    const effectiveMode = modeOverride || mode;
    try {
      const payload = await postJSON(`${apiBase}/click`, {
        image_name: imageName,
        class_id: currentClassId,
        mode: effectiveMode,
        x: coords.x,
        y: coords.y,
      });
      setMaskFromBase64(payload.mask_png_base64);
      if (payload.changed) {
        setDirty(true);
      }
      setStatus(
        `Superpixel ${payload.superpixel_id} ${effectiveMode === "add" ? "added" : "erased"}.`,
        false
      );
    } catch (err) {
      setStatus(err.message, true);
      showToast(err.message, "error");
    }
  }

  async function applyBrushSelection(points) {
    if (!points || points.length === 0) {
      return;
    }
    try {
      const payload = await postJSON(`${apiBase}/select_ids`, {
        image_name: imageName,
        tool: "brush",
        points: points,
      });
      const added = addSelectedIds(payload.selected_ids || []);
      await refreshSelectionOverlay();
      setStatus(
        `Brush selected ${payload.count || 0} superpixels (${added} new, total ${selectedIds.size}).`,
        false
      );
    } catch (err) {
      setStatus(err.message, true);
      showToast(err.message, "error");
    }
  }

  async function applyMarqueeSelection(startViewport, endViewport) {
    if (!startViewport || !endViewport) {
      return;
    }

    const startImage = viewportToImage(startViewport.x, startViewport.y, true);
    const endImage = viewportToImage(endViewport.x, endViewport.y, true);
    if (!startImage || !endImage) {
      return;
    }

    try {
      const payload = await postJSON(`${apiBase}/select_ids`, {
        image_name: imageName,
        tool: "marquee",
        marquee_rule: marqueeRule,
        rect: {
          x1: startImage.x,
          y1: startImage.y,
          x2: endImage.x,
          y2: endImage.y,
        },
      });
      const added = addSelectedIds(payload.selected_ids || []);
      await refreshSelectionOverlay();
      setStatus(
        `Marquee selected ${payload.count || 0} superpixels (${added} new, total ${selectedIds.size}).`,
        false
      );
    } catch (err) {
      setStatus(err.message, true);
      showToast(err.message, "error");
    }
  }

  async function bulkApplySelection() {
    if (selectedIds.size === 0) {
      setStatus("No selected superpixels to apply.", true);
      return;
    }
    try {
      const payload = await postJSON(`${apiBase}/bulk_apply`, {
        image_name: imageName,
        class_id: currentClassId,
        mode: mode,
        selected_ids: sortedSelectedIds(),
      });
      setMaskFromBase64(payload.mask_png_base64);
      if (payload.changed_count > 0) {
        setDirty(true);
      }
      const message = `Applied ${payload.changed_count}/${payload.selected_count} superpixels as ${mode}.`;
      setStatus(message, false);
      showToast(message, "success");
      await clearSelection();
    } catch (err) {
      setStatus(err.message, true);
      showToast(err.message, "error");
    }
  }

  async function recomputeSuperpixels(applyRemaining, forceOverwrite) {
    const payload = currentSlicPayload(applyRemaining, forceOverwrite);
    try {
      const response = await postJSON(`${apiBase}/recompute_superpixels`, payload);
      if (response.requires_confirmation) {
        const proceed = window.confirm(
          response.message || "Recomputing may overwrite saved masks. Continue?"
        );
        if (!proceed) {
          return;
        }
        await recomputeSuperpixels(applyRemaining, true);
        return;
      }

      const boundaryBase = String(boundaryImage.getAttribute("src") || "").split("?")[0];
      if (boundaryBase) {
        boundaryImage.src = `${boundaryBase}?v=${Date.now()}`;
      }
      await refreshMaskForClass();
      await clearSelection();
      setDirty(false);
      const removed = Number(response.removed_mask_files || 0);
      const msg = removed > 0
        ? `${response.message} Removed ${removed} saved mask file(s).`
        : String(response.message || "Superpixels recomputed.");
      setStatus(msg, false);
      showToast(msg, "success");
    } catch (err) {
      setStatus(err.message, true);
      showToast(err.message, "error");
    }
  }

  async function undoAction() {
    try {
      const payload = await postJSON(`${apiBase}/undo`, {
        image_name: imageName,
        class_id: currentClassId,
      });
      setMaskFromBase64(payload.mask_png_base64);
      if (payload.undone) {
        setDirty(true);
      }
      setStatus(payload.undone ? "Undo complete." : "Nothing to undo.", false);
    } catch (err) {
      setStatus(err.message, true);
      showToast(err.message, "error");
    }
  }

  async function redoAction() {
    try {
      const payload = await postJSON(`${apiBase}/redo`, {
        image_name: imageName,
        class_id: currentClassId,
      });
      setMaskFromBase64(payload.mask_png_base64);
      if (payload.redone) {
        setDirty(true);
      }
      setStatus(payload.redone ? "Redo complete." : "Nothing to redo.", false);
    } catch (err) {
      setStatus(err.message, true);
      showToast(err.message, "error");
    }
  }

  async function saveMasks() {
    if (isSaving) {
      return;
    }
    setSaving(true);
    try {
      const payload = await postJSON(`${apiBase}/save`, { image_name: imageName });
      setDirty(false);
      setStatus(payload.message, false);
      showToast(payload.message, "success");
    } catch (err) {
      setStatus(err.message, true);
      showToast(err.message, "error");
    } finally {
      setSaving(false);
    }
  }

  function startPan(event) {
    isPanning = true;
    viewport.classList.add("is-panning");
    panOriginX = event.clientX - panX;
    panOriginY = event.clientY - panY;
    suppressNextClick = true;
  }

  function stopPan() {
    isPanning = false;
    viewport.classList.remove("is-panning");
  }

  function normalizeUrlPath(url) {
    try {
      const normalized = new URL(String(url || ""), window.location.href);
      return `${normalized.pathname}${normalized.search}${normalized.hash}`;
    } catch (err) {
      return String(url || "");
    }
  }

  function readImageNameFromItem(item) {
    if (!item) {
      return "";
    }
    const fromData = String(item.getAttribute("data-image-name") || "").trim();
    if (fromData) {
      return fromData;
    }
    return String(item.getAttribute("aria-label") || "").trim();
  }

  function findFilmstripItemByUrl(url) {
    if (!filmstripEl || !url) {
      return null;
    }
    const targetPath = normalizeUrlPath(url);
    const items = Array.from(filmstripEl.querySelectorAll(".filmstrip-item"));
    for (let i = 0; i < items.length; i += 1) {
      const item = items[i];
      const href = item.getAttribute("href");
      if (!href) {
        continue;
      }
      if (normalizeUrlPath(href) === targetPath) {
        return item;
      }
    }
    return null;
  }

  function setCurrentFilmstripItem(currentItem) {
    if (!filmstripEl) {
      return;
    }
    const items = Array.from(filmstripEl.querySelectorAll(".filmstrip-item"));
    items.forEach((item) => {
      item.classList.toggle("is-current", item === currentItem);
    });
    if (currentItem) {
      window.requestAnimationFrame(() => {
        currentItem.scrollIntoView({ block: "nearest", inline: "center" });
      });
    }
  }

  function updateKeyboardNavigationFromFilmstrip() {
    if (!filmstripEl) {
      return;
    }
    const items = Array.from(filmstripEl.querySelectorAll(".filmstrip-item"));
    const index = items.findIndex((item) => item.classList.contains("is-current"));
    if (index < 0) {
      prevImageUrl = "";
      nextImageUrl = "";
      return;
    }
    prevImageUrl = index > 0 ? String(items[index - 1].getAttribute("href") || "") : "";
    nextImageUrl = index + 1 < items.length ? String(items[index + 1].getAttribute("href") || "") : "";
  }

  function applySlicContext(nextSlic) {
    slicCurrent = nextSlic && typeof nextSlic === "object" ? nextSlic : {};
    const allowedAlgorithms = new Set(["slic", "slico", "quickshift", "felzenszwalb"]);
    const allowedPresetModes = new Set(["dataset_default", "detail", "custom"]);
    const allowedDetailLevels = new Set(["low", "medium", "high"]);

    const nextAlgorithm = normalizeSlicAlgorithm(slicCurrent.algorithm, "slic");
    slicAlgorithm = allowedAlgorithms.has(nextAlgorithm) ? nextAlgorithm : "slic";
    if (slicAlgorithmSelect) {
      slicAlgorithmSelect.value = slicAlgorithm;
    }

    const nextPresetMode = String(slicCurrent.preset_mode || "dataset_default");
    slicPresetMode = allowedPresetModes.has(nextPresetMode) ? nextPresetMode : "dataset_default";
    if (slicPresetModeSelect) {
      slicPresetModeSelect.value = slicPresetMode;
    }

    const nextDetailLevel = String(slicCurrent.detail_level || "medium");
    slicDetailLevel = allowedDetailLevels.has(nextDetailLevel) ? nextDetailLevel : "medium";
    if (slicDetailSelect) {
      slicDetailSelect.value = slicDetailLevel;
    }

    setSlicInputsFromValues(slicCurrent);
    updateSlicPresetUI();
  }

  async function switchToImage(nextImageName, targetUrl, pushHistory) {
    if (!nextImageName || isImageSwitching || nextImageName === imageName) {
      return;
    }
    if (isDirty) {
      const proceed = window.confirm(
        "You have unsaved changes on this image. Continue without saving?"
      );
      if (!proceed) {
        return;
      }
    }

    isImageSwitching = true;
    viewport.classList.add("is-loading-image");
    setStatus(`Loading ${nextImageName}...`, false);

    try {
      const context = await getJSON(`${apiBase}/context/${encodeURIComponent(nextImageName)}`);
      const contextImageName = String(context.image_name || nextImageName);

      const loadBaseImage = new Promise((resolve, reject) => {
        const onLoad = () => resolve();
        const onError = () => reject(new Error("Failed to load image."));
        baseImage.addEventListener("load", onLoad, { once: true });
        baseImage.addEventListener("error", onError, { once: true });
      });

      imageName = contextImageName;
      baseImage.src = `${apiBase}/image/${encodeURIComponent(imageName)}`;
      boundaryImage.src = `${apiBase}/boundary/${encodeURIComponent(imageName)}?v=${Date.now()}`;

      await loadBaseImage;
      clearSimselectStateForImage();
      await clearSelection();
      hideMarqueeRect();
      await refreshMaskForClass();
      setDirty(false);

      if (typeof context.prev_image_url === "string") {
        prevImageUrl = context.prev_image_url;
      }
      if (typeof context.next_image_url === "string") {
        nextImageUrl = context.next_image_url;
      }
      applySlicContext(context.slic_current || {});
      if (freezeMasksToggle && Object.prototype.hasOwnProperty.call(context, "mask_freeze_default")) {
        freezeMasksToggle.checked = Boolean(context.mask_freeze_default);
      }
      updateSelectionToolUI();
      window.requestAnimationFrame(() => {
        normalizeToolGridLayout();
      });

      const targetItem =
        findFilmstripItemByUrl(targetUrl) ||
        (filmstripEl
          ? Array.from(filmstripEl.querySelectorAll(".filmstrip-item")).find(
              (item) => readImageNameFromItem(item) === imageName
            )
          : null);
      setCurrentFilmstripItem(targetItem || null);
      updateKeyboardNavigationFromFilmstrip();

      if (imageNameEl) {
        imageNameEl.textContent = imageName;
      }

      if (pushHistory && targetUrl) {
        const nextPath = normalizeUrlPath(targetUrl);
        const currentPath = normalizeUrlPath(window.location.href);
        if (nextPath && nextPath !== currentPath) {
          window.history.pushState({ image_name: imageName }, "", nextPath);
        }
      }

      setStatus(`Loaded ${imageName}.`, false);
    } catch (err) {
      const message = err && err.message ? err.message : "Failed to switch image.";
      setStatus(message, true);
      showToast(message, "error");
      if (targetUrl) {
        window.location.href = targetUrl;
      }
    } finally {
      isImageSwitching = false;
      viewport.classList.remove("is-loading-image");
    }
  }

  function goToImage(url) {
    if (!url || isImageSwitching) {
      return;
    }
    const item = findFilmstripItemByUrl(url);
    const nextImageName = readImageNameFromItem(item);
    if (!item || !nextImageName) {
      window.location.href = url;
      return;
    }
    switchToImage(nextImageName, String(item.getAttribute("href") || url), true);
  }

  function setupFilmstrip() {
    if (!filmstripEl) {
      return;
    }
    const lazyThumbs = Array.from(filmstripEl.querySelectorAll("img.filmstrip-thumb[data-src]"));
    const revealThumb = (thumb) => {
      const source = String(thumb.getAttribute("data-src") || "").trim();
      if (!source || thumb.getAttribute("src")) {
        return;
      }
      thumb.setAttribute("src", source);
    };

    if ("IntersectionObserver" in window) {
      const observer = new IntersectionObserver(
        (entries, io) => {
          entries.forEach((entry) => {
            if (!entry.isIntersecting) {
              return;
            }
            const target = entry.target;
            if (target instanceof HTMLImageElement) {
              revealThumb(target);
            }
            io.unobserve(target);
          });
        },
        {
          root: filmstripEl,
          rootMargin: "120px",
        }
      );
      lazyThumbs.forEach((thumb) => observer.observe(thumb));
    } else {
      lazyThumbs.forEach((thumb) => revealThumb(thumb));
    }

    const currentItem = filmstripEl.querySelector(".filmstrip-item.is-current");
    if (currentItem) {
      window.requestAnimationFrame(() => {
        currentItem.scrollIntoView({ block: "nearest", inline: "center" });
      });
    }
    updateKeyboardNavigationFromFilmstrip();

    filmstripEl.addEventListener(
      "wheel",
      (event) => {
        if (Math.abs(event.deltaY) <= Math.abs(event.deltaX)) {
          return;
        }
        filmstripEl.scrollLeft += event.deltaY;
        event.preventDefault();
      },
      { passive: false }
    );

    filmstripEl.addEventListener("click", (event) => {
      const target = event.target;
      if (!(target instanceof HTMLElement)) {
        return;
      }
      const link = target.closest("a.filmstrip-item");
      if (!link) {
        return;
      }
      if (event.defaultPrevented || event.button !== 0) {
        return;
      }
      if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) {
        return;
      }
      const nextImageName = readImageNameFromItem(link);
      if (!nextImageName) {
        return;
      }
      event.preventDefault();
      switchToImage(nextImageName, String(link.getAttribute("href") || ""), true);
    });

    window.addEventListener("popstate", () => {
      window.location.reload();
    });
  }

  baseImage.addEventListener("load", () => {
    imageWidth = baseImage.naturalWidth;
    imageHeight = baseImage.naturalHeight;
    stage.style.width = `${imageWidth}px`;
    stage.style.height = `${imageHeight}px`;
    resetSimselectRoiCanvas();
    fitToViewport();
    window.requestAnimationFrame(() => {
      normalizeToolGridLayout();
    });
  });
  if (baseImage.complete && baseImage.naturalWidth > 0) {
    imageWidth = baseImage.naturalWidth;
    imageHeight = baseImage.naturalHeight;
    stage.style.width = `${imageWidth}px`;
    stage.style.height = `${imageHeight}px`;
    resetSimselectRoiCanvas();
    fitToViewport();
    window.requestAnimationFrame(() => {
      normalizeToolGridLayout();
    });
  }

  window.addEventListener("resize", () => {
    clampPanToViewport();
    applyTransform();
  });

  viewport.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      const factor = event.deltaY < 0 ? 1.1 : 1 / 1.1;
      zoomAt(event.clientX, event.clientY, factor);
    },
    { passive: false }
  );

  viewport.addEventListener("mousedown", (event) => {
    if (isImageSwitching) {
      return;
    }
    const wantsPan = event.button === 1 || (event.button === 0 && spacePressed);
    if (wantsPan) {
      event.preventDefault();
      startPan(event);
      return;
    }

    if (event.button !== 0) {
      return;
    }

    if (selectionTool === "similarity") {
      if (simselectInputMode !== "brush") {
        return;
      }
      const coords = imageCoordinatesFromEvent(event, true);
      if (!coords) {
        return;
      }
      if (!ensureSimselectRoiMask()) {
        return;
      }
      isSimselectRoiPainting = true;
      simselectRoiStrokeMoved = false;
      simselectLastPoint = { x: coords.x, y: coords.y };
      drawRoiBrushStroke(simselectLastPoint, simselectLastPoint);
      suppressNextClick = true;
      return;
    }

    if (selectionTool === "brush") {
      const coords = imageCoordinatesFromEvent(event, true);
      if (!coords) {
        return;
      }
      isBrushSelecting = true;
      brushPoints = [[coords.x, coords.y]];
      suppressNextClick = true;
      return;
    }

    if (selectionTool === "marquee") {
      const point = viewportCoordsFromEvent(event);
      const clamped = {
        x: clamp(point.x, 0, point.width),
        y: clamp(point.y, 0, point.height),
      };
      marqueeStartViewport = clamped;
      isMarqueeSelecting = true;
      suppressNextClick = true;
      renderMarqueeRect(clamped, clamped);
    }
  });

  window.addEventListener("mousemove", (event) => {
    if (isPanning) {
      panX = event.clientX - panOriginX;
      panY = event.clientY - panOriginY;
      clampPanToViewport();
      applyTransform();
      return;
    }

    if (isBrushSelecting) {
      const coords = imageCoordinatesFromEvent(event, true);
      if (!coords) {
        return;
      }
      const last = brushPoints[brushPoints.length - 1];
      if (!last || last[0] !== coords.x || last[1] !== coords.y) {
        brushPoints.push([coords.x, coords.y]);
      }
      return;
    }

    if (isSimselectRoiPainting) {
      const coords = imageCoordinatesFromEvent(event, true);
      if (!coords || !simselectLastPoint) {
        return;
      }
      if (simselectLastPoint.x !== coords.x || simselectLastPoint.y !== coords.y) {
        simselectRoiStrokeMoved = true;
        drawRoiBrushStroke(simselectLastPoint, coords);
        simselectLastPoint = { x: coords.x, y: coords.y };
      }
      return;
    }

    if (isMarqueeSelecting && marqueeStartViewport) {
      const point = viewportCoordsFromEvent(event);
      const current = {
        x: clamp(point.x, 0, point.width),
        y: clamp(point.y, 0, point.height),
      };
      renderMarqueeRect(marqueeStartViewport, current);
    }
  });

  window.addEventListener("mouseup", (event) => {
    if (isPanning) {
      stopPan();
      return;
    }

    if (isBrushSelecting) {
      isBrushSelecting = false;
      const points = brushPoints.slice();
      brushPoints = [];
      applyBrushSelection(points);
      return;
    }

    if (isSimselectRoiPainting) {
      isSimselectRoiPainting = false;
      simselectLastPoint = null;
      scheduleSimselectQuery(simselectRoiStrokeMoved ? 120 : 60);
      return;
    }

    if (isMarqueeSelecting) {
      isMarqueeSelecting = false;
      const endPoint = viewportCoordsFromEvent(event);
      const clampedEnd = {
        x: clamp(endPoint.x, 0, endPoint.width),
        y: clamp(endPoint.y, 0, endPoint.height),
      };
      const startPoint = marqueeStartViewport;
      marqueeStartViewport = null;
      hideMarqueeRect();
      applyMarqueeSelection(startPoint, clampedEnd);
    }
  });

  viewport.addEventListener("click", async (event) => {
    if (isImageSwitching) {
      return;
    }
    if (suppressNextClick) {
      suppressNextClick = false;
      return;
    }
    if (event.button !== 0) {
      return;
    }
    if (selectionTool === "similarity") {
      if (simselectInputMode === "seed") {
        await setSimselectSeedAtEvent(event);
      }
      return;
    }
    if (selectionTool !== "single") {
      return;
    }
    await applyLabelAction(event, null);
  });

  viewport.addEventListener("contextmenu", async (event) => {
    if (isImageSwitching) {
      return;
    }
    if (selectionTool !== "single") {
      return;
    }
    const coords = imageCoordinatesFromEvent(event, false);
    if (!coords) {
      return;
    }
    event.preventDefault();
    await applyLabelAction(event, "remove");
  });

  if (classSelect) {
    classSelect.value = String(currentClassId);
    classSelect.addEventListener("change", async () => {
      const nextId = Number(classSelect.value);
      if (Number.isInteger(nextId) && nextId > 0) {
        currentClassId = nextId;
      }
      await refreshMaskForClass();
      setStatus(`Class changed to ${classNameForId(currentClassId)}.`, false);
    });
  }

  if (slicPresetModeSelect) {
    slicPresetModeSelect.addEventListener("change", () => {
      slicPresetMode = String(slicPresetModeSelect.value || "dataset_default");
      updateSlicPresetUI();
    });
  }
  if (slicAlgorithmSelect) {
    slicAlgorithmSelect.addEventListener("change", () => {
      slicAlgorithm = normalizeSlicAlgorithm(slicAlgorithmSelect.value, "slic");
      updateSlicPresetUI();
    });
  }
  if (slicDetailSelect) {
    slicDetailSelect.addEventListener("change", () => {
      slicDetailLevel = String(slicDetailSelect.value || "medium");
      if (slicPresetMode === "detail") {
        applyDetailPresetInputs(slicDetailLevel);
      }
    });
  }
  if (slicNSegmentsInput) {
    slicNSegmentsInput.addEventListener("input", updateSlicSliderLabels);
  }
  if (slicCompactnessInput) {
    slicCompactnessInput.addEventListener("input", updateSlicSliderLabels);
  }
  if (slicSigmaInput) {
    slicSigmaInput.addEventListener("input", updateSlicSliderLabels);
  }
  if (quickshiftRatioInput) {
    quickshiftRatioInput.addEventListener("input", updateSlicSliderLabels);
  }
  if (quickshiftKernelSizeInput) {
    quickshiftKernelSizeInput.addEventListener("input", updateSlicSliderLabels);
  }
  if (quickshiftMaxDistInput) {
    quickshiftMaxDistInput.addEventListener("input", updateSlicSliderLabels);
  }
  if (quickshiftSigmaInput) {
    quickshiftSigmaInput.addEventListener("input", updateSlicSliderLabels);
  }
  if (felzenszwalbScaleInput) {
    felzenszwalbScaleInput.addEventListener("input", updateSlicSliderLabels);
  }
  if (felzenszwalbSigmaInput) {
    felzenszwalbSigmaInput.addEventListener("input", updateSlicSliderLabels);
  }
  if (felzenszwalbMinSizeInput) {
    felzenszwalbMinSizeInput.addEventListener("input", updateSlicSliderLabels);
  }
  if (textureEnabledInput) {
    textureEnabledInput.addEventListener("change", updateTextureControlsUI);
  }
  if (textureLbpEnabledInput) {
    textureLbpEnabledInput.addEventListener("change", updateTextureControlsUI);
  }
  if (textureGaborEnabledInput) {
    textureGaborEnabledInput.addEventListener("change", updateTextureControlsUI);
  }
  if (recomputeSlicBtn) {
    recomputeSlicBtn.addEventListener("click", () => recomputeSuperpixels(false, false));
  }
  if (resetSlicDefaultBtn) {
    resetSlicDefaultBtn.addEventListener("click", () => {
      slicPresetMode = "dataset_default";
      if (slicPresetModeSelect) {
        slicPresetModeSelect.value = "dataset_default";
      }
      updateSlicPresetUI();
      recomputeSuperpixels(false, false);
    });
  }
  if (applySlicRemainingBtn) {
    applySlicRemainingBtn.addEventListener("click", () => recomputeSuperpixels(true, false));
  }

  if (addBtn) {
    addBtn.addEventListener("click", () => setMode("add"));
  }
  if (removeBtn) {
    removeBtn.addEventListener("click", () => setMode("remove"));
  }

  selectionToolButtons.forEach((button) => {
    button.addEventListener("click", () => {
      setSelectionTool(String(button.dataset.tool || "single"));
    });
  });
  if (marqueeRuleSelect) {
    marqueeRule = marqueeRuleSelect.value || "centroid";
    marqueeRuleSelect.addEventListener("change", () => {
      marqueeRule = marqueeRuleSelect.value || "centroid";
    });
  }
  setSelectionTool(selectionTool);

  if (simselectBrushSizeInput) {
    simselectBrushSizeInput.value = "24";
    simselectBrushSizeInput.addEventListener("input", updateSimselectThresholdLabels);
  }
  if (simselectColorEnabledInput) {
    simselectColorEnabledInput.checked = Boolean(SIMSELECT_DEFAULTS.colorEnabled);
    simselectColorEnabledInput.addEventListener("change", () => scheduleSimselectQuery(60));
  }
  if (simselectTextureEnabledInput) {
    simselectTextureEnabledInput.checked = Boolean(SIMSELECT_DEFAULTS.textureEnabled);
    simselectTextureEnabledInput.addEventListener("change", () => scheduleSimselectQuery(60));
  }
  if (simselectColorThresholdInput) {
    simselectColorThresholdInput.value = String(clamp(SIMSELECT_DEFAULTS.colorThreshold, 0, 120));
    simselectColorThresholdInput.addEventListener("input", () => {
      updateSimselectThresholdLabels();
      scheduleSimselectQuery(80);
    });
  }
  if (simselectTextureThresholdInput) {
    simselectTextureThresholdInput.value = String(clamp(SIMSELECT_DEFAULTS.textureThreshold, 0, 5));
    simselectTextureThresholdInput.addEventListener("input", () => {
      updateSimselectThresholdLabels();
      scheduleSimselectQuery(80);
    });
  }
  if (simselectLbpPointsInput) {
    simselectLbpPointsInput.value = String(clamp(SIMSELECT_DEFAULTS.lbpPoints, 4, 64));
    simselectLbpPointsInput.addEventListener("change", () => scheduleSimselectQuery(120));
  }
  if (simselectLbpRadiusInput) {
    simselectLbpRadiusInput.value = String(clamp(SIMSELECT_DEFAULTS.lbpRadius, 1, 16));
    simselectLbpRadiusInput.addEventListener("change", () => scheduleSimselectQuery(120));
  }
  if (simselectLbpMethodSelect) {
    simselectLbpMethodSelect.value = normalizeSimselectLbpMethod(SIMSELECT_DEFAULTS.lbpMethod);
    simselectLbpMethodSelect.addEventListener("change", () => scheduleSimselectQuery(120));
  }
  if (simselectInputBrushBtn) {
    simselectInputBrushBtn.addEventListener("click", () => setSimselectInputMode("brush"));
  }
  if (simselectInputSeedBtn) {
    simselectInputSeedBtn.addEventListener("click", () => setSimselectInputMode("seed"));
  }
  if (simselectRoiPaintBtn) {
    simselectRoiPaintBtn.addEventListener("click", () => setSimselectRoiMode("paint"));
  }
  if (simselectRoiEraseBtn) {
    simselectRoiEraseBtn.addEventListener("click", () => setSimselectRoiMode("erase"));
  }
  if (simselectPrepareBtn) {
    simselectPrepareBtn.addEventListener("click", async () => {
      try {
        await prepareSimselectFeatures();
      } catch (err) {
        setStatus(err.message, true);
        showToast(err.message, "error");
      }
    });
  }
  if (simselectAddBtn) {
    simselectAddBtn.addEventListener("click", () => commitSimselectMatches("add"));
  }
  if (simselectSubtractBtn) {
    simselectSubtractBtn.addEventListener("click", () => commitSimselectMatches("subtract"));
  }
  if (simselectResetBtn) {
    simselectResetBtn.addEventListener("click", () => {
      clearSimselectPreview(false);
      if (simselectSeedId !== null && simselectRoiPixels > 0) {
        scheduleSimselectQuery(0);
      }
    });
  }
  if (simselectClearRoiBtn) {
    simselectClearRoiBtn.addEventListener("click", () => clearSimselectRoiMask({ silent: false }));
  }

  if (applySelectionBtn) {
    applySelectionBtn.addEventListener("click", bulkApplySelection);
  }
  if (clearSelectionBtn) {
    clearSelectionBtn.addEventListener("click", clearSelection);
  }

  if (boundaryToggle) {
    boundaryToggle.addEventListener("change", () => {
      boundaryImage.style.display = boundaryToggle.checked ? "block" : "none";
    });
  }

  if (undoBtn) {
    undoBtn.addEventListener("click", undoAction);
  }
  if (redoBtn) {
    redoBtn.addEventListener("click", redoAction);
  }

  if (saveBtn) {
    saveBtn.addEventListener("click", saveMasks);
  }

  if (exportBtn) {
    exportBtn.addEventListener("click", async () => {
      try {
        const payload = await postJSON(`${apiBase}/export_coco`, {});
        const msg = `${payload.message} (${payload.output_path})`;
        setStatus(msg, false);
        showToast(msg, "success");
      } catch (err) {
        setStatus(err.message, true);
        showToast(err.message, "error");
      }
    });
  }

  if (zoomOutBtn) {
    zoomOutBtn.addEventListener("click", () => {
      const rect = viewport.getBoundingClientRect();
      zoomAt(rect.left + rect.width / 2, rect.top + rect.height / 2, 1 / 1.15);
    });
  }
  if (zoomInBtn) {
    zoomInBtn.addEventListener("click", () => {
      const rect = viewport.getBoundingClientRect();
      zoomAt(rect.left + rect.width / 2, rect.top + rect.height / 2, 1.15);
    });
  }
  if (zoomFitBtn) {
    zoomFitBtn.addEventListener("click", fitToViewport);
  }

  document.addEventListener("keydown", (event) => {
    if (event.code === "Space" && !isFormElement(event.target)) {
      spacePressed = true;
      event.preventDefault();
    }

    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
      event.preventDefault();
      if (event.shiftKey) {
        redoAction();
      } else {
        undoAction();
      }
      return;
    }
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
      event.preventDefault();
      saveMasks();
      return;
    }

    if (isFormElement(event.target) || event.altKey || event.metaKey || event.ctrlKey) {
      return;
    }

    const lower = event.key.toLowerCase();
    if (lower === "a") {
      setMode("add");
      event.preventDefault();
      return;
    }
    if (lower === "e") {
      setMode("remove");
      event.preventDefault();
      return;
    }
    if (lower === "v") {
      setSelectionTool("single");
      event.preventDefault();
      return;
    }
    if (lower === "b") {
      if (selectionTool === "similarity") {
        setSimselectInputMode(simselectInputMode === "brush" ? "seed" : "brush");
      } else {
        setSelectionTool("brush");
      }
      event.preventDefault();
      return;
    }
    if (lower === "s") {
      setSelectionTool("similarity");
      event.preventDefault();
      return;
    }
    if (lower === "m") {
      setSelectionTool("marquee");
      event.preventDefault();
      return;
    }
    if (event.key === "Enter" && selectionTool === "similarity") {
      event.preventDefault();
      if (event.shiftKey) {
        commitSimselectMatches("subtract");
      } else {
        commitSimselectMatches("add");
      }
      return;
    }
    if (event.key === "Escape" && selectionTool === "similarity") {
      event.preventDefault();
      clearSimselectPreview(false);
      return;
    }
    if (event.key === "ArrowLeft") {
      if (prevImageUrl) {
        event.preventDefault();
        goToImage(prevImageUrl);
      }
      return;
    }
    if (event.key === "ArrowRight") {
      if (nextImageUrl) {
        event.preventDefault();
        goToImage(nextImageUrl);
      }
    }
  });

  document.addEventListener("keyup", (event) => {
    if (event.code === "Space") {
      spacePressed = false;
      if (isPanning) {
        stopPan();
      }
    }
  });

  window.addEventListener("beforeunload", (event) => {
    if (!isDirty) {
      return;
    }
    event.preventDefault();
    event.returnValue = "";
  });

  setMode("add");
  setDirty(false);
  setSaving(false);
  if (slicPresetModeSelect && ["dataset_default", "detail", "custom"].indexOf(slicPresetMode) >= 0) {
    slicPresetModeSelect.value = slicPresetMode;
  } else if (slicPresetModeSelect) {
    slicPresetModeSelect.value = "dataset_default";
    slicPresetMode = "dataset_default";
  }
  if (slicDetailSelect && ["low", "medium", "high"].indexOf(slicDetailLevel) >= 0) {
    slicDetailSelect.value = slicDetailLevel;
  }
  setSlicInputsFromValues(slicCurrent);
  updateSlicPresetUI();
  updateSelectionCount();
  setSelectionOverlay("");
  setSimselectInputMode("brush");
  setSimselectRoiMode("paint");
  updateSimselectThresholdLabels();
  clearSimselectStateForImage();
  hideMarqueeRect();
  setupFilmstrip();
  refreshMaskForClass().catch(() => {
    /* Best effort on first load. */
  });
})();
