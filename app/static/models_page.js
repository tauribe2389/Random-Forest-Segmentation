(function () {
  var doc = document;
  var body = doc.body;

  function asArray(nodeList) {
    return Array.prototype.slice.call(nodeList || []);
  }

  function lower(value) {
    return String(value || "").toLowerCase();
  }

  function parseDateValue(value) {
    var parsed = Date.parse(String(value || ""));
    return Number.isFinite(parsed) ? parsed : 0;
  }

  var modal = doc.getElementById("train-model-modal");
  var openButtons = asArray(doc.querySelectorAll("[data-open-train-modal]"));
  var closeButton = doc.getElementById("close-train-modal");
  var form = doc.getElementById("train-model-form");
  var modelNameInput = doc.getElementById("train-model-name");
  var datasetSelect = doc.getElementById("train-dataset-id");
  var datasetContext = doc.getElementById("train-dataset-context");
  var submitButton = doc.getElementById("train-submit-btn");
  var resetButton = doc.getElementById("train-reset-btn");
  var formError = doc.getElementById("train-form-error");

  function openModal() {
    if (!modal) {
      return;
    }
    modal.hidden = false;
    body.classList.add("modal-open");
    if (modelNameInput) {
      window.setTimeout(function () {
        modelNameInput.focus();
      }, 20);
    }
    renderDatasetContext();
  }

  function closeModal() {
    if (!modal) {
      return;
    }
    modal.hidden = true;
    body.classList.remove("modal-open");
    if (formError) {
      formError.hidden = true;
      formError.textContent = "";
    }
  }

  function isTypingTarget(target) {
    if (!target) {
      return false;
    }
    var tag = String(target.tagName || "").toLowerCase();
    if (tag === "input" || tag === "textarea" || tag === "select") {
      return true;
    }
    return !!target.isContentEditable;
  }

  function getErrorSlot(name) {
    return doc.querySelector('[data-error-for="' + name + '"]');
  }

  function setFieldError(name, message) {
    var slot = getErrorSlot(name);
    if (slot) {
      slot.textContent = message || "";
    }
  }

  function readNumber(inputName) {
    if (!form) {
      return null;
    }
    var field = form.elements[inputName];
    if (!field) {
      return null;
    }
    var value = String(field.value || "").trim();
    if (!value) {
      return null;
    }
    var num = Number(value);
    return Number.isFinite(num) ? num : null;
  }

  function readText(inputName) {
    if (!form) {
      return "";
    }
    var field = form.elements[inputName];
    if (!field) {
      return "";
    }
    return String(field.value || "").trim();
  }

  function parseCsvNumbers(raw, positiveOnly, integerOnly) {
    var text = String(raw || "").trim();
    if (!text) {
      return [];
    }
    var tokens = text.split(",");
    var parsed = [];
    for (var i = 0; i < tokens.length; i += 1) {
      var token = String(tokens[i] || "").trim();
      if (!token) {
        continue;
      }
      var number = Number(token);
      if (!Number.isFinite(number)) {
        return null;
      }
      if (integerOnly && !Number.isInteger(number)) {
        return null;
      }
      if (positiveOnly && number <= 0) {
        return null;
      }
      parsed.push(number);
    }
    return parsed;
  }

  function isChecked(fieldName) {
    if (!form) {
      return false;
    }
    var field = form.elements[fieldName];
    return !!(field && field.checked);
  }

  function validateNamedCsv(name, positiveOnly, integerOnly, requiredWhenEnabled) {
    var value = readText(name);
    if (!value) {
      if (requiredWhenEnabled) {
        setFieldError(name, "Provide at least one value.");
        return false;
      }
      setFieldError(name, "");
      return true;
    }
    var parsed = parseCsvNumbers(value, positiveOnly, integerOnly);
    if (!parsed || parsed.length < 1) {
      setFieldError(name, "Use comma-separated numeric values.");
      return false;
    }
    setFieldError(name, "");
    return true;
  }

  function featureInputs() {
    if (!form) {
      return [];
    }
    return asArray(form.querySelectorAll('input[type="checkbox"][name^="use_"]'));
  }

  function renderDatasetContext() {
    if (!datasetSelect || !datasetContext) {
      return;
    }
    var selected = datasetSelect.options[datasetSelect.selectedIndex];
    if (!selected) {
      datasetContext.textContent = "Select the dataset used for this training run.";
      return;
    }
    var created = selected.getAttribute("data-created") || "-";
    var draftId = selected.getAttribute("data-draft-id") || "";
    var draftVersion = selected.getAttribute("data-draft-version") || "";
    var sourceText = "Registered dataset";
    if (draftId) {
      sourceText = "From draft #" + draftId + (draftVersion ? (" v" + draftVersion) : "");
    }
    datasetContext.textContent = sourceText + " • Created " + created;
  }

  function validateField(name) {
    if (!form) {
      return true;
    }
    var field = form.elements[name];
    if (!field) {
      return true;
    }
    var value = String(field.value || "").trim();

    if (name === "model_name") {
      if (!value) {
        setFieldError(name, "Model name is required.");
        return false;
      }
      if (value.length < 3) {
        setFieldError(name, "Use at least 3 characters.");
        return false;
      }
      setFieldError(name, "");
      return true;
    }

    if (name === "dataset_id") {
      if (!value) {
        setFieldError(name, "Dataset is required.");
        return false;
      }
      setFieldError(name, "");
      return true;
    }

    if (name === "n_estimators") {
      var estimators = readNumber(name);
      if (estimators === null || estimators < 1) {
        setFieldError(name, "Must be >= 1.");
        return false;
      }
      setFieldError(name, "");
      return true;
    }

    if (name === "min_samples_split") {
      var split = readNumber(name);
      if (split === null || split < 2) {
        setFieldError(name, "Must be >= 2.");
        return false;
      }
      setFieldError(name, "");
      return true;
    }

    if (name === "min_samples_leaf") {
      var leaf = readNumber(name);
      if (leaf === null || leaf < 1) {
        setFieldError(name, "Must be >= 1.");
        return false;
      }
      setFieldError(name, "");
      return true;
    }

    if (name === "max_depth") {
      if (!value) {
        setFieldError(name, "");
        return true;
      }
      var depth = Number(value);
      if (!Number.isFinite(depth) || depth < 1) {
        setFieldError(name, "Leave blank or use a value >= 1.");
        return false;
      }
      setFieldError(name, "");
      return true;
    }

    if (name === "max_samples_per_class") {
      var maxSamples = readNumber(name);
      if (maxSamples === null || maxSamples < 1) {
        setFieldError(name, "Must be >= 1.");
        return false;
      }
      setFieldError(name, "");
      return true;
    }

    if (name === "validation_split") {
      var validationSplit = readNumber(name);
      if (validationSplit === null || validationSplit < 0 || validationSplit > 0.95) {
        setFieldError(name, "Use a value between 0 and 0.95.");
        return false;
      }
      setFieldError(name, "");
      return true;
    }

    if (name === "gaussian_sigmas") {
      return validateNamedCsv(name, true, false, true);
    }

    if (name === "lbp_points" || name === "lbp_radii") {
      var useLbp = form.elements.use_lbp;
      if (useLbp && !useLbp.checked) {
        setFieldError(name, "");
        return true;
      }
      if (name === "lbp_points") {
        var numeric = readNumber(name);
        if (numeric === null || numeric < 1) {
          setFieldError(name, "Must be >= 1 when LBP is enabled.");
          return false;
        }
        setFieldError(name, "");
        return true;
      }
      var radii = parseCsvNumbers(readText("lbp_radii"), true, true);
      if (!radii || radii.length < 1) {
        setFieldError("lbp_radii", "Use comma-separated positive integers (for example 1,2,3).");
        return false;
      }
      setFieldError("lbp_radii", "");
      return true;
    }

    if (name === "gabor_frequencies" || name === "gabor_thetas" || name === "gabor_bandwidth") {
      var useGabor = form.elements.use_gabor;
      if (useGabor && !useGabor.checked) {
        setFieldError(name, "");
        return true;
      }
      if (name === "gabor_bandwidth") {
        var bandwidth = readNumber("gabor_bandwidth");
        if (bandwidth === null || bandwidth <= 0) {
          setFieldError("gabor_bandwidth", "Must be > 0.");
          return false;
        }
        setFieldError("gabor_bandwidth", "");
        return true;
      }
      var listValid = validateNamedCsv(name, name === "gabor_frequencies", false, true);
      if (!listValid) {
        setFieldError(name, "Use comma-separated numeric values.");
        return false;
      }
      return true;
    }

    if (name === "laws_vectors" || name === "laws_energy_window") {
      var useLaws = form.elements.use_laws;
      if (useLaws && !useLaws.checked) {
        setFieldError(name, "");
        return true;
      }
      if (name === "laws_energy_window") {
        var lawsWindow = readNumber("laws_energy_window");
        if (lawsWindow === null || lawsWindow < 1) {
          setFieldError("laws_energy_window", "Must be >= 1.");
          return false;
        }
        setFieldError("laws_energy_window", "");
        return true;
      }
      var vectorsRaw = readText("laws_vectors");
      if (!vectorsRaw) {
        setFieldError("laws_vectors", "Provide one or more vectors (L5,E5,S5,R5,W5).");
        return false;
      }
      var vectors = vectorsRaw.split(",").map(function (item) {
        return String(item || "").trim().toUpperCase();
      }).filter(function (item) {
        return !!item;
      });
      var allowed = { L5: true, E5: true, S5: true, R5: true, W5: true };
      if (!vectors.length || vectors.some(function (item) { return !allowed[item]; })) {
        setFieldError("laws_vectors", "Allowed values: L5,E5,S5,R5,W5.");
        return false;
      }
      setFieldError("laws_vectors", "");
      return true;
    }

    if (name === "structure_tensor_sigma" || name === "structure_tensor_rho") {
      var useStructure = form.elements.use_structure_tensor;
      if (useStructure && !useStructure.checked) {
        setFieldError(name, "");
        return true;
      }
      if (name === "structure_tensor_sigma") {
        var sigma = readNumber("structure_tensor_sigma");
        if (sigma === null || sigma <= 0) {
          setFieldError("structure_tensor_sigma", "Must be > 0.");
          return false;
        }
        setFieldError("structure_tensor_sigma", "");
        return true;
      }
      var rho = readNumber("structure_tensor_rho");
      if (rho === null || rho < 0) {
        setFieldError("structure_tensor_rho", "Must be >= 0.");
        return false;
      }
      setFieldError("structure_tensor_rho", "");
      return true;
    }

    if (name === "local_stats_sigmas") {
      var useLocalStats = form.elements.use_multiscale_local_stats;
      if (useLocalStats && !useLocalStats.checked) {
        setFieldError(name, "");
        return true;
      }
      return validateNamedCsv("local_stats_sigmas", true, false, true);
    }

    if (
      name === "gabor_include_real" ||
      name === "gabor_include_imag" ||
      name === "gabor_include_magnitude" ||
      name === "structure_tensor_include_eigenvalues" ||
      name === "structure_tensor_include_coherence" ||
      name === "structure_tensor_include_orientation" ||
      name === "local_stats_include_mean" ||
      name === "local_stats_include_std" ||
      name === "local_stats_include_min" ||
      name === "local_stats_include_max"
    ) {
      setFieldError(name, "");
      return true;
    }

    return true;
  }

  function validateFeatures() {
    var checked = featureInputs().filter(function (item) {
      return !!item.checked;
    }).length;
    if (checked < 1) {
      setFieldError("feature_flags", "Select at least one feature channel.");
      return false;
    }
    setFieldError("feature_flags", "");
    return true;
  }

  function validateFeatureComponentSelections() {
    var valid = true;

    if (isChecked("use_gabor")) {
      var hasGaborComponent =
        isChecked("gabor_include_real") ||
        isChecked("gabor_include_imag") ||
        isChecked("gabor_include_magnitude");
      if (!hasGaborComponent) {
        setFieldError("gabor_components", "Enable at least one Gabor response output.");
        valid = false;
      } else {
        setFieldError("gabor_components", "");
      }
    } else {
      setFieldError("gabor_components", "");
    }

    if (isChecked("use_structure_tensor")) {
      var hasStructureOutput =
        isChecked("structure_tensor_include_eigenvalues") ||
        isChecked("structure_tensor_include_coherence") ||
        isChecked("structure_tensor_include_orientation");
      if (!hasStructureOutput) {
        setFieldError("structure_tensor_outputs", "Enable at least one structure tensor output.");
        valid = false;
      } else {
        setFieldError("structure_tensor_outputs", "");
      }
    } else {
      setFieldError("structure_tensor_outputs", "");
    }

    if (isChecked("use_multiscale_local_stats")) {
      var hasLocalStatsOutput =
        isChecked("local_stats_include_mean") ||
        isChecked("local_stats_include_std") ||
        isChecked("local_stats_include_min") ||
        isChecked("local_stats_include_max");
      if (!hasLocalStatsOutput) {
        setFieldError("local_stats_outputs", "Enable at least one local stats output.");
        valid = false;
      } else {
        setFieldError("local_stats_outputs", "");
      }
    } else {
      setFieldError("local_stats_outputs", "");
    }

    return valid;
  }

  function validateForm() {
    if (!form) {
      return true;
    }
    var checks = [
      "model_name",
      "dataset_id",
      "n_estimators",
      "max_depth",
      "min_samples_split",
      "min_samples_leaf",
      "max_samples_per_class",
      "validation_split",
      "gaussian_sigmas",
      "lbp_points",
      "lbp_radii",
      "gabor_frequencies",
      "gabor_thetas",
      "gabor_bandwidth",
      "laws_vectors",
      "laws_energy_window",
      "structure_tensor_sigma",
      "structure_tensor_rho",
      "local_stats_sigmas"
    ];
    var allValid = true;
    checks.forEach(function (name) {
      if (!validateField(name)) {
        allValid = false;
      }
    });
    if (!validateFeatures()) {
      allValid = false;
    }
    if (!validateFeatureComponentSelections()) {
      allValid = false;
    }
    return allValid;
  }

  function lockSubmitState(isSubmitting) {
    if (!form || !submitButton) {
      return;
    }
    submitButton.disabled = !!isSubmitting;
    submitButton.textContent = isSubmitting ? "Queueing..." : "Queue Training Job";
    form.setAttribute("data-submitting", isSubmitting ? "true" : "false");
  }

  openButtons.forEach(function (button) {
    button.addEventListener("click", openModal);
  });
  if (closeButton) {
    closeButton.addEventListener("click", closeModal);
  }
  if (modal) {
    modal.addEventListener("click", function (event) {
      if (event.target === modal) {
        closeModal();
      }
    });
  }

  if (form) {
    renderDatasetContext();
    datasetSelect.addEventListener("change", function () {
      renderDatasetContext();
      validateField("dataset_id");
    });
    asArray(form.querySelectorAll("input, select")).forEach(function (control) {
      control.addEventListener("input", function () {
        if (control.name) {
          validateField(control.name);
        }
        if (String(control.name || "").indexOf("use_") === 0) {
          validateFeatures();
        }
        validateFeatureComponentSelections();
      });
      control.addEventListener("change", function () {
        if (control.name) {
          validateField(control.name);
        }
        if (String(control.name || "").indexOf("use_") === 0) {
          validateFeatures();
        }
        validateFeatureComponentSelections();
      });
    });
    form.addEventListener("submit", function (event) {
      if (form.getAttribute("data-submitting") === "true") {
        event.preventDefault();
        return;
      }
      if (!validateForm()) {
        event.preventDefault();
        if (formError) {
          formError.hidden = false;
          formError.textContent = "Fix validation errors before starting training.";
        }
        var firstError = form.querySelector(".field-error:not(:empty)");
        if (firstError) {
          var parentLabel = firstError.closest("label");
          var input = parentLabel ? parentLabel.querySelector("input,select,textarea") : null;
          if (input) {
            input.focus();
          }
        }
        return;
      }
      if (formError) {
        formError.hidden = true;
        formError.textContent = "";
      }
      lockSubmitState(true);
    });
    if (resetButton) {
      resetButton.addEventListener("click", function () {
        form.reset();
        asArray(form.querySelectorAll(".field-error")).forEach(function (slot) {
          slot.textContent = "";
        });
        if (formError) {
          formError.hidden = true;
          formError.textContent = "";
        }
        lockSubmitState(false);
        renderDatasetContext();
        validateFeatureComponentSelections();
      });
    }
  }

  var table = doc.getElementById("models-table");
  if (!table) {
    doc.addEventListener("keydown", function (event) {
      var typing = isTypingTarget(doc.activeElement);
      if (event.key === "Escape" && modal && !modal.hidden) {
        event.preventDefault();
        closeModal();
        return;
      }
      if (typing && (event.ctrlKey || event.metaKey) && event.key === "Enter" && form && modal && !modal.hidden) {
        event.preventDefault();
        form.requestSubmit();
        return;
      }
      if (!typing && (event.key === "n" || event.key === "N") && !event.ctrlKey && !event.metaKey && !event.altKey) {
        if (openButtons.length) {
          event.preventDefault();
          openModal();
        }
      }
    });
    return;
  }

  var tbody = table.querySelector("tbody");
  var rows = asArray(tbody.querySelectorAll("tr.model-row"));
  var searchInput = doc.getElementById("model-search");
  var statusFilter = doc.getElementById("model-status-filter");
  var datasetFilter = doc.getElementById("model-dataset-filter");
  var clearFilters = doc.getElementById("clear-model-filters");
  var selectAll = doc.getElementById("model-select-all");
  var selectionPill = doc.getElementById("model-selection-pill");
  var visiblePill = doc.getElementById("models-visible-pill");
  var sortHeaders = asArray(table.querySelectorAll("th[data-sort-key]"));
  var activeIndex = -1;
  var anchorIndex = -1;
  var sortState = { key: "created", direction: "desc" };
  var jobsPollUrl = String(table.getAttribute("data-jobs-poll-url") || "");
  var knownJobStatuses = {};
  var hasJobPollInitialized = false;

  function statusClass(statusText) {
    var key = lower(statusText);
    if (key === "trained" || key === "completed" || key === "success") {
      return "is-good";
    }
    if (key === "queued" || key === "running" || key === "training" || key === "cancel_requested") {
      return "is-progress";
    }
    if (key === "failed" || key === "error" || key === "canceled") {
      return "is-bad";
    }
    return "is-neutral";
  }

  function stageLabel(stageText) {
    var text = String(stageText || "").trim();
    if (!text) {
      return "";
    }
    return text.replace(/_/g, " ");
  }

  function showToast(message, category) {
    var wrap = doc.querySelector(".wrap");
    if (!wrap) {
      return;
    }
    var el = doc.createElement("div");
    el.className = "flash " + (category || "success");
    el.setAttribute("data-flash", "");
    el.setAttribute("data-duration", "4200");
    el.textContent = String(message || "");
    wrap.insertBefore(el, wrap.firstChild);
    window.setTimeout(function () {
      el.classList.add("hide");
      window.setTimeout(function () {
        if (el.parentNode) {
          el.parentNode.removeChild(el);
        }
      }, 420);
    }, 4200);
  }

  function updateRowStatusFromJob(job) {
    if (String(job.entity_type || "").toLowerCase() !== "model") {
      return;
    }
    var modelId = Number(job.entity_id);
    if (!Number.isInteger(modelId) || modelId <= 0) {
      return;
    }
    var row = table.querySelector('tr.model-row[data-model-id="' + String(modelId) + '"]');
    if (!row) {
      return;
    }
    var statusKey = lower(job.status);
    var label = String(job.status || "");
    var stage = stageLabel(job.stage);
    if ((statusKey === "running" || statusKey === "queued") && stage && stage !== statusKey) {
      label = String(job.status || "") + " (" + stage + ")";
    }
    row.setAttribute("data-status", statusKey);
    row.setAttribute("data-status-label", label);
    var pill = row.querySelector(".status-pill");
    if (pill) {
      pill.classList.remove("is-good", "is-progress", "is-bad", "is-neutral");
      pill.classList.add(statusClass(statusKey));
      pill.textContent = label;
    }
  }

  function pollTrainingJobs() {
    if (!jobsPollUrl) {
      return;
    }
    var url = jobsPollUrl + (jobsPollUrl.indexOf("?") >= 0 ? "&" : "?") + "limit=120";
    fetch(url, {
      headers: {
        "X-Requested-With": "XMLHttpRequest"
      }
    })
      .then(function (response) {
        if (!response.ok) {
          throw new Error("poll failed");
        }
        return response.json();
      })
      .then(function (payload) {
        var jobs = Array.isArray(payload.jobs) ? payload.jobs : [];
        var nextKnown = {};
        jobs.forEach(function (job) {
          var jobId = Number(job.id);
          if (!Number.isInteger(jobId) || jobId <= 0) {
            return;
          }
          var nextStatus = lower(job.status);
          nextKnown[jobId] = nextStatus;
          updateRowStatusFromJob(job);
          if (!hasJobPollInitialized) {
            return;
          }
          var previous = lower(knownJobStatuses[jobId] || "");
          if (!previous || previous === nextStatus) {
            return;
          }
          if (nextStatus === "running") {
            showToast("Training job #" + String(jobId) + " started.", "success");
          } else if (nextStatus === "completed") {
            showToast("Training job #" + String(jobId) + " completed.", "success");
          } else if (nextStatus === "failed") {
            showToast("Training job #" + String(jobId) + " failed.", "error");
          } else if (nextStatus === "canceled") {
            showToast("Training job #" + String(jobId) + " canceled.", "warning");
          }
        });
        knownJobStatuses = nextKnown;
        hasJobPollInitialized = true;
        populateFilters();
        applyFilters();
      })
      .catch(function () {
        // Ignore transient polling failures.
      });
  }

  function visibleRows() {
    return rows.filter(function (row) {
      return !row.hidden;
    });
  }

  function selectedRows() {
    return rows.filter(function (row) {
      var box = row.querySelector(".model-row-select");
      return box && box.checked;
    });
  }

  function setRowSelected(row, selected) {
    var box = row.querySelector(".model-row-select");
    if (!box) {
      return;
    }
    box.checked = !!selected;
    row.classList.toggle("is-selected", !!selected);
  }

  function clearAllSelections() {
    rows.forEach(function (row) {
      setRowSelected(row, false);
    });
  }

  function setActiveRow(nextIndex) {
    if (!rows.length) {
      activeIndex = -1;
      return;
    }
    if (nextIndex < 0) {
      rows.forEach(function (row) {
        row.classList.remove("is-active");
      });
      activeIndex = -1;
      return;
    }
    if (nextIndex >= rows.length) {
      nextIndex = rows.length - 1;
    }
    rows.forEach(function (row, index) {
      row.classList.toggle("is-active", index === nextIndex);
    });
    activeIndex = nextIndex;
  }

  function firstVisibleRowIndex() {
    for (var i = 0; i < rows.length; i += 1) {
      if (!rows[i].hidden) {
        return i;
      }
    }
    return -1;
  }

  function updateSelectionPills() {
    var selectedCount = selectedRows().length;
    var visibleCount = visibleRows().length;
    if (selectionPill) {
      selectionPill.textContent = selectedCount + " selected";
    }
    if (visiblePill) {
      visiblePill.textContent = visibleCount + " / " + rows.length + " shown";
    }
    if (selectAll) {
      var visible = visibleRows();
      var visibleSelected = visible.filter(function (row) {
        var box = row.querySelector(".model-row-select");
        return box && box.checked;
      }).length;
      selectAll.checked = visible.length > 0 && visibleSelected === visible.length;
    }
  }

  function applyFilters() {
    var searchText = lower(searchInput ? searchInput.value : "");
    var statusText = lower(statusFilter ? statusFilter.value : "");
    var datasetText = lower(datasetFilter ? datasetFilter.value : "");
    rows.forEach(function (row) {
      var matchesSearch =
        !searchText ||
        lower(row.getAttribute("data-name")).indexOf(searchText) >= 0 ||
        lower(row.getAttribute("data-dataset")).indexOf(searchText) >= 0 ||
        lower(row.getAttribute("data-status")).indexOf(searchText) >= 0;
      var matchesStatus = !statusText || lower(row.getAttribute("data-status")) === statusText;
      var matchesDataset = !datasetText || lower(row.getAttribute("data-dataset")) === datasetText;
      row.hidden = !(matchesSearch && matchesStatus && matchesDataset);
    });
    if (activeIndex >= 0 && activeIndex < rows.length && !rows[activeIndex].hidden) {
      setActiveRow(activeIndex);
    } else {
      setActiveRow(-1);
    }
    updateSelectionPills();
  }

  function populateFilters() {
    var statuses = {};
    var datasets = {};
    var currentStatus = statusFilter ? String(statusFilter.value || "") : "";
    var currentDataset = datasetFilter ? String(datasetFilter.value || "") : "";
    if (statusFilter) {
      statusFilter.innerHTML = '<option value="">All statuses</option>';
    }
    if (datasetFilter) {
      datasetFilter.innerHTML = '<option value="">All datasets</option>';
    }
    rows.forEach(function (row) {
      var statusKey = lower(row.getAttribute("data-status"));
      var statusLabel = String(row.getAttribute("data-status-label") || statusKey);
      var datasetKey = lower(row.getAttribute("data-dataset"));
      var datasetLabel = String(row.getAttribute("data-dataset-label") || datasetKey);
      if (statusKey) {
        statuses[statusKey] = statusLabel;
      }
      if (datasetKey) {
        datasets[datasetKey] = datasetLabel;
      }
    });
    Object.keys(statuses).sort().forEach(function (status) {
      var option = doc.createElement("option");
      option.value = status;
      option.textContent = statuses[status];
      statusFilter.appendChild(option);
    });
    Object.keys(datasets).sort().forEach(function (dataset) {
      var option = doc.createElement("option");
      option.value = dataset;
      option.textContent = datasets[dataset];
      datasetFilter.appendChild(option);
    });
    if (statusFilter && currentStatus) {
      statusFilter.value = currentStatus;
    }
    if (datasetFilter && currentDataset) {
      datasetFilter.value = currentDataset;
    }
  }

  function sortRows(key, direction) {
    rows.sort(function (a, b) {
      var aVal = "";
      var bVal = "";
      if (key === "created") {
        aVal = parseDateValue(a.getAttribute("data-created"));
        bVal = parseDateValue(b.getAttribute("data-created"));
      } else if (key === "dataset") {
        aVal = lower(a.getAttribute("data-dataset"));
        bVal = lower(b.getAttribute("data-dataset"));
      } else if (key === "status") {
        aVal = lower(a.getAttribute("data-status"));
        bVal = lower(b.getAttribute("data-status"));
      } else {
        aVal = lower(a.getAttribute("data-name"));
        bVal = lower(b.getAttribute("data-name"));
      }
      if (aVal < bVal) {
        return direction === "asc" ? -1 : 1;
      }
      if (aVal > bVal) {
        return direction === "asc" ? 1 : -1;
      }
      return 0;
    });
    rows.forEach(function (row) {
      tbody.appendChild(row);
    });
    sortHeaders.forEach(function (header) {
      header.classList.remove("is-sorted-asc");
      header.classList.remove("is-sorted-desc");
      if (header.getAttribute("data-sort-key") === key) {
        header.classList.add(direction === "asc" ? "is-sorted-asc" : "is-sorted-desc");
      }
    });
    applyFilters();
  }

  rows.forEach(function (row, index) {
    var checkbox = row.querySelector(".model-row-select");
    row.addEventListener("click", function (event) {
      var target = event.target;
      var targetTag = target ? String(target.tagName || "").toLowerCase() : "";
      if (targetTag === "a" || targetTag === "button") {
        return;
      }
      if (target && target.classList && target.classList.contains("model-row-select")) {
        row.classList.toggle("is-selected", !!target.checked);
        anchorIndex = index;
        setActiveRow(index);
        updateSelectionPills();
        return;
      }
      if (event.shiftKey && anchorIndex >= 0) {
        var minIndex = Math.min(anchorIndex, index);
        var maxIndex = Math.max(anchorIndex, index);
        clearAllSelections();
        for (var i = minIndex; i <= maxIndex; i += 1) {
          if (!rows[i].hidden) {
            setRowSelected(rows[i], true);
          }
        }
      } else if (event.ctrlKey || event.metaKey) {
        setRowSelected(row, !checkbox.checked);
        anchorIndex = index;
      } else {
        clearAllSelections();
        setRowSelected(row, true);
        anchorIndex = index;
      }
      setActiveRow(index);
      updateSelectionPills();
    });
    row.addEventListener("focus", function () {
      setActiveRow(index);
    });
  });

  sortHeaders.forEach(function (header) {
    header.addEventListener("click", function () {
      var key = String(header.getAttribute("data-sort-key") || "created");
      if (sortState.key === key) {
        sortState.direction = sortState.direction === "asc" ? "desc" : "asc";
      } else {
        sortState.key = key;
        sortState.direction = key === "created" ? "desc" : "asc";
      }
      sortRows(sortState.key, sortState.direction);
    });
  });

  if (selectAll) {
    selectAll.addEventListener("change", function () {
      visibleRows().forEach(function (row) {
        setRowSelected(row, selectAll.checked);
      });
      updateSelectionPills();
    });
  }

  searchInput.addEventListener("input", applyFilters);
  statusFilter.addEventListener("change", applyFilters);
  datasetFilter.addEventListener("change", applyFilters);
  clearFilters.addEventListener("click", function () {
    searchInput.value = "";
    statusFilter.value = "";
    datasetFilter.value = "";
    applyFilters();
  });

  function moveActiveRow(delta) {
    if (!rows.length) {
      return;
    }
    if (activeIndex < 0) {
      var first = firstVisibleRowIndex();
      if (first < 0) {
        return;
      }
      setActiveRow(first);
      rows[first].focus();
      rows[first].scrollIntoView({ block: "nearest" });
      return;
    }
    var current = activeIndex;
    var next = current;
    do {
      next += delta > 0 ? 1 : -1;
    } while (next >= 0 && next < rows.length && rows[next].hidden);
    if (next >= 0 && next < rows.length) {
      setActiveRow(next);
      rows[next].focus();
      rows[next].scrollIntoView({ block: "nearest" });
    }
  }

  function openActiveRowUrl(type) {
    if (activeIndex < 0 || !rows[activeIndex] || rows[activeIndex].hidden) {
      return;
    }
    var key = type === "analysis" ? "data-analysis-url" : "data-details-url";
    var url = rows[activeIndex].getAttribute(key);
    if (url) {
      window.location.assign(url);
    }
  }

  doc.addEventListener("keydown", function (event) {
    var typing = isTypingTarget(doc.activeElement);
    if (event.key === "Escape") {
      if (modal && !modal.hidden) {
        event.preventDefault();
        closeModal();
        return;
      }
      if (!typing) {
        clearAllSelections();
        updateSelectionPills();
      }
    }
    if (typing) {
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter" && form && modal && !modal.hidden) {
        event.preventDefault();
        form.requestSubmit();
      }
      return;
    }
    if (modal && !modal.hidden) {
      return;
    }
    if (event.key === "/") {
      event.preventDefault();
      searchInput.focus();
      searchInput.select();
      return;
    }
    if ((event.key === "n" || event.key === "N") && !event.ctrlKey && !event.metaKey && !event.altKey) {
      if (openButtons.length) {
        event.preventDefault();
        openModal();
      }
      return;
    }
    if (event.key === "j" || event.key === "J") {
      event.preventDefault();
      moveActiveRow(1);
      return;
    }
    if (event.key === "k" || event.key === "K") {
      event.preventDefault();
      moveActiveRow(-1);
      return;
    }
    if (event.key === "Enter") {
      event.preventDefault();
      if (event.shiftKey) {
        openActiveRowUrl("analysis");
      } else {
        openActiveRowUrl("details");
      }
    }
  });

  populateFilters();
  sortRows(sortState.key, sortState.direction);
  updateSelectionPills();
  pollTrainingJobs();
  window.setInterval(function () {
    if (doc.hidden) {
      return;
    }
    pollTrainingJobs();
  }, 10000);
})();
