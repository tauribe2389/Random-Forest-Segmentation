(function () {
  var doc = document;

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

  function parseNumberValue(value) {
    var parsed = Number(String(value || "").replace(/,/g, ""));
    return Number.isFinite(parsed) ? parsed : 0;
  }

  function isTypingTarget(target) {
    if (!target) {
      return false;
    }
    var tag = lower(target.tagName);
    if (tag === "input" || tag === "textarea" || tag === "select") {
      return true;
    }
    return !!target.isContentEditable;
  }

  var table = doc.getElementById("analysis-runs-table");
  if (!table) {
    return;
  }

  var tbody = table.querySelector("tbody");
  var rows = asArray(tbody ? tbody.querySelectorAll("tr.analysis-run-row") : []);
  if (!rows.length) {
    return;
  }

  var searchInput = doc.getElementById("analysis-search");
  var statusFilter = doc.getElementById("analysis-status-filter");
  var modelFilter = doc.getElementById("analysis-model-filter");
  var refinementFilter = doc.getElementById("analysis-refinement-filter");
  var clearFilters = doc.getElementById("analysis-clear-filters");
  var visiblePill = doc.getElementById("analysis-visible-pill");
  var emptyState = doc.getElementById("analysis-filter-empty");
  var sortHeaders = asArray(table.querySelectorAll("th[data-sort-key]"));
  var jobsPollUrl = String(table.getAttribute("data-jobs-poll-url") || "");
  var knownJobStatuses = {};
  var hasJobPollInitialized = false;

  var sortState = {
    key: String(table.getAttribute("data-default-sort-key") || "created"),
    direction: String(table.getAttribute("data-default-sort-direction") || "desc")
  };

  function statusClass(statusText) {
    var key = lower(statusText);
    if (key === "completed" || key === "trained" || key === "success") {
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

  function stageLabel(stageText) {
    var text = String(stageText || "").trim();
    if (!text) {
      return "";
    }
    return text.replace(/_/g, " ");
  }

  function updateHeaderSortState(key, direction) {
    sortHeaders.forEach(function (header) {
      header.classList.remove("is-sorted-asc");
      header.classList.remove("is-sorted-desc");
      if (String(header.getAttribute("data-sort-key")) === key) {
        header.classList.add(direction === "asc" ? "is-sorted-asc" : "is-sorted-desc");
      }
    });
  }

  function readSortValue(row, key) {
    if (key === "run_id") {
      return parseNumberValue(row.getAttribute("data-run-id"));
    }
    if (key === "created") {
      return parseDateValue(row.getAttribute("data-created"));
    }
    if (key === "model") {
      return lower(row.getAttribute("data-model"));
    }
    if (key === "images") {
      return parseNumberValue(row.getAttribute("data-images"));
    }
    if (key === "status") {
      return lower(row.getAttribute("data-status"));
    }
    if (key === "refinement") {
      return lower(row.getAttribute("data-refinement"));
    }
    return lower(row.getAttribute("data-search"));
  }

  function sortRows(key, direction) {
    rows.sort(function (a, b) {
      var aVal = readSortValue(a, key);
      var bVal = readSortValue(b, key);
      if (aVal < bVal) {
        return direction === "asc" ? -1 : 1;
      }
      if (aVal > bVal) {
        return direction === "asc" ? 1 : -1;
      }
      return 0;
    });
    rows.forEach(function (row) {
      if (tbody) {
        tbody.appendChild(row);
      }
    });
    updateHeaderSortState(key, direction);
  }

  function populateFilters() {
    var statuses = {};
    var models = {};
    var currentStatus = statusFilter ? String(statusFilter.value || "") : "";
    var currentModel = modelFilter ? String(modelFilter.value || "") : "";
    if (statusFilter) {
      statusFilter.innerHTML = '<option value="">All statuses</option>';
    }
    if (modelFilter) {
      modelFilter.innerHTML = '<option value="">All models</option>';
    }
    rows.forEach(function (row) {
      var statusKey = lower(row.getAttribute("data-status"));
      var statusLabel = String(row.getAttribute("data-status-label") || statusKey);
      var modelKey = lower(row.getAttribute("data-model"));
      var modelLabel = String(row.getAttribute("data-model-label") || modelKey);
      if (statusKey) {
        statuses[statusKey] = statusLabel;
      }
      if (modelKey) {
        models[modelKey] = modelLabel;
      }
    });

    if (statusFilter) {
      Object.keys(statuses).sort().forEach(function (statusKey) {
        var option = doc.createElement("option");
        option.value = statusKey;
        option.textContent = statuses[statusKey];
        statusFilter.appendChild(option);
      });
    }
    if (modelFilter) {
      Object.keys(models).sort().forEach(function (modelKey) {
        var option = doc.createElement("option");
        option.value = modelKey;
        option.textContent = models[modelKey];
        modelFilter.appendChild(option);
      });
    }
    if (statusFilter && currentStatus) {
      statusFilter.value = currentStatus;
    }
    if (modelFilter && currentModel) {
      modelFilter.value = currentModel;
    }
  }

  function updateVisibleCount() {
    var visible = 0;
    rows.forEach(function (row) {
      if (!row.hidden) {
        visible += 1;
      }
    });
    if (visiblePill) {
      visiblePill.textContent = visible + " / " + rows.length + " shown";
    }
    if (emptyState) {
      emptyState.hidden = visible > 0;
    }
  }

  function applyFilters() {
    var searchText = lower(searchInput ? searchInput.value : "");
    var statusText = lower(statusFilter ? statusFilter.value : "");
    var modelText = lower(modelFilter ? modelFilter.value : "");
    var refinementText = lower(refinementFilter ? refinementFilter.value : "");

    rows.forEach(function (row) {
      var corpus = lower(row.getAttribute("data-search"));
      var rowStatus = lower(row.getAttribute("data-status"));
      var rowModel = lower(row.getAttribute("data-model"));
      var rowRefinement = lower(row.getAttribute("data-refinement"));

      var matchesSearch = !searchText || corpus.indexOf(searchText) >= 0;
      var matchesStatus = !statusText || rowStatus === statusText;
      var matchesModel = !modelText || rowModel === modelText;
      var matchesRefinement = !refinementText || rowRefinement === refinementText;

      row.hidden = !(matchesSearch && matchesStatus && matchesModel && matchesRefinement);
    });

    updateVisibleCount();
  }

  function updateRowStatusFromJob(job) {
    if (String(job.entity_type || "").toLowerCase() !== "analysis_run") {
      return;
    }
    var runId = Number(job.entity_id);
    if (!Number.isInteger(runId) || runId <= 0) {
      return;
    }
    var row = table.querySelector('tr.analysis-run-row[data-run-id="' + String(runId) + '"]');
    if (!row) {
      return;
    }
    var statusKey = lower(job.status);
    var statusLabel = String(job.status || "");
    var stage = stageLabel(job.stage);
    if ((statusKey === "queued" || statusKey === "running") && stage && stage !== statusKey) {
      statusLabel = String(job.status || "") + " (" + stage + ")";
    }
    row.setAttribute("data-status", statusKey);
    row.setAttribute("data-status-label", statusLabel);
    var statusPill = row.querySelector(".status-pill");
    if (statusPill) {
      statusPill.classList.remove("is-good", "is-progress", "is-bad", "is-neutral");
      statusPill.classList.add(statusClass(statusKey));
      statusPill.textContent = statusLabel;
    }
  }

  function pollAnalysisJobs() {
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
            showToast("Analysis job #" + String(jobId) + " started.", "success");
          } else if (nextStatus === "completed") {
            showToast("Analysis job #" + String(jobId) + " completed.", "success");
          } else if (nextStatus === "failed") {
            showToast("Analysis job #" + String(jobId) + " failed.", "error");
          } else if (nextStatus === "canceled") {
            showToast("Analysis job #" + String(jobId) + " canceled.", "warning");
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
      applyFilters();
    });
  });

  if (searchInput) {
    searchInput.addEventListener("input", applyFilters);
  }
  if (statusFilter) {
    statusFilter.addEventListener("change", applyFilters);
  }
  if (modelFilter) {
    modelFilter.addEventListener("change", applyFilters);
  }
  if (refinementFilter) {
    refinementFilter.addEventListener("change", applyFilters);
  }
  if (clearFilters) {
    clearFilters.addEventListener("click", function () {
      if (searchInput) {
        searchInput.value = "";
      }
      if (statusFilter) {
        statusFilter.value = "";
      }
      if (modelFilter) {
        modelFilter.value = "";
      }
      if (refinementFilter) {
        refinementFilter.value = "";
      }
      applyFilters();
    });
  }

  doc.addEventListener("keydown", function (event) {
    if (isTypingTarget(doc.activeElement)) {
      return;
    }
    if (event.key === "/") {
      if (!searchInput || searchInput.disabled) {
        return;
      }
      event.preventDefault();
      searchInput.focus();
      searchInput.select();
      return;
    }
    if (event.key === "Escape") {
      var hasFilters =
        (searchInput && searchInput.value) ||
        (statusFilter && statusFilter.value) ||
        (modelFilter && modelFilter.value) ||
        (refinementFilter && refinementFilter.value);
      if (!hasFilters) {
        return;
      }
      if (searchInput) {
        searchInput.value = "";
      }
      if (statusFilter) {
        statusFilter.value = "";
      }
      if (modelFilter) {
        modelFilter.value = "";
      }
      if (refinementFilter) {
        refinementFilter.value = "";
      }
      applyFilters();
    }
  });

  populateFilters();
  sortRows(sortState.key, sortState.direction);
  applyFilters();
  pollAnalysisJobs();
  window.setInterval(function () {
    if (doc.hidden) {
      return;
    }
    pollAnalysisJobs();
  }, 10000);
})();
