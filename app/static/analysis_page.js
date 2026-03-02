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

  var sortState = {
    key: String(table.getAttribute("data-default-sort-key") || "created"),
    direction: String(table.getAttribute("data-default-sort-direction") || "desc")
  };

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
})();
