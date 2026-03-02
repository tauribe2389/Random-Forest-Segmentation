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

  var searchInput = doc.getElementById("dataset-search");
  var typeFilter = doc.getElementById("dataset-type-filter");
  var statusFilter = doc.getElementById("dataset-status-filter");
  var clearFilters = doc.getElementById("clear-dataset-filters");
  var visiblePill = doc.getElementById("datasets-visible-pill");

  var tableSets = asArray(doc.querySelectorAll(".datasets-table-section")).map(function (section) {
    var table = section.querySelector("table.datasets-table");
    var tbody = table ? table.querySelector("tbody") : null;
    var rows = tbody ? asArray(tbody.querySelectorAll("tr.dataset-row")) : [];
    return {
      section: section,
      table: table,
      tbody: tbody,
      rows: rows,
      tableWrap: section.querySelector("[data-table-wrap]"),
      filterEmpty: section.querySelector("[data-filter-empty]"),
      sectionPill: section.querySelector("[data-section-visible-pill]"),
      headers: table ? asArray(table.querySelectorAll("th[data-sort-key]")) : [],
      sortState: {
        key: table ? String(table.getAttribute("data-default-sort-key") || "name") : "name",
        direction: table ? String(table.getAttribute("data-default-sort-direction") || "asc") : "asc"
      }
    };
  });

  var allRows = [];
  tableSets.forEach(function (set) {
    set.rows.forEach(function (row) {
      allRows.push(row);
    });
  });

  if (!allRows.length) {
    return;
  }

  function populateStatusFilter() {
    if (!statusFilter) {
      return;
    }
    var statuses = {};
    allRows.forEach(function (row) {
      var key = lower(row.getAttribute("data-status"));
      var label = String(row.getAttribute("data-status-label") || key);
      if (key) {
        statuses[key] = label;
      }
    });
    Object.keys(statuses).sort().forEach(function (statusKey) {
      var option = doc.createElement("option");
      option.value = statusKey;
      option.textContent = statuses[statusKey];
      statusFilter.appendChild(option);
    });
  }

  function updateHeaderSortState(set, key, direction) {
    set.headers.forEach(function (header) {
      header.classList.remove("is-sorted-asc");
      header.classList.remove("is-sorted-desc");
      if (String(header.getAttribute("data-sort-key")) === key) {
        header.classList.add(direction === "asc" ? "is-sorted-asc" : "is-sorted-desc");
      }
    });
  }

  function readSortValue(row, key) {
    if (key === "images") {
      return parseNumberValue(row.getAttribute("data-images"));
    }
    if (key === "classes") {
      return parseNumberValue(row.getAttribute("data-classes"));
    }
    if (key === "categories") {
      return parseNumberValue(row.getAttribute("data-categories"));
    }
    if (key === "updated") {
      return parseDateValue(row.getAttribute("data-updated"));
    }
    if (key === "created") {
      return parseDateValue(row.getAttribute("data-created"));
    }
    if (key === "status") {
      return lower(row.getAttribute("data-status"));
    }
    return lower(row.getAttribute("data-name"));
  }

  function sortRows(set, key, direction) {
    if (!set.tbody || !set.rows.length) {
      return;
    }
    set.rows.sort(function (a, b) {
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
    set.rows.forEach(function (row) {
      set.tbody.appendChild(row);
    });
    updateHeaderSortState(set, key, direction);
  }

  function updateVisibleCounts() {
    var visibleCount = 0;
    allRows.forEach(function (row) {
      if (!row.hidden) {
        visibleCount += 1;
      }
    });
    if (visiblePill) {
      visiblePill.textContent = visibleCount + " / " + allRows.length + " shown";
    }

    var activeType = lower(typeFilter ? typeFilter.value : "");
    tableSets.forEach(function (set) {
      var sectionKind = lower(set.section.getAttribute("data-kind"));
      var sectionVisible = 0;
      set.rows.forEach(function (row) {
        if (!row.hidden) {
          sectionVisible += 1;
        }
      });

      if (set.sectionPill) {
        set.sectionPill.textContent = sectionVisible + " / " + set.rows.length + " shown";
      }
      if (set.tableWrap) {
        set.tableWrap.hidden = sectionVisible === 0;
      }
      if (set.filterEmpty) {
        set.filterEmpty.hidden = !(set.rows.length > 0 && sectionVisible === 0);
      }
      set.section.hidden = !!activeType && activeType !== sectionKind;
    });
  }

  function applyFilters() {
    var searchText = lower(searchInput ? searchInput.value : "");
    var typeText = lower(typeFilter ? typeFilter.value : "");
    var statusText = lower(statusFilter ? statusFilter.value : "");

    allRows.forEach(function (row) {
      var rowType = lower(row.getAttribute("data-kind"));
      var rowStatus = lower(row.getAttribute("data-status"));
      var searchCorpus = lower(row.getAttribute("data-search"));
      var matchesSearch = !searchText || searchCorpus.indexOf(searchText) >= 0;
      var matchesType = !typeText || rowType === typeText;
      var matchesStatus = !statusText || rowStatus === statusText;
      row.hidden = !(matchesSearch && matchesType && matchesStatus);
    });

    updateVisibleCounts();
  }

  tableSets.forEach(function (set) {
    set.headers.forEach(function (header) {
      header.addEventListener("click", function () {
        var key = String(header.getAttribute("data-sort-key") || "name");
        if (set.sortState.key === key) {
          set.sortState.direction = set.sortState.direction === "asc" ? "desc" : "asc";
        } else {
          set.sortState.key = key;
          set.sortState.direction = key === "updated" || key === "created" ? "desc" : "asc";
        }
        sortRows(set, set.sortState.key, set.sortState.direction);
        applyFilters();
      });
    });
  });

  if (searchInput) {
    searchInput.addEventListener("input", applyFilters);
  }
  if (typeFilter) {
    typeFilter.addEventListener("change", applyFilters);
  }
  if (statusFilter) {
    statusFilter.addEventListener("change", applyFilters);
  }
  if (clearFilters) {
    clearFilters.addEventListener("click", function () {
      if (searchInput) {
        searchInput.value = "";
      }
      if (typeFilter) {
        typeFilter.value = "";
      }
      if (statusFilter) {
        statusFilter.value = "";
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
        (typeFilter && typeFilter.value) ||
        (statusFilter && statusFilter.value);
      if (!hasFilters) {
        return;
      }
      if (searchInput) {
        searchInput.value = "";
      }
      if (typeFilter) {
        typeFilter.value = "";
      }
      if (statusFilter) {
        statusFilter.value = "";
      }
      applyFilters();
    }
  });

  populateStatusFilter();
  tableSets.forEach(function (set) {
    sortRows(set, set.sortState.key, set.sortState.direction);
  });
  applyFilters();
})();
