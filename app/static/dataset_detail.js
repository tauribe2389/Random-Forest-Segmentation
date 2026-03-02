(function () {
  var doc = document;

  function asArray(nodeList) {
    return Array.prototype.slice.call(nodeList || []);
  }

  function lower(value) {
    return String(value || "").toLowerCase();
  }

  function arraysEqual(a, b) {
    if (!a || !b || a.length !== b.length) {
      return false;
    }
    for (var i = 0; i < a.length; i += 1) {
      if (a[i] !== b[i]) {
        return false;
      }
    }
    return true;
  }

  function parseCategories(rawText) {
    var text = String(rawText || "");
    var tokens = text.replace(/,/g, "\n").split(/\r?\n/);
    var seen = {};
    var parsed = [];
    tokens.forEach(function (token) {
      var item = String(token || "").trim();
      if (!item) {
        return;
      }
      var key = lower(item);
      if (seen[key]) {
        return;
      }
      seen[key] = true;
      parsed.push(item);
    });
    return parsed;
  }

  function summarizeClassDiff(current, proposed) {
    var existingByName = {};
    current.forEach(function (name, index) {
      existingByName[lower(name)] = index;
    });

    var planned = [];
    var usedExistingIds = {};
    proposed.forEach(function (name) {
      var key = lower(name);
      var existingId = existingByName[key];
      if (existingId !== undefined && !usedExistingIds[existingId]) {
        planned.push({ id: existingId, name: name, existed: true });
        usedExistingIds[existingId] = true;
        return;
      }
      planned.push({ id: null, name: name, existed: false });
    });

    var unmatchedExisting = [];
    current.forEach(function (_name, index) {
      if (!usedExistingIds[index]) {
        unmatchedExisting.push(index);
      }
    });

    var renamed = [];
    var added = [];
    var nextVirtualId = current.length;
    planned.forEach(function (item) {
      if (item.id !== null && item.id !== undefined) {
        return;
      }
      if (unmatchedExisting.length > 0) {
        var reusedId = unmatchedExisting.shift();
        item.id = reusedId;
        item.existed = true;
        var beforeName = String(current[reusedId] || "");
        if (beforeName !== item.name) {
          renamed.push({
            id: reusedId + 1,
            from: beforeName,
            to: item.name
          });
        }
        return;
      }
      item.id = nextVirtualId;
      nextVirtualId += 1;
      item.existed = false;
      added.push(item.name);
    });

    var removedIds = unmatchedExisting.slice();
    var removed = removedIds.map(function (id) {
      return current[id];
    });

    var survivingExistingOrder = [];
    for (var i = 0; i < current.length; i += 1) {
      if (removedIds.indexOf(i) < 0) {
        survivingExistingOrder.push(i);
      }
    }
    var proposedExistingOrder = planned
      .filter(function (item) {
        return !!item.existed;
      })
      .map(function (item) {
        return item.id;
      });
    var reordered = !arraysEqual(survivingExistingOrder, proposedExistingOrder);

    var appendOnly = !removed.length && !renamed.length && !reordered && proposed.length >= current.length;
    var destructive = removed.length > 0;

    return {
      added: added,
      removed: removed,
      renamed: renamed,
      reordered: reordered && !appendOnly,
      append_only: appendOnly,
      destructive: destructive
    };
  }

  function renderList(listNode, items, formatter) {
    if (!listNode) {
      return;
    }
    listNode.innerHTML = "";
    if (!items || !items.length) {
      var none = doc.createElement("li");
      none.className = "muted";
      none.textContent = "None";
      listNode.appendChild(none);
      return;
    }
    items.forEach(function (item) {
      var li = doc.createElement("li");
      li.textContent = formatter ? formatter(item) : String(item);
      listNode.appendChild(li);
    });
  }

  function setupClassEditor() {
    var panel = doc.getElementById("class-editor-panel");
    var openButton = doc.getElementById("open-class-edit-btn");
    var cancelButton = doc.getElementById("cancel-class-edit-btn");
    var form = doc.getElementById("dataset-class-form");
    var textarea = doc.getElementById("dataset-classes-text");
    if (!panel || !form || !textarea) {
      return;
    }

    var currentClasses = [];
    try {
      currentClasses = JSON.parse(form.getAttribute("data-current-classes") || "[]");
      if (!Array.isArray(currentClasses)) {
        currentClasses = [];
      }
    } catch (err) {
      currentClasses = [];
    }
    var hasMasks = form.getAttribute("data-has-masks") === "1";
    var initialTextareaValue = textarea.value;

    var addedList = doc.getElementById("class-preview-added");
    var removedList = doc.getElementById("class-preview-removed");
    var renamedList = doc.getElementById("class-preview-renamed");
    var orderText = doc.getElementById("class-preview-order");
    var warningCallout = doc.getElementById("class-destructive-warning");
    var appendOnlyNote = doc.getElementById("class-append-note");
    var destructiveGuard = doc.getElementById("class-destructive-guard");
    var confirmCheckbox = doc.getElementById("class-confirm-destructive");

    function applyPreview() {
      var proposedClasses = parseCategories(textarea.value);
      var diff = summarizeClassDiff(currentClasses, proposedClasses);

      renderList(addedList, diff.added);
      renderList(removedList, diff.removed);
      renderList(renamedList, diff.renamed, function (item) {
        return "id=" + item.id + ": " + item.from + " -> " + item.to;
      });

      if (orderText) {
        orderText.textContent = diff.reordered
          ? "Order changes detected."
          : "No order change detected.";
      }

      if (warningCallout) {
        warningCallout.hidden = !diff.destructive;
        if (diff.destructive && hasMasks) {
          warningCallout.textContent = "Class removals detected. Existing masks for removed classes will be deleted.";
        } else if (diff.destructive) {
          warningCallout.textContent = "Class removals detected. Saving will permanently delete removed-class masks.";
        }
      }

      if (appendOnlyNote) {
        appendOnlyNote.hidden = !(diff.append_only && diff.added.length > 0 && !diff.destructive);
      }

      if (destructiveGuard) {
        destructiveGuard.hidden = !diff.destructive;
      }
      if (confirmCheckbox) {
        confirmCheckbox.required = !!diff.destructive;
        if (!diff.destructive) {
          confirmCheckbox.checked = false;
        }
      }
    }

    if (openButton) {
      openButton.addEventListener("click", function () {
        panel.open = true;
        textarea.focus();
      });
    }
    if (cancelButton) {
      cancelButton.addEventListener("click", function () {
        textarea.value = initialTextareaValue;
        panel.open = false;
        applyPreview();
      });
    }

    textarea.addEventListener("input", applyPreview);
    applyPreview();
  }

  function setupDraftImages() {
    var grid = doc.getElementById("draft-images-grid");
    if (!grid) {
      return;
    }
    var cards = asArray(grid.querySelectorAll(".draft-image-card"));
    var searchInput = doc.getElementById("draft-image-search");
    var filterButtons = asArray(doc.querySelectorAll("[data-draft-filter]"));
    var bulkToggle = doc.getElementById("draft-bulk-toggle");
    var selectVisibleButton = doc.getElementById("draft-select-visible-btn");
    var visiblePill = doc.getElementById("draft-visible-pill");
    var selectedPill = doc.getElementById("draft-selected-pill");
    var bulkBar = doc.getElementById("draft-bulk-remove-form");
    var bulkCount = doc.getElementById("draft-bulk-count");
    var selectedInputs = doc.getElementById("draft-bulk-selected-inputs");
    var removeForms = asArray(doc.querySelectorAll(".single-remove-form"));
    var activeFilter = "all";

    function eachCard(action) {
      cards.forEach(function (card) {
        var checkbox = card.querySelector(".draft-image-select-checkbox");
        action(card, checkbox);
      });
    }

    function isCardVisible(card) {
      return !card.hidden;
    }

    function selectedRows() {
      return cards.filter(function (card) {
        var checkbox = card.querySelector(".draft-image-select-checkbox");
        return checkbox && checkbox.checked;
      });
    }

    function updateBulkBar() {
      var bulkEnabled = !!(bulkToggle && bulkToggle.checked);
      var selected = selectedRows();
      var visibleCount = cards.filter(isCardVisible).length;
      if (grid) {
        grid.classList.toggle("is-bulk-enabled", bulkEnabled || selected.length > 0);
      }
      if (visiblePill) {
        visiblePill.textContent = visibleCount + " / " + cards.length + " shown";
      }
      if (selectedPill) {
        selectedPill.textContent = selected.length + " selected";
      }

      if (selectedInputs) {
        selectedInputs.innerHTML = "";
        selected.forEach(function (card) {
          var checkbox = card.querySelector(".draft-image-select-checkbox");
          if (!checkbox) {
            return;
          }
          var input = doc.createElement("input");
          input.type = "hidden";
          input.name = "selected_images";
          input.value = checkbox.value;
          selectedInputs.appendChild(input);
        });
      }

      if (bulkBar) {
        bulkBar.hidden = selected.length === 0;
      }
      if (bulkCount) {
        bulkCount.textContent = selected.length + " selected";
      }
    }

    function applyFilters() {
      var searchText = lower(searchInput ? searchInput.value : "");
      cards.forEach(function (card) {
        var imageName = lower(card.getAttribute("data-image-name"));
        var isLabeled = card.getAttribute("data-labeled") === "1";
        var matchesSearch = !searchText || imageName.indexOf(searchText) >= 0;
        var matchesFilter = activeFilter === "all" ||
          (activeFilter === "labeled" && isLabeled) ||
          (activeFilter === "unlabeled" && !isLabeled);
        card.hidden = !(matchesSearch && matchesFilter);
      });
      updateBulkBar();
    }

    if (searchInput) {
      searchInput.addEventListener("input", applyFilters);
    }

    filterButtons.forEach(function (button) {
      button.addEventListener("click", function () {
        activeFilter = String(button.getAttribute("data-draft-filter") || "all");
        filterButtons.forEach(function (candidate) {
          candidate.classList.toggle("is-active", candidate === button);
        });
        applyFilters();
      });
    });

    if (bulkToggle) {
      bulkToggle.addEventListener("change", function () {
        if (!bulkToggle.checked) {
          eachCard(function (card, checkbox) {
            if (checkbox) {
              checkbox.checked = false;
            }
          });
        }
        updateBulkBar();
      });
    }

    if (selectVisibleButton) {
      selectVisibleButton.addEventListener("click", function () {
        if (bulkToggle && !bulkToggle.checked) {
          bulkToggle.checked = true;
        }
        eachCard(function (card, checkbox) {
          if (!checkbox || card.hidden) {
            return;
          }
          checkbox.checked = true;
        });
        updateBulkBar();
      });
    }

    eachCard(function (card, checkbox) {
      if (!checkbox) {
        return;
      }
      checkbox.addEventListener("change", function () {
        if (checkbox.checked && bulkToggle && !bulkToggle.checked) {
          bulkToggle.checked = true;
        }
        updateBulkBar();
      });
    });

    removeForms.forEach(function (form) {
      form.addEventListener("submit", function (event) {
        var imageName = String(form.getAttribute("data-image-name") || "this image");
        var hasMask = form.getAttribute("data-has-mask") === "1";
        var message = "Remove image '" + imageName + "' from this draft? Workspace original will remain.";
        if (hasMask) {
          message += " Associated masks will also be removed.";
        }
        if (!window.confirm(message)) {
          event.preventDefault();
        }
      });
    });

    if (bulkBar) {
      bulkBar.addEventListener("submit", function (event) {
        var selected = selectedRows();
        var hasMaskedSelection = selected.some(function (card) {
          return card.getAttribute("data-labeled") === "1";
        });
        var message = "Remove " + selected.length + " selected image(s) from this draft?";
        if (hasMaskedSelection) {
          message += " Some selected images have masks that will also be removed.";
        }
        message += " Workspace originals will not be deleted.";
        if (!window.confirm(message)) {
          event.preventDefault();
        }
      });
    }

    applyFilters();
  }

  function setupWorkspaceAdd() {
    var form = doc.getElementById("workspace-add-form");
    if (!form) {
      return;
    }
    var grid = doc.getElementById("workspace-images-grid");
    if (!grid) {
      return;
    }
    var cards = asArray(grid.querySelectorAll(".workspace-image-card"));
    var searchInput = doc.getElementById("workspace-image-search");
    var sortSelect = doc.getElementById("workspace-image-sort");
    var selectVisibleButton = doc.getElementById("workspace-select-visible-btn");
    var visiblePill = doc.getElementById("workspace-visible-pill");
    var selectedPill = doc.getElementById("workspace-selected-pill");
    var addBar = doc.getElementById("workspace-add-bar");
    var addCount = doc.getElementById("workspace-add-count");

    function selectedCount() {
      return cards.filter(function (card) {
        var checkbox = card.querySelector(".workspace-image-select");
        return checkbox && checkbox.checked;
      }).length;
    }

    function sortCards() {
      var mode = sortSelect ? String(sortSelect.value || "name_asc") : "name_asc";
      cards.sort(function (a, b) {
        var aName = lower(a.getAttribute("data-image-name"));
        var bName = lower(b.getAttribute("data-image-name"));
        if (aName < bName) {
          return mode === "name_desc" ? 1 : -1;
        }
        if (aName > bName) {
          return mode === "name_desc" ? -1 : 1;
        }
        return 0;
      });
      cards.forEach(function (card) {
        grid.appendChild(card);
      });
    }

    function applyFilterAndCounts() {
      var searchText = lower(searchInput ? searchInput.value : "");
      cards.forEach(function (card) {
        var name = lower(card.getAttribute("data-image-name"));
        var matches = !searchText || name.indexOf(searchText) >= 0;
        card.hidden = !matches;
      });

      var visibleCount = cards.filter(function (card) {
        return !card.hidden;
      }).length;
      var selected = selectedCount();

      if (visiblePill) {
        visiblePill.textContent = visibleCount + " / " + cards.length + " shown";
      }
      if (selectedPill) {
        selectedPill.textContent = selected + " selected";
      }
      if (addBar) {
        addBar.hidden = selected <= 0;
      }
      if (addCount) {
        addCount.textContent = selected + " selected";
      }
    }

    if (sortSelect) {
      sortSelect.addEventListener("change", function () {
        sortCards();
        applyFilterAndCounts();
      });
    }
    if (searchInput) {
      searchInput.addEventListener("input", applyFilterAndCounts);
    }
    if (selectVisibleButton) {
      selectVisibleButton.addEventListener("click", function () {
        cards.forEach(function (card) {
          if (card.hidden) {
            return;
          }
          var checkbox = card.querySelector(".workspace-image-select");
          if (checkbox) {
            checkbox.checked = true;
          }
        });
        applyFilterAndCounts();
      });
    }

    cards.forEach(function (card) {
      var checkbox = card.querySelector(".workspace-image-select");
      if (!checkbox) {
        return;
      }
      checkbox.addEventListener("change", applyFilterAndCounts);
    });

    sortCards();
    applyFilterAndCounts();
  }

  function setupGlobalShortcuts() {
    var draftSearch = doc.getElementById("draft-image-search");
    doc.addEventListener("keydown", function (event) {
      if (event.key !== "/" || !draftSearch || draftSearch.disabled) {
        return;
      }
      var active = doc.activeElement;
      if (active && (
        lower(active.tagName) === "input" ||
        lower(active.tagName) === "textarea" ||
        lower(active.tagName) === "select" ||
        active.isContentEditable
      )) {
        return;
      }
      event.preventDefault();
      draftSearch.focus();
      draftSearch.select();
    });
  }

  setupClassEditor();
  setupDraftImages();
  setupWorkspaceAdd();
  setupGlobalShortcuts();
})();
