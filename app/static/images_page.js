(function () {
  var doc = document;

  function asArray(nodeList) {
    return Array.prototype.slice.call(nodeList || []);
  }

  function lower(value) {
    return String(value || "").toLowerCase();
  }

  function formatBytes(bytes) {
    var value = Number(bytes);
    if (!Number.isFinite(value) || value <= 0) {
      return "0 B";
    }
    var units = ["B", "KB", "MB", "GB"];
    var unitIndex = 0;
    while (value >= 1024 && unitIndex < units.length - 1) {
      value /= 1024;
      unitIndex += 1;
    }
    var rounded = unitIndex === 0 ? Math.round(value) : value.toFixed(1);
    return String(rounded) + " " + units[unitIndex];
  }

  function sortCards(cards, sortValue) {
    var key = String(sortValue || "modified_desc");
    cards.sort(function (a, b) {
      var aName = lower(a.getAttribute("data-name"));
      var bName = lower(b.getAttribute("data-name"));
      var aSize = Number(a.getAttribute("data-size")) || 0;
      var bSize = Number(b.getAttribute("data-size")) || 0;
      var aModified = Number(a.getAttribute("data-modified")) || 0;
      var bModified = Number(b.getAttribute("data-modified")) || 0;
      if (key === "name_asc") {
        return aName.localeCompare(bName);
      }
      if (key === "name_desc") {
        return bName.localeCompare(aName);
      }
      if (key === "size_asc") {
        return aSize - bSize;
      }
      if (key === "size_desc") {
        return bSize - aSize;
      }
      if (key === "modified_asc") {
        return aModified - bModified;
      }
      return bModified - aModified;
    });
  }

  var uploadAccordion = doc.getElementById("images-upload-accordion");
  var openUploadButtons = asArray(doc.querySelectorAll("[data-open-upload]"));
  var fileInput = doc.getElementById("images-file-input");
  var folderInput = doc.getElementById("images-folder-input");
  var folderButton = doc.getElementById("images-folder-btn");
  var dropzone = doc.getElementById("images-dropzone");
  var uploadSelection = doc.getElementById("images-upload-selection");
  var inspectModal = doc.getElementById("images-inspect-modal");
  var inspectClose = doc.getElementById("images-inspect-close");
  var inspectDismiss = doc.getElementById("images-inspect-dismiss");
  var inspectPreview = doc.getElementById("images-inspect-preview");
  var inspectName = doc.getElementById("images-inspect-name");
  var inspectDimensions = doc.getElementById("images-inspect-dimensions");
  var inspectSize = doc.getElementById("images-inspect-size");
  var inspectType = doc.getElementById("images-inspect-type");
  var inspectModified = doc.getElementById("images-inspect-modified");
  var inspectPreviewWrap = doc.getElementById("images-inspect-preview-wrap");
  var zoomOutButton = doc.getElementById("images-zoom-out");
  var zoomRange = doc.getElementById("images-zoom-range");
  var zoomInButton = doc.getElementById("images-zoom-in");
  var zoomFitButton = doc.getElementById("images-zoom-fit");
  var zoomActualButton = doc.getElementById("images-zoom-actual");
  var zoomLabel = doc.getElementById("images-zoom-label");
  var currentInspectCard = null;
  var inspectZoom = 1;
  var inspectNaturalWidth = 0;
  var inspectNaturalHeight = 0;
  var shouldFitOnNextLoad = false;
  var ZOOM_MIN = 0.2;
  var ZOOM_MAX = 8;
  var ZOOM_STEP = 0.15;

  function forceUploadPanelOpen() {
    if (!uploadAccordion) {
      return;
    }
    var tagName = String(uploadAccordion.tagName || "").toLowerCase();
    if (tagName === "details") {
      uploadAccordion.open = true;
    }
  }

  forceUploadPanelOpen();
  if (uploadAccordion && String(uploadAccordion.tagName || "").toLowerCase() === "details") {
    uploadAccordion.addEventListener("toggle", function () {
      if (!uploadAccordion.open) {
        uploadAccordion.open = true;
      }
    });
  }

  function openUploadPanel() {
    forceUploadPanelOpen();
    if (uploadAccordion) {
      uploadAccordion.scrollIntoView({ behavior: "smooth", block: "start" });
    }
    if (fileInput) {
      fileInput.focus();
    }
  }

  openUploadButtons.forEach(function (button) {
    button.addEventListener("click", openUploadPanel);
  });

  function updateUploadSelection() {
    if (!uploadSelection) {
      return;
    }
    var fileCount = fileInput && fileInput.files ? fileInput.files.length : 0;
    var folderCount = folderInput && folderInput.files ? folderInput.files.length : 0;
    var total = fileCount + folderCount;
    if (total <= 0) {
      uploadSelection.textContent = "No files selected yet.";
      return;
    }
    uploadSelection.textContent = String(total) + " file(s) selected.";
  }

  if (folderButton && folderInput) {
    folderButton.addEventListener("click", function () {
      folderInput.click();
    });
  }
  if (fileInput) {
    fileInput.addEventListener("change", updateUploadSelection);
  }
  if (folderInput) {
    folderInput.addEventListener("change", updateUploadSelection);
  }

  function clearDropzoneState() {
    if (dropzone) {
      dropzone.classList.remove("is-dragover");
    }
  }

  if (dropzone && fileInput) {
    dropzone.addEventListener("click", function () {
      fileInput.click();
    });
    dropzone.addEventListener("keydown", function (event) {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        fileInput.click();
      }
    });
    dropzone.addEventListener("dragenter", function (event) {
      event.preventDefault();
      dropzone.classList.add("is-dragover");
    });
    dropzone.addEventListener("dragover", function (event) {
      event.preventDefault();
      dropzone.classList.add("is-dragover");
    });
    dropzone.addEventListener("dragleave", function (event) {
      if (event.target === dropzone) {
        clearDropzoneState();
      }
    });
    dropzone.addEventListener("drop", function (event) {
      event.preventDefault();
      clearDropzoneState();
      var dt = event.dataTransfer;
      if (!dt || !dt.files || !dt.files.length) {
        return;
      }
      try {
        fileInput.files = dt.files;
      } catch (err) {
        // Some browsers prevent direct assignment; keep drop affordance anyway.
      }
      updateUploadSelection();
    });
  }

  function isInspectOpen() {
    return Boolean(inspectModal && !inspectModal.hidden);
  }

  function clampZoom(value) {
    var numeric = Number(value);
    if (!Number.isFinite(numeric)) {
      return 1;
    }
    return Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, numeric));
  }

  function setInspectNaturalSize(width, height) {
    var w = Number(width) || 0;
    var h = Number(height) || 0;
    inspectNaturalWidth = w > 0 ? w : 0;
    inspectNaturalHeight = h > 0 ? h : 0;
  }

  function updateZoomReadout() {
    var zoomPercent = Math.round(inspectZoom * 100);
    if (zoomLabel) {
      zoomLabel.textContent = String(zoomPercent) + "%";
    }
    if (zoomRange) {
      zoomRange.value = String(zoomPercent);
    }
  }

  function applyInspectZoom() {
    if (!inspectPreview) {
      return;
    }
    if (inspectNaturalWidth > 0 && inspectNaturalHeight > 0) {
      inspectPreview.style.width = String(Math.round(inspectNaturalWidth * inspectZoom)) + "px";
      inspectPreview.style.height = String(Math.round(inspectNaturalHeight * inspectZoom)) + "px";
    } else {
      inspectPreview.style.removeProperty("width");
      inspectPreview.style.removeProperty("height");
    }
    updateZoomReadout();
  }

  function setInspectZoom(nextZoom, preserveCenter) {
    var previousZoom = inspectZoom;
    inspectZoom = clampZoom(nextZoom);
    if (
      preserveCenter &&
      inspectPreviewWrap &&
      inspectNaturalWidth > 0 &&
      inspectNaturalHeight > 0
    ) {
      var centerX = inspectPreviewWrap.scrollLeft + (inspectPreviewWrap.clientWidth / 2);
      var centerY = inspectPreviewWrap.scrollTop + (inspectPreviewWrap.clientHeight / 2);
      var imageCenterX = centerX / previousZoom;
      var imageCenterY = centerY / previousZoom;
      applyInspectZoom();
      inspectPreviewWrap.scrollLeft = Math.max(0, (imageCenterX * inspectZoom) - (inspectPreviewWrap.clientWidth / 2));
      inspectPreviewWrap.scrollTop = Math.max(0, (imageCenterY * inspectZoom) - (inspectPreviewWrap.clientHeight / 2));
    } else {
      applyInspectZoom();
    }
    if (inspectPreviewWrap) {
      inspectPreviewWrap.setAttribute("data-zoom", String(inspectZoom));
    }
  }

  function fitInspectZoom() {
    if (!inspectPreviewWrap || inspectNaturalWidth <= 0 || inspectNaturalHeight <= 0) {
      setInspectZoom(1, false);
      return;
    }
    var fitWidth = inspectPreviewWrap.clientWidth / inspectNaturalWidth;
    var fitHeight = inspectPreviewWrap.clientHeight / inspectNaturalHeight;
    var fitZoom = clampZoom(Math.min(fitWidth, fitHeight));
    setInspectZoom(fitZoom, false);
    inspectPreviewWrap.scrollLeft = 0;
    inspectPreviewWrap.scrollTop = 0;
  }

  function closeInspectModal() {
    if (!inspectModal) {
      return;
    }
    inspectModal.hidden = true;
    inspectModal.setAttribute("aria-hidden", "true");
    doc.body.classList.remove("modal-open");
    currentInspectCard = null;
    inspectZoom = 1;
    setInspectNaturalSize(0, 0);
    shouldFitOnNextLoad = false;
    if (inspectPreview) {
      inspectPreview.removeAttribute("src");
      inspectPreview.alt = "";
      inspectPreview.style.removeProperty("width");
      inspectPreview.style.removeProperty("height");
    }
    if (inspectPreviewWrap) {
      inspectPreviewWrap.scrollLeft = 0;
      inspectPreviewWrap.scrollTop = 0;
      inspectPreviewWrap.setAttribute("data-zoom", "1");
    }
    updateZoomReadout();
  }

  function openInspectModal(card) {
    if (!inspectModal || !card) {
      return;
    }
    currentInspectCard = card;
    var imageName = String(card.getAttribute("data-image-name") || "");
    var imageUrl = String(card.getAttribute("data-image-url") || "");
    var imageExt = String(card.getAttribute("data-extension") || "").toLowerCase();
    var imageSize = formatBytes(card.getAttribute("data-size"));
    var modifiedLabel = String(card.getAttribute("data-modified-label") || "");
    var width = Number(card.getAttribute("data-width")) || 0;
    var height = Number(card.getAttribute("data-height")) || 0;
    setInspectNaturalSize(width, height);
    shouldFitOnNextLoad = true;

    if (inspectName) {
      inspectName.textContent = imageName || "-";
    }
    if (inspectSize) {
      inspectSize.textContent = imageSize;
    }
    if (inspectType) {
      inspectType.textContent = imageExt ? "." + imageExt : "-";
    }
    if (inspectModified) {
      inspectModified.textContent = modifiedLabel || "-";
    }
    if (inspectDimensions) {
      inspectDimensions.textContent = width > 0 && height > 0 ? String(width) + " x " + String(height) : "Loading...";
    }
    if (inspectPreview) {
      inspectPreview.src = imageUrl;
      inspectPreview.alt = imageName;
    }

    inspectModal.hidden = false;
    inspectModal.setAttribute("aria-hidden", "false");
    doc.body.classList.add("modal-open");
    if (inspectNaturalWidth > 0 && inspectNaturalHeight > 0) {
      window.requestAnimationFrame(function () {
        fitInspectZoom();
      });
    } else {
      setInspectZoom(1, false);
    }
    if (inspectClose) {
      inspectClose.focus();
    }
  }

  if (inspectClose) {
    inspectClose.addEventListener("click", closeInspectModal);
  }
  if (inspectDismiss) {
    inspectDismiss.addEventListener("click", closeInspectModal);
  }
  if (inspectModal) {
    inspectModal.addEventListener("click", function (event) {
      if (event.target === inspectModal) {
        closeInspectModal();
      }
    });
  }
  if (inspectPreview) {
    inspectPreview.addEventListener("load", function () {
      if (!currentInspectCard || !inspectDimensions) {
        return;
      }
      var width = inspectPreview.naturalWidth;
      var height = inspectPreview.naturalHeight;
      if (width > 0 && height > 0) {
        currentInspectCard.setAttribute("data-width", String(width));
        currentInspectCard.setAttribute("data-height", String(height));
        inspectDimensions.textContent = String(width) + " x " + String(height);
        setInspectNaturalSize(width, height);
        if (shouldFitOnNextLoad) {
          fitInspectZoom();
          shouldFitOnNextLoad = false;
        } else {
          applyInspectZoom();
        }
      }
    });
    inspectPreview.addEventListener("error", function () {
      if (inspectDimensions) {
        inspectDimensions.textContent = "Preview unavailable";
      }
      setInspectNaturalSize(0, 0);
      shouldFitOnNextLoad = false;
      setInspectZoom(1, false);
    });
  }

  if (zoomOutButton) {
    zoomOutButton.addEventListener("click", function () {
      setInspectZoom(inspectZoom - ZOOM_STEP, true);
    });
  }
  if (zoomInButton) {
    zoomInButton.addEventListener("click", function () {
      setInspectZoom(inspectZoom + ZOOM_STEP, true);
    });
  }
  if (zoomFitButton) {
    zoomFitButton.addEventListener("click", function () {
      fitInspectZoom();
    });
  }
  if (zoomActualButton) {
    zoomActualButton.addEventListener("click", function () {
      setInspectZoom(1, false);
      if (inspectPreviewWrap) {
        inspectPreviewWrap.scrollLeft = 0;
        inspectPreviewWrap.scrollTop = 0;
      }
    });
  }
  if (zoomRange) {
    zoomRange.addEventListener("input", function () {
      var nextPercent = Number(zoomRange.value) || 100;
      setInspectZoom(nextPercent / 100, true);
    });
  }
  if (inspectPreviewWrap) {
    inspectPreviewWrap.addEventListener(
      "wheel",
      function (event) {
        if (!isInspectOpen()) {
          return;
        }
        if (!event.ctrlKey && !event.metaKey) {
          return;
        }
        event.preventDefault();
        if (event.deltaY < 0) {
          setInspectZoom(inspectZoom + ZOOM_STEP, true);
        } else if (event.deltaY > 0) {
          setInspectZoom(inspectZoom - ZOOM_STEP, true);
        }
      },
      { passive: false }
    );
  }

  function isTypingTarget(target) {
    var node = target || doc.activeElement;
    var tag = node ? String(node.tagName || "").toLowerCase() : "";
    if (tag === "textarea" || tag === "select") {
      return true;
    }
    if (tag !== "input") {
      return false;
    }
    var inputType = String(node.type || "").toLowerCase();
    return inputType !== "range";
  }

  function handleInspectHotkeys(event) {
    if (!isInspectOpen()) {
      return false;
    }
    if (event.key === "Escape") {
      event.preventDefault();
      closeInspectModal();
      return true;
    }
    if (isTypingTarget(event.target)) {
      return true;
    }
    if (event.key === "+" || event.key === "=") {
      event.preventDefault();
      setInspectZoom(inspectZoom + ZOOM_STEP, true);
      return true;
    }
    if (event.key === "-" || event.key === "_") {
      event.preventDefault();
      setInspectZoom(inspectZoom - ZOOM_STEP, true);
      return true;
    }
    if (event.key === "0") {
      event.preventDefault();
      setInspectZoom(1, false);
      if (inspectPreviewWrap) {
        inspectPreviewWrap.scrollLeft = 0;
        inspectPreviewWrap.scrollTop = 0;
      }
      return true;
    }
    if (event.key === "f" || event.key === "F") {
      event.preventDefault();
      fitInspectZoom();
      return true;
    }
    return true;
  }

  var grid = doc.getElementById("images-grid");
  if (!grid) {
    doc.addEventListener("keydown", function (event) {
      if (handleInspectHotkeys(event)) {
        return;
      }
      if ((event.key === "u" || event.key === "U") && !event.ctrlKey && !event.metaKey && !event.altKey) {
        var focused = doc.activeElement;
        var activeTag = focused ? String(focused.tagName || "").toLowerCase() : "";
        if (activeTag !== "input" && activeTag !== "textarea" && activeTag !== "select") {
          event.preventDefault();
          openUploadPanel();
        }
      }
    });
    return;
  }
  var cards = asArray(grid.querySelectorAll(".image-lib-card"));
  var searchInput = doc.getElementById("images-search");
  var extFilter = doc.getElementById("images-ext-filter");
  var sortSelect = doc.getElementById("images-sort");
  var clearFiltersButton = doc.getElementById("images-clear-filters");
  var visiblePill = doc.getElementById("images-visible-pill");
  var loadMoreButton = doc.getElementById("images-load-more-btn");
  var loadMoreNote = doc.getElementById("images-load-more-note");
  var loadSentinel = doc.getElementById("images-load-sentinel");
  var chunkSize = 60;
  var visibleLimit = chunkSize;

  asArray(doc.querySelectorAll("[data-delete-image-form]")).forEach(function (form) {
    form.addEventListener("submit", function (event) {
      var target = event.currentTarget;
      if (!target) {
        return;
      }
      var hiddenInput = target.querySelector("input[name='image_name']");
      var imageName = hiddenInput ? String(hiddenInput.value || "").trim() : "";
      var label = imageName || "this image";
      var message =
        "Delete " + label + " from this workspace?\n\n" +
        "This removes only the workspace copy. Existing draft datasets keep their copied files.\n" +
        "Deletion is blocked when active analysis jobs/runs still reference the image.";
      if (!window.confirm(message)) {
        event.preventDefault();
      }
    });
  });

  function wireInspectTriggers(card) {
    asArray(card.querySelectorAll("[data-inspect-image]")).forEach(function (trigger) {
      trigger.addEventListener("click", function (event) {
        event.preventDefault();
        openInspectModal(card);
      });
    });
  }

  function populateExtFilter() {
    if (!extFilter) {
      return;
    }
    var extMap = {};
    cards.forEach(function (card) {
      var ext = lower(card.getAttribute("data-extension"));
      if (ext) {
        extMap[ext] = true;
      }
    });
    Object.keys(extMap).sort().forEach(function (ext) {
      var option = doc.createElement("option");
      option.value = ext;
      option.textContent = "." + ext;
      extFilter.appendChild(option);
    });
  }

  function filteredCards() {
    var searchText = lower(searchInput ? searchInput.value : "");
    var extText = lower(extFilter ? extFilter.value : "");
    return cards.filter(function (card) {
      var name = lower(card.getAttribute("data-name"));
      var ext = lower(card.getAttribute("data-extension"));
      var matchesSearch = !searchText || name.indexOf(searchText) >= 0;
      var matchesExt = !extText || ext === extText;
      return matchesSearch && matchesExt;
    });
  }

  function refreshCounts(visibleCount, totalFiltered) {
    if (visiblePill) {
      visiblePill.textContent = String(visibleCount) + " / " + String(cards.length) + " shown";
    }
    if (loadMoreNote) {
      loadMoreNote.textContent = totalFiltered > visibleCount
        ? String(totalFiltered - visibleCount) + " remaining"
        : "All filtered images shown";
    }
  }

  function applyGrid() {
    var list = filteredCards();
    sortCards(list, sortSelect ? sortSelect.value : "modified_desc");
    list.forEach(function (card) {
      grid.appendChild(card);
    });
    var visibleCount = 0;
    cards.forEach(function (card) {
      card.hidden = true;
    });
    list.forEach(function (card, index) {
      var show = index < visibleLimit;
      card.hidden = !show;
      if (show) {
        visibleCount += 1;
      }
    });
    if (loadMoreButton) {
      loadMoreButton.hidden = list.length <= visibleLimit;
      loadMoreButton.disabled = list.length <= visibleLimit;
    }
    refreshCounts(visibleCount, list.length);
  }

  if (searchInput) {
    searchInput.addEventListener("input", function () {
      visibleLimit = chunkSize;
      applyGrid();
    });
  }
  if (extFilter) {
    extFilter.addEventListener("change", function () {
      visibleLimit = chunkSize;
      applyGrid();
    });
  }
  if (sortSelect) {
    sortSelect.addEventListener("change", function () {
      applyGrid();
    });
  }
  if (clearFiltersButton) {
    clearFiltersButton.addEventListener("click", function () {
      if (searchInput) {
        searchInput.value = "";
      }
      if (extFilter) {
        extFilter.value = "";
      }
      if (sortSelect) {
        sortSelect.value = "modified_desc";
      }
      visibleLimit = chunkSize;
      applyGrid();
    });
  }
  if (loadMoreButton) {
    loadMoreButton.addEventListener("click", function () {
      visibleLimit += chunkSize;
      applyGrid();
    });
  }

  if (loadSentinel && window.IntersectionObserver) {
    var observer = new window.IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting && loadMoreButton && !loadMoreButton.hidden) {
          visibleLimit += chunkSize;
          applyGrid();
        }
      });
    }, { rootMargin: "140px 0px 140px 0px" });
    observer.observe(loadSentinel);
  }

  cards.forEach(function (card) {
    var img = card.querySelector(".image-lib-thumb");
    var skeleton = card.querySelector(".image-thumb-skeleton");
    var errorText = card.querySelector(".image-thumb-error");
    var metaLine = card.querySelector("[data-image-meta]");
    var sizeSpan = card.querySelector(".image-meta-secondary span");
    if (sizeSpan) {
      sizeSpan.textContent = formatBytes(card.getAttribute("data-size"));
    }
    function onLoaded() {
      card.classList.add("is-thumb-ready");
      card.classList.remove("is-thumb-error");
      if (skeleton) {
        skeleton.hidden = true;
      }
      if (errorText) {
        errorText.hidden = true;
      }
      if (metaLine && img) {
        metaLine.textContent = String(img.naturalWidth) + " x " + String(img.naturalHeight);
      }
      if (img) {
        card.setAttribute("data-width", String(img.naturalWidth));
        card.setAttribute("data-height", String(img.naturalHeight));
      }
    }
    function onError() {
      card.classList.remove("is-thumb-ready");
      card.classList.add("is-thumb-error");
      if (skeleton) {
        skeleton.hidden = true;
      }
      if (errorText) {
        errorText.hidden = false;
      }
      if (metaLine) {
        metaLine.textContent = "Preview failed to load";
      }
    }
    wireInspectTriggers(card);
    if (!img) {
      return;
    }
    img.addEventListener("load", onLoaded);
    img.addEventListener("error", onError);
    if (img.complete && img.naturalWidth > 0) {
      onLoaded();
    } else if (img.complete) {
      onError();
    }
  });

  populateExtFilter();
  applyGrid();

  doc.addEventListener("keydown", function (event) {
    if (handleInspectHotkeys(event)) {
      return;
    }
    if (event.key === "/" && searchInput) {
      var active = doc.activeElement;
      var tag = active ? String(active.tagName || "").toLowerCase() : "";
      if (tag !== "input" && tag !== "textarea" && tag !== "select") {
        event.preventDefault();
        searchInput.focus();
        searchInput.select();
      }
      return;
    }
    if ((event.key === "u" || event.key === "U") && !event.ctrlKey && !event.metaKey && !event.altKey) {
      var focused = doc.activeElement;
      var activeTag = focused ? String(focused.tagName || "").toLowerCase() : "";
      if (activeTag !== "input" && activeTag !== "textarea" && activeTag !== "select") {
        event.preventDefault();
        openUploadPanel();
      }
    }
  });
})();
