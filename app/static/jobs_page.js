(function () {
  var doc = document;
  var table = doc.getElementById("jobs-table");
  if (!table) {
    return;
  }

  var tbody = table.querySelector("tbody");
  var pollUrl = String(table.getAttribute("data-jobs-poll-url") || "");
  var reorderUrl = String(table.getAttribute("data-jobs-reorder-url") || "");
  var knownStatuses = {};
  var hasInitializedPoll = false;
  var dragState = {
    draggedRow: null
  };

  function statusClass(status) {
    var key = String(status || "").toLowerCase();
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

  function stageLabel(stage) {
    var text = String(stage || "").trim();
    if (!text) {
      return "-";
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

  function rowByJobId(jobId) {
    return tbody ? tbody.querySelector('tr[data-job-id="' + String(jobId) + '"]') : null;
  }

  function setRowDraggable(row, isQueued) {
    if (!row) {
      return;
    }
    if (isQueued) {
      row.setAttribute("draggable", "true");
      row.classList.add("is-queue-row");
    } else {
      row.removeAttribute("draggable");
      row.classList.remove("is-queue-row");
    }
  }

  function updateRow(job) {
    var row = rowByJobId(job.id);
    if (!row) {
      return;
    }
    var statusText = String(job.status || "");
    var statusKey = statusText.toLowerCase();
    row.setAttribute("data-status", statusKey);
    row.setAttribute("data-queue-position", String(job.queue_position || ""));
    setRowDraggable(row, statusKey === "queued");

    var queueCell = row.querySelector(".job-queue-col");
    if (queueCell) {
      if (statusKey === "queued" && job.queue_position) {
        queueCell.innerHTML = '<span class="queue-chip">#' + String(job.queue_position) + "</span>";
      } else {
        queueCell.innerHTML = '<span class="muted">-</span>';
      }
    }

    var statusPill = row.querySelector(".status-pill");
    if (statusPill) {
      statusPill.classList.remove("is-good", "is-progress", "is-bad", "is-neutral");
      statusPill.classList.add(statusClass(statusKey));
      statusPill.textContent = statusText;
    }

    var stageCell = row.querySelector(".job-stage");
    if (stageCell) {
      stageCell.textContent = stageLabel(job.stage);
    }

    var progressWrap = row.querySelector(".job-progress");
    if (progressWrap) {
      var progressEl = progressWrap.querySelector("progress");
      var percentEl = progressWrap.querySelector("span");
      var counterEl = progressWrap.querySelector("[data-progress-counter]");
      var percentValue = Number(job.progress_percent);
      if (!Number.isFinite(percentValue)) {
        percentValue = 0;
      }
      if (progressEl) {
        progressEl.value = percentValue;
      }
      if (percentEl) {
        percentEl.textContent = String(Math.round(percentValue)) + "%";
      }
      if (counterEl) {
        counterEl.textContent = String(job.progress_counter_label || "");
      }
    }

    var lastLogCell = row.querySelector(".job-last-log");
    if (lastLogCell && job.last_log) {
      lastLogCell.textContent = String(job.last_log);
    }
  }

  function queuedRowIdsInDomOrder() {
    if (!tbody) {
      return [];
    }
    var rows = Array.prototype.slice.call(tbody.querySelectorAll("tr[data-job-id]"));
    return rows
      .filter(function (row) {
        return String(row.getAttribute("data-status") || "").toLowerCase() === "queued";
      })
      .map(function (row) {
        return Number(row.getAttribute("data-job-id"));
      })
      .filter(function (id) {
        return Number.isInteger(id) && id > 0;
      });
  }

  function persistQueueOrder() {
    if (!reorderUrl) {
      return Promise.resolve();
    }
    var orderedIds = queuedRowIdsInDomOrder();
    return fetch(reorderUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Requested-With": "XMLHttpRequest"
      },
      body: JSON.stringify({ ordered_job_ids: orderedIds })
    })
      .then(function (response) {
        if (!response.ok) {
          throw new Error("Reorder request failed.");
        }
        return response.json();
      })
      .then(function () {
        showToast("Queue order updated.", "success");
      })
      .catch(function () {
        showToast("Failed to save queue order.", "error");
      });
  }

  function reorderQueuedRowsByPosition() {
    if (!tbody) {
      return;
    }
    var allRows = Array.prototype.slice.call(tbody.querySelectorAll("tr[data-job-id]"));
    if (!allRows.length) {
      return;
    }
    var queued = allRows.filter(function (row) {
      return String(row.getAttribute("data-status") || "").toLowerCase() === "queued";
    });
    var others = allRows.filter(function (row) {
      return String(row.getAttribute("data-status") || "").toLowerCase() !== "queued";
    });
    queued.sort(function (a, b) {
      var aPos = Number(a.getAttribute("data-queue-position") || "999999");
      var bPos = Number(b.getAttribute("data-queue-position") || "999999");
      if (aPos < bPos) {
        return -1;
      }
      if (aPos > bPos) {
        return 1;
      }
      return 0;
    });
    queued.concat(others).forEach(function (row) {
      tbody.appendChild(row);
    });
  }

  function bindDragAndDrop() {
    if (!tbody) {
      return;
    }
    tbody.addEventListener("dragstart", function (event) {
      var row = event.target && event.target.closest ? event.target.closest("tr[data-job-id]") : null;
      if (!row) {
        return;
      }
      if (String(row.getAttribute("data-status") || "").toLowerCase() !== "queued") {
        event.preventDefault();
        return;
      }
      dragState.draggedRow = row;
      row.classList.add("is-dragging");
      if (event.dataTransfer) {
        event.dataTransfer.effectAllowed = "move";
        event.dataTransfer.setData("text/plain", row.getAttribute("data-job-id") || "");
      }
    });

    tbody.addEventListener("dragend", function () {
      var dragging = tbody.querySelector("tr.is-dragging");
      if (dragging) {
        dragging.classList.remove("is-dragging");
      }
      dragState.draggedRow = null;
      tbody.querySelectorAll("tr.is-drop-target").forEach(function (row) {
        row.classList.remove("is-drop-target");
      });
    });

    tbody.addEventListener("dragover", function (event) {
      var targetRow = event.target && event.target.closest ? event.target.closest("tr[data-job-id]") : null;
      if (!dragState.draggedRow || !targetRow || targetRow === dragState.draggedRow) {
        return;
      }
      if (String(targetRow.getAttribute("data-status") || "").toLowerCase() !== "queued") {
        return;
      }
      event.preventDefault();
      tbody.querySelectorAll("tr.is-drop-target").forEach(function (row) {
        row.classList.remove("is-drop-target");
      });
      targetRow.classList.add("is-drop-target");
      if (event.dataTransfer) {
        event.dataTransfer.dropEffect = "move";
      }
    });

    tbody.addEventListener("drop", function (event) {
      var targetRow = event.target && event.target.closest ? event.target.closest("tr[data-job-id]") : null;
      if (!dragState.draggedRow || !targetRow || targetRow === dragState.draggedRow) {
        return;
      }
      if (String(targetRow.getAttribute("data-status") || "").toLowerCase() !== "queued") {
        return;
      }
      event.preventDefault();
      targetRow.classList.remove("is-drop-target");
      var draggedId = Number(dragState.draggedRow.getAttribute("data-job-id"));
      var targetId = Number(targetRow.getAttribute("data-job-id"));
      if (!Number.isInteger(draggedId) || !Number.isInteger(targetId)) {
        return;
      }
      var rows = Array.prototype.slice.call(tbody.querySelectorAll("tr[data-job-id]"));
      var draggedIndex = rows.indexOf(dragState.draggedRow);
      var targetIndex = rows.indexOf(targetRow);
      if (draggedIndex < 0 || targetIndex < 0) {
        return;
      }
      if (draggedIndex < targetIndex) {
        tbody.insertBefore(dragState.draggedRow, targetRow.nextSibling);
      } else {
        tbody.insertBefore(dragState.draggedRow, targetRow);
      }
      persistQueueOrder();
    });
  }

  function pollJobs() {
    if (!pollUrl) {
      return;
    }
    var fetchUrl = pollUrl + (pollUrl.indexOf("?") >= 0 ? "&" : "?") + "limit=240";
    fetch(fetchUrl, {
      headers: {
        "X-Requested-With": "XMLHttpRequest"
      }
    })
      .then(function (response) {
        if (!response.ok) {
          throw new Error("Polling failed.");
        }
        return response.json();
      })
      .then(function (payload) {
        var jobs = Array.isArray(payload.jobs) ? payload.jobs : [];
        var nextStatuses = {};
        jobs.forEach(function (job) {
          nextStatuses[job.id] = String(job.status || "");
          updateRow(job);
          if (!hasInitializedPoll) {
            return;
          }
          var previousStatus = String(knownStatuses[job.id] || "");
          if (!previousStatus || previousStatus === String(job.status || "")) {
            return;
          }
          var newStatus = String(job.status || "").toLowerCase();
          if (newStatus === "running") {
            showToast("Job #" + String(job.id) + " started.", "success");
          } else if (newStatus === "completed") {
            showToast("Job #" + String(job.id) + " completed.", "success");
          } else if (newStatus === "failed") {
            showToast("Job #" + String(job.id) + " failed.", "error");
          } else if (newStatus === "canceled") {
            showToast("Job #" + String(job.id) + " canceled.", "warning");
          }
        });
        knownStatuses = nextStatuses;
        hasInitializedPoll = true;
        reorderQueuedRowsByPosition();
      })
      .catch(function () {
        // Keep polling even when one request fails.
      });
  }

  bindDragAndDrop();
  pollJobs();
  window.setInterval(function () {
    if (doc.hidden) {
      return;
    }
    pollJobs();
  }, 3000);
})();
