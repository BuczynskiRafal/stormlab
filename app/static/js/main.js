document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(function (tooltipTriggerEl) {
        new bootstrap.Tooltip(tooltipTriggerEl, {
            animation: true,
            delay: { "show": 800, "hide": 100 },
            placement: 'auto',
            html: false
        });
    });
});

const initBackToTop = function() {
    const button = document.getElementById('backToTop');
    if (!button) {
        return;
    }

    const SHOW_AFTER_PX = 600;
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
    let pending = false;

    const syncVisibility = function() {
        pending = false;
        button.classList.toggle('is-visible', window.scrollY > SHOW_AFTER_PX);
    };

    // The scroll event fires far more often than the class can meaningfully change,
    // so the read of scrollY is deferred to one animation frame per burst.
    window.addEventListener('scroll', function() {
        if (pending) {
            return;
        }
        pending = true;
        window.requestAnimationFrame(syncVisibility);
    }, { passive: true });

    button.addEventListener('click', function() {
        window.scrollTo({ top: 0, behavior: reducedMotion.matches ? 'auto' : 'smooth' });
    });

    syncVisibility();
}

/**
 * File controls built as .file-input-wrapper[data-file-input]: a visually hidden
 * <input type="file"> and a <label> drop zone. The label opens the picker by itself;
 * this adds the chosen file's name and size, and drag and drop.
 */
const formatBytes = function(bytes) {
    if (bytes >= 1024 * 1024) {
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }
    return Math.max(1, Math.round(bytes / 1024)) + ' kB';
};

const initFileInputs = function() {
    document.querySelectorAll('[data-file-input]').forEach(function(wrapper) {
        const input = wrapper.querySelector('input[type="file"]');
        const display = wrapper.querySelector('.file-input-display');
        const name = wrapper.querySelector('[data-file-name]');
        const detail = wrapper.querySelector('[data-file-detail]');
        if (!input || !display) return;

        const sync = function() {
            const file = input.files && input.files[0];
            display.classList.toggle('file-input-selected', Boolean(file));
            if (file) {
                wrapper.classList.remove('has-error');
            }
            if (name) name.textContent = file ? file.name : name.dataset.emptyText;
            if (detail) detail.textContent = file ? formatBytes(file.size) : detail.dataset.emptyText;
            wrapper.dispatchEvent(new CustomEvent('file-input:change', { bubbles: true, detail: { file: file || null } }));
        };

        input.addEventListener('change', sync);

        display.addEventListener('dragover', function(event) {
            event.preventDefault();
            display.classList.add('drag-over');
        });
        display.addEventListener('dragleave', function() {
            display.classList.remove('drag-over');
        });
        display.addEventListener('drop', function(event) {
            event.preventDefault();
            display.classList.remove('drag-over');
            if (event.dataTransfer && event.dataTransfer.files.length > 0) {
                input.files = event.dataTransfer.files;
                sync();
            }
        });

        // A page restored from the back/forward cache keeps the chosen file.
        sync();
    });
};

/**
 * The analysis form (#uploadForm).
 * - An empty "Upload & Analyse" shows an inline error under the drop zone instead of
 *   the browser bubble (the demo button has formnovalidate and is not checked).
 * - While SWMM runs (up to a minute) both buttons are disabled, a spinner and a status
 *   line are shown and the form is aria-busy; a second submit is ignored.
 * - Coming back with the Back button restores the form from the bfcache, where it
 *   would otherwise stay disabled.
 */
const initAnalysisForm = function() {
    const form = document.getElementById('uploadForm');
    if (!form) return;
    const input = form.querySelector('input[type="file"]');
    const wrapper = form.querySelector('[data-file-input]');
    const error = form.querySelector('[data-file-error]');
    const status = document.getElementById('uploadStatus');
    const buttons = Array.from(form.querySelectorAll('button[type="submit"]'));
    const EMPTY_MESSAGE = 'Choose a .inp file to analyse, or try the demo data.';
    let busy = false;

    const showError = function(text) {
        if (!error) return;
        error.textContent = text;
        error.classList.remove('d-none');
        if (wrapper) wrapper.classList.add('has-error');
    };

    const clearError = function() {
        if (!error) return;
        error.textContent = '';
        error.classList.add('d-none');
        if (wrapper) wrapper.classList.remove('has-error');
    };

    const setBusy = function(on) {
        busy = on;
        form.setAttribute('aria-busy', on ? 'true' : 'false');
        buttons.forEach(function(button) { button.disabled = on; });
        if (status) status.classList.toggle('d-none', !on);
    };

    if (input) {
        input.addEventListener('invalid', function(event) {
            event.preventDefault();
            showError(EMPTY_MESSAGE);
            input.focus();
        });
    }
    if (wrapper) {
        wrapper.addEventListener('file-input:change', function(event) {
            if (event.detail.file) clearError();
        });
    }

    form.addEventListener('submit', function(event) {
        if (busy) {
            event.preventDefault();
            return;
        }
        // Disabled buttons are left out of the form data, so the clicked one (the demo
        // button carries name="use_test_file") is disabled only after it was sent.
        busy = true;
        window.setTimeout(function() { setBusy(true); }, 0);
    });

    window.addEventListener('pageshow', function(event) {
        if (event.persisted) setBusy(false);
    });
};

/**
 * The account page names the section whose form was rejected; move focus to its first
 * invalid field so the error is the next thing a keyboard or screen reader user meets.
 */
const focusRejectedSection = function() {
    const page = document.querySelector('[data-error-section]');
    if (!page) return;
    const section = document.getElementById(page.dataset.errorSection);
    if (!section) return;
    const field = section.querySelector('[aria-invalid="true"]') || section.querySelector('input:not([type="hidden"])');
    section.scrollIntoView({ block: 'start' });
    if (field) field.focus({ preventScroll: true });
};

document.addEventListener('DOMContentLoaded', function() {
    initFileInputs();
    initAnalysisForm();
    focusRejectedSection();

    // =========================================================================
    // Keyboard Navigation for Detail Pages
    // =========================================================================

    /**
     * Left/right arrow keys move to the previous/next conduit on the conduit panel.
     * The targets are the rel="prev"/rel="next" links of the pager in the profile card
     * head; a missing neighbour is a span there, so the key does nothing at either end.
     * Widgets that own the arrow keys (the profile scroller, the tablist in
     * conduit-panel.js) stop the event before it reaches the document.
     */
    function initKeyboardNavigation() {
        const pager = document.querySelector('.conduit-pager');
        if (!pager) return;

        document.addEventListener('keydown', (event) => {
            if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
            // Alt+Left is the browser's Back, Shift/Ctrl/Cmd+arrows select or jump text.
            if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
            // Ignore if user is typing in an input field
            const target = event.target;
            const tagName = target.tagName.toLowerCase();
            if (tagName === 'input' || tagName === 'textarea' || tagName === 'select' || target.isContentEditable) {
                return;
            }

            const rel = event.key === 'ArrowLeft' ? 'prev' : 'next';
            const link = pager.querySelector(`a[rel="${rel}"]`);
            if (!link) return;
            event.preventDefault();
            link.click();
        });
    }

    initKeyboardNavigation();

    // =========================================================================
    // History Page: per-page form, single delete, bulk selection
    // =========================================================================

    function initHistoryFilters() {
        const form = document.getElementById('historyFilters');
        if (!form) return;

        // Generic over the form's selects: a control added to the form navigates
        // without any change here.
        form.querySelectorAll('select').forEach((select) => {
            select.addEventListener('change', function() {
                // requestSubmit() runs validation and fires `submit`; Safari < 16 and
                // older WebKit lack it, where submit() still navigates (this form is a
                // plain GET with no required fields, so skipping validation is safe).
                if (typeof form.requestSubmit === 'function') {
                    form.requestSubmit();
                } else {
                    form.submit();
                }
            });
        });
    }

    /**
     * Where to go after rows were deleted: the current page if it still exists,
     * otherwise the last page that does. Reloading a page that no longer exists (the
     * last rows of the last page) answered 404.
     */
    function reloadAfterDelete(deletedCount) {
        const table = document.getElementById('sessionsTable');
        const url = new URL(window.location.href);
        if (table) {
            const page = parseInt(table.dataset.page, 10) || 1;
            const perPage = parseInt(table.dataset.perPage, 10) || 10;
            const total = parseInt(table.dataset.total, 10) || 0;
            const lastPage = Math.max(1, Math.ceil(Math.max(0, total - deletedCount) / perPage));
            if (page > lastPage) {
                if (lastPage === 1) {
                    url.searchParams.delete('page');
                } else {
                    url.searchParams.set('page', String(lastPage));
                }
                window.location.assign(url.toString());
                return;
            }
        }
        window.location.reload();
    }

    function sessionCount(count) {
        return count + (count === 1 ? ' session' : ' sessions');
    }

    function showModalError(element, text) {
        if (!element) return;
        element.textContent = text;
        element.classList.remove('d-none');
    }

    function hideModalError(element) {
        if (!element) return;
        element.textContent = '';
        element.classList.add('d-none');
    }

    function postJson(url, body) {
        const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]')?.value;
        return fetch(url, {
            method: 'POST',
            headers: { 'X-CSRFToken': csrfToken, 'Content-Type': 'application/json' },
            body: body === undefined ? undefined : JSON.stringify(body),
        }).then(function(response) {
            return response.json().catch(function() { return { success: false }; });
        });
    }

    // "View Error" opens the full message in the row below (click, tap, Enter or Space).
    function initErrorToggles() {
        document.querySelectorAll('.history-error-toggle').forEach(function(button) {
            const row = document.getElementById(button.getAttribute('aria-controls'));
            if (!row) return;
            button.addEventListener('click', function() {
                const open = button.getAttribute('aria-expanded') !== 'true';
                button.setAttribute('aria-expanded', open ? 'true' : 'false');
                button.textContent = open ? 'Hide Error' : 'View Error';
                row.hidden = !open;
            });
        });
    }

    function initSingleDelete() {
        const deleteModal = document.getElementById('deleteModal');
        const nameElement = document.getElementById('sessionToDelete');
        const dateElement = document.getElementById('sessionToDeleteDate');
        const confirmDeleteButton = document.getElementById('confirmDelete');
        const errorElement = document.getElementById('deleteError');

        if (!deleteModal || !nameElement || !confirmDeleteButton) return;

        const cancelButton = deleteModal.querySelector('.modal-footer [data-bs-dismiss="modal"]');
        const deleteText = confirmDeleteButton.querySelector('.delete-text');
        const deleteLoading = confirmDeleteButton.querySelector('.delete-loading');

        if (!cancelButton || !deleteText || !deleteLoading) return;

        let sessionIdToDelete = null;

        const setLoading = function(on) {
            deleteText.classList.toggle('d-none', on);
            deleteLoading.classList.toggle('d-none', !on);
            confirmDeleteButton.disabled = on;
            cancelButton.disabled = on;
        };

        // Bootstrap opens the modal declaratively (data-bs-target); this only fills it
        // from the row whose bin was clicked: the file and the date the row shows.
        document.addEventListener('click', function(e) {
            const button = e.target.closest('[data-bs-target="#deleteModal"]');
            if (!button) return;

            sessionIdToDelete = button.getAttribute('data-session-id');
            nameElement.textContent = button.getAttribute('data-session-name');
            if (dateElement) dateElement.textContent = button.getAttribute('data-session-date');
            hideModalError(errorElement);
            setLoading(false);
        });

        confirmDeleteButton.addEventListener('click', function() {
            if (!sessionIdToDelete) return;
            hideModalError(errorElement);
            setLoading(true);

            postJson(`/session/${sessionIdToDelete}/delete/`)
                .then(function(data) {
                    if (data.success) {
                        reloadAfterDelete(1);
                        return;
                    }
                    setLoading(false);
                    showModalError(errorElement, data.message || 'The session could not be deleted. Try again.');
                })
                .catch(function() {
                    setLoading(false);
                    showModalError(errorElement, 'The session could not be deleted. Check the connection and try again.');
                });
        });
    }

    function initBulkSelection() {
        const bulkActionBar = document.getElementById('bulkActionBar');
        const selectedCountElement = document.getElementById('selectedCount');
        const selectAllCheckbox = document.getElementById('selectAllCheckbox');
        const selectAllVisibleBtn = document.getElementById('selectAllVisible');
        const clearSelectionBtn = document.getElementById('clearSelection');
        const bulkDeleteBtn = document.getElementById('bulkDeleteBtn');
        const bulkDeleteModalElement = document.getElementById('bulkDeleteModal');
        const confirmBulkDeleteBtn = document.getElementById('confirmBulkDelete');
        const bulkDeleteCountElement = document.getElementById('bulkDeleteCount');
        const bulkErrorElement = document.getElementById('bulkDeleteError');

        if (!bulkActionBar || !selectAllCheckbox || !bulkDeleteModalElement || !confirmBulkDeleteBtn) return;

        const bulkDeleteModal = new bootstrap.Modal(bulkDeleteModalElement);
        const bulkDeleteText = confirmBulkDeleteBtn.querySelector('.bulk-delete-text');
        const bulkDeleteLoading = confirmBulkDeleteBtn.querySelector('.bulk-delete-loading');
        const cancelButton = bulkDeleteModalElement.querySelector('.modal-footer [data-bs-dismiss="modal"]');
        const selectedSessions = new Set();

        const rows = function() {
            return Array.from(document.querySelectorAll('#sessionsTable tbody tr[data-session-id]'));
        };

        const setLoading = function(on) {
            bulkDeleteText.classList.toggle('d-none', on);
            bulkDeleteLoading.classList.toggle('d-none', !on);
            confirmBulkDeleteBtn.disabled = on;
            if (cancelButton) cancelButton.disabled = on;
        };

        function updateUI() {
            const count = selectedSessions.size;
            selectedCountElement.textContent = count;
            bulkDeleteCountElement.textContent = sessionCount(count);
            bulkActionBar.style.display = count > 0 ? 'block' : 'none';

            const visibleRows = rows();
            const checked = visibleRows.filter(function(row) {
                const checkbox = row.querySelector('.row-checkbox');
                return checkbox && checkbox.checked;
            });
            selectAllCheckbox.checked = visibleRows.length > 0 && checked.length === visibleRows.length;
            selectAllCheckbox.indeterminate = checked.length > 0 && checked.length < visibleRows.length;

            visibleRows.forEach(function(row) {
                row.classList.toggle('selected', selectedSessions.has(row.dataset.sessionId));
            });
        }

        function setAll(checked) {
            rows().forEach(function(row) {
                const checkbox = row.querySelector('.row-checkbox');
                if (!checkbox) return;
                checkbox.checked = checked;
                if (checked) {
                    selectedSessions.add(row.dataset.sessionId);
                } else {
                    selectedSessions.delete(row.dataset.sessionId);
                }
            });
            updateUI();
        }

        document.addEventListener('change', function(e) {
            if (!e.target.classList.contains('row-checkbox')) return;
            const sessionId = e.target.dataset.sessionId;
            if (e.target.checked) {
                selectedSessions.add(sessionId);
            } else {
                selectedSessions.delete(sessionId);
            }
            updateUI();
        });

        selectAllCheckbox.addEventListener('change', function() { setAll(this.checked); });
        selectAllVisibleBtn.addEventListener('click', function() { setAll(true); });
        clearSelectionBtn.addEventListener('click', function() { setAll(false); });

        bulkDeleteBtn.addEventListener('click', function() {
            if (selectedSessions.size === 0) return;
            hideModalError(bulkErrorElement);
            bulkDeleteModal.show();
        });

        confirmBulkDeleteBtn.addEventListener('click', function() {
            if (selectedSessions.size === 0) return;
            hideModalError(bulkErrorElement);
            setLoading(true);

            const sessionIds = Array.from(selectedSessions).map(function(id) { return parseInt(id, 10); });
            postJson('/sessions/bulk-delete/', { session_ids: sessionIds })
                .then(function(data) {
                    if (data.success) {
                        reloadAfterDelete(data.deleted_count || sessionIds.length);
                        return;
                    }
                    setLoading(false);
                    showModalError(bulkErrorElement, data.message || 'The sessions could not be deleted. Try again.');
                })
                .catch(function() {
                    setLoading(false);
                    showModalError(bulkErrorElement, 'The sessions could not be deleted. Check the connection and try again.');
                });
        });

        bulkDeleteModalElement.addEventListener('hidden.bs.modal', function() {
            setLoading(false);
        });
    }

    initErrorToggles();

    initHistoryFilters();
    initSingleDelete();
    initBulkSelection();
    initBackToTop();

});
