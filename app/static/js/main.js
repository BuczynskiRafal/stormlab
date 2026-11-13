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

document.addEventListener('DOMContentLoaded', function() {
    const accordionHeaders = document.querySelectorAll('.accordion-header');
    const accordionItems = document.querySelectorAll('.accordion-item');

    // Pins the open section's header under the navigation bar while that section
    // scrolls past (the accordions of the profile page).
    function handleStickyHeaders() {
        if (accordionItems.length === 0) return;

        const header = document.querySelector('header');
        const headerHeight = header ? header.offsetHeight : 0;
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;

        // Get current position of navigation header
        const headerRect = header ? header.getBoundingClientRect() : null;
        const navBottomPosition = headerRect ? Math.max(0, headerRect.bottom) : 0;

        accordionItems.forEach((item) => {
            const accordionHeader = item.querySelector('.accordion-header');
            const collapse = item.querySelector('.accordion-collapse');
            // An accordion item without a header or a collapse pane is not ours to stick.
            if (!accordionHeader || !collapse) return;

            const itemRect = item.getBoundingClientRect();
            const itemTop = itemRect.top + scrollTop;
            const itemBottom = itemRect.bottom + scrollTop;

            const isExpanded = collapse.classList.contains('show');
            const shouldBeSticky = isExpanded &&
                                 scrollTop > (itemTop - headerHeight) &&
                                 scrollTop < (itemBottom - headerHeight - 100);

            if (shouldBeSticky) {
                accordionHeader.classList.add('sticky');

                // Position sticky header to follow navigation bar smoothly
                accordionHeader.style.top = navBottomPosition + 'px';

                // Add appropriate class based on nav position
                if (navBottomPosition > 0) {
                    accordionHeader.classList.add('sticky-with-nav');
                    accordionHeader.classList.remove('sticky-top');
                } else {
                    accordionHeader.classList.add('sticky-top');
                    accordionHeader.classList.remove('sticky-with-nav');
                }

                // Keep original position and width
                const itemCurrentRect = item.getBoundingClientRect();
                accordionHeader.style.left = itemCurrentRect.left + 'px';
                accordionHeader.style.width = item.offsetWidth + 'px';

                // Adjust accordion content padding
                collapse.style.paddingTop = (accordionHeader.offsetHeight || 60) + 'px';
            } else {
                accordionHeader.classList.remove('sticky', 'sticky-with-nav', 'sticky-top');
                accordionHeader.style.top = '';
                accordionHeader.style.left = '';
                accordionHeader.style.width = '';
                collapse.style.paddingTop = '';
            }
        });
    }

    // Throttle scroll events for better performance
    let ticking = false;
    function requestTick() {
        if (!ticking) {
            requestAnimationFrame(handleStickyHeaders);
            ticking = true;
            setTimeout(() => { ticking = false; }, 16);
        }
    }

    window.addEventListener('scroll', requestTick);
    window.addEventListener('resize', handleStickyHeaders);

    // Handle accordion toggle events
    accordionHeaders.forEach(header => {
        header.addEventListener('click', function() {
            // Small delay to allow accordion animation to start
            setTimeout(handleStickyHeaders, 50);
        });
    });

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

    function initSingleDelete() {
        const deleteModal = document.getElementById('deleteModal');
        const sessionToDeleteElement = document.getElementById('sessionToDelete');
        const confirmDeleteButton = document.getElementById('confirmDelete');
        const deleteDescription = document.getElementById('deleteDescription');
        const modalTitle = document.getElementById('deleteModalLabel');

        if (!deleteModal || !sessionToDeleteElement || !confirmDeleteButton || !deleteDescription || !modalTitle) return;

        const cancelButton = deleteModal.querySelector('[data-bs-dismiss="modal"]');
        const deleteText = confirmDeleteButton.querySelector('.delete-text');
        const deleteLoading = confirmDeleteButton.querySelector('.delete-loading');

        if (!cancelButton || !deleteText || !deleteLoading) return;

        let sessionIdToDelete = null;

        // Bootstrap opens the modal declaratively (data-bs-target); this only fills it
        // from the row whose trash button was clicked.
        document.addEventListener('click', function(e) {
            const button = e.target.closest('[data-bs-target="#deleteModal"]');
            if (!button) return;

            sessionIdToDelete = button.getAttribute('data-session-id');
            sessionToDeleteElement.textContent = button.getAttribute('data-session-name');

            const row = button.closest('tr');
            const statusBadge = row ? row.querySelector('.badge') : null;
            const isProcessing = statusBadge !== null && statusBadge.textContent.trim() === 'Processing';

            if (isProcessing) {
                modalTitle.textContent = 'Cancel Processing';
                deleteDescription.textContent = 'This will cancel the current processing and remove the session. This action cannot be undone.';
            } else {
                modalTitle.textContent = 'Confirm Deletion';
                deleteDescription.textContent = 'This action cannot be undone. All related data (conduits, nodes, subcatchments) will be permanently deleted.';
            }

            const buttonText = isProcessing ? 'Cancel Processing' : 'Delete Session';
            deleteText.innerHTML = `<i class="fas fa-trash me-1" aria-hidden="true"></i>${buttonText}`;
            deleteText.classList.remove('d-none');
            deleteLoading.classList.add('d-none');
            confirmDeleteButton.disabled = false;
            cancelButton.disabled = false;
            confirmDeleteButton.classList.remove('btn-success');
            confirmDeleteButton.classList.add('btn-outline-danger');
        });

        confirmDeleteButton.addEventListener('click', function() {
            if (!sessionIdToDelete) return;

            deleteText.classList.add('d-none');
            deleteLoading.classList.remove('d-none');
            confirmDeleteButton.disabled = true;
            cancelButton.disabled = true;

            const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]')?.value;

            fetch(`/session/${sessionIdToDelete}/delete/`, {
                method: 'POST',
                headers: {
                    'X-CSRFToken': csrfToken,
                    'Content-Type': 'application/json'
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    deleteLoading.classList.add('d-none');

                    // The title still says what the row was when the modal opened.
                    const successText = modalTitle.textContent === 'Cancel Processing' ? 'Cancelled' : 'Deleted';
                    deleteText.innerHTML = `<i class="fas fa-check me-1"></i>${successText}`;
                    deleteText.classList.remove('d-none');
                    confirmDeleteButton.classList.remove('btn-outline-danger');
                    confirmDeleteButton.classList.add('btn-success');

                    setTimeout(() => {
                        window.location.reload();
                    }, 800);
                } else {
                    deleteLoading.classList.add('d-none');
                    deleteText.classList.remove('d-none');
                    confirmDeleteButton.disabled = false;
                    cancelButton.disabled = false;
                    alert('Error: ' + data.message);
                }
            })
            .catch(error => {
                deleteLoading.classList.add('d-none');
                deleteText.classList.remove('d-none');
                confirmDeleteButton.disabled = false;
                cancelButton.disabled = false;
                console.error('Error:', error);
                alert('An error occurred while deleting the session.');
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

        // Only initialize if bulk selection elements exist
        if (!bulkActionBar || !selectAllCheckbox) return;

        const bulkDeleteModal = new bootstrap.Modal(bulkDeleteModalElement);
        let selectedSessions = new Set();

        function updateUI() {
            const count = selectedSessions.size;
            selectedCountElement.textContent = count;
            bulkDeleteCountElement.textContent = count;

            // Show/hide bulk action bar
            if (count > 0) {
                bulkActionBar.style.display = 'block';
            } else {
                bulkActionBar.style.display = 'none';
            }

            // Update select all checkbox state
            const visibleRows = document.querySelectorAll('#sessionsTable tbody tr[data-session-id]');
            const visibleChecked = Array.from(visibleRows).filter(row => {
                const checkbox = row.querySelector('.row-checkbox');
                return checkbox && checkbox.checked;
            });

            if (visibleRows.length > 0 && visibleChecked.length === visibleRows.length) {
                selectAllCheckbox.checked = true;
                selectAllCheckbox.indeterminate = false;
            } else if (visibleChecked.length > 0) {
                selectAllCheckbox.checked = false;
                selectAllCheckbox.indeterminate = true;
            } else {
                selectAllCheckbox.checked = false;
                selectAllCheckbox.indeterminate = false;
            }

            // Update row styling
            document.querySelectorAll('#sessionsTable tbody tr[data-session-id]').forEach(row => {
                const sessionId = row.dataset.sessionId;
                if (selectedSessions.has(sessionId)) {
                    row.classList.add('selected');
                } else {
                    row.classList.remove('selected');
                }
            });
        }

        // Handle individual checkbox changes
        document.addEventListener('change', function(e) {
            if (e.target.classList.contains('row-checkbox')) {
                const sessionId = e.target.dataset.sessionId;
                if (e.target.checked) {
                    selectedSessions.add(sessionId);
                } else {
                    selectedSessions.delete(sessionId);
                }
                updateUI();
            }
        });

        // Handle select all checkbox
        selectAllCheckbox.addEventListener('change', function() {
            const visibleRows = document.querySelectorAll('#sessionsTable tbody tr[data-session-id]');
            visibleRows.forEach(row => {
                const checkbox = row.querySelector('.row-checkbox');
                const sessionId = row.dataset.sessionId;
                if (checkbox) {
                    checkbox.checked = this.checked;
                    if (this.checked) {
                        selectedSessions.add(sessionId);
                    } else {
                        selectedSessions.delete(sessionId);
                    }
                }
            });
            updateUI();
        });

        // Handle "Select all visible" button
        selectAllVisibleBtn.addEventListener('click', function() {
            const visibleRows = document.querySelectorAll('#sessionsTable tbody tr[data-session-id]');
            visibleRows.forEach(row => {
                const checkbox = row.querySelector('.row-checkbox');
                const sessionId = row.dataset.sessionId;
                if (checkbox) {
                    checkbox.checked = true;
                    selectedSessions.add(sessionId);
                }
            });
            updateUI();
        });

        // Handle "Clear selection" button
        clearSelectionBtn.addEventListener('click', function() {
            document.querySelectorAll('.row-checkbox').forEach(checkbox => {
                checkbox.checked = false;
            });
            selectAllCheckbox.checked = false;
            selectedSessions.clear();
            updateUI();
        });

        // Handle bulk delete button
        bulkDeleteBtn.addEventListener('click', function() {
            if (selectedSessions.size > 0) {
                bulkDeleteModal.show();
            }
        });

        // Handle confirm bulk delete
        confirmBulkDeleteBtn.addEventListener('click', function() {
            if (selectedSessions.size === 0) return;

            const bulkDeleteText = confirmBulkDeleteBtn.querySelector('.bulk-delete-text');
            const bulkDeleteLoading = confirmBulkDeleteBtn.querySelector('.bulk-delete-loading');
            const cancelButton = document.querySelector('#bulkDeleteModal [data-bs-dismiss="modal"]');

            // Show loading state
            bulkDeleteText.classList.add('d-none');
            bulkDeleteLoading.classList.remove('d-none');
            confirmBulkDeleteBtn.disabled = true;
            cancelButton.disabled = true;

            const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]')?.value;
            const sessionIds = Array.from(selectedSessions).map(id => parseInt(id));

            fetch('/sessions/bulk-delete/', {
                method: 'POST',
                headers: {
                    'X-CSRFToken': csrfToken,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ session_ids: sessionIds })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Show success state briefly before reload
                    bulkDeleteLoading.classList.add('d-none');
                    bulkDeleteText.innerHTML = `<i class="fas fa-check me-1"></i>Deleted ${data.deleted_count} session(s)`;
                    bulkDeleteText.classList.remove('d-none');
                    confirmBulkDeleteBtn.classList.remove('btn-danger');
                    confirmBulkDeleteBtn.classList.add('btn-success');

                    setTimeout(() => {
                        window.location.reload();
                    }, 800);
                } else {
                    // Reset loading state on error
                    bulkDeleteLoading.classList.add('d-none');
                    bulkDeleteText.classList.remove('d-none');
                    confirmBulkDeleteBtn.disabled = false;
                    cancelButton.disabled = false;
                    alert('Error: ' + data.message);
                }
            })
            .catch(error => {
                // Reset loading state on error
                bulkDeleteLoading.classList.add('d-none');
                bulkDeleteText.classList.remove('d-none');
                confirmBulkDeleteBtn.disabled = false;
                cancelButton.disabled = false;
                console.error('Error:', error);
                alert('An error occurred while deleting sessions.');
            });
        });

        // Reset bulk delete modal state when hidden
        bulkDeleteModalElement.addEventListener('hidden.bs.modal', function() {
            const bulkDeleteText = confirmBulkDeleteBtn.querySelector('.bulk-delete-text');
            const bulkDeleteLoading = confirmBulkDeleteBtn.querySelector('.bulk-delete-loading');

            bulkDeleteText.innerHTML = '<i class="fas fa-trash me-1" aria-hidden="true"></i>Delete All Selected';
            bulkDeleteText.classList.remove('d-none');
            bulkDeleteLoading.classList.add('d-none');
            confirmBulkDeleteBtn.disabled = false;
            confirmBulkDeleteBtn.classList.remove('btn-success');
            if (!confirmBulkDeleteBtn.classList.contains('btn-danger')) {
                confirmBulkDeleteBtn.classList.add('btn-danger');
            }
        });
    }

    initHistoryFilters();
    initSingleDelete();
    initBulkSelection();
    initBackToTop();

});
