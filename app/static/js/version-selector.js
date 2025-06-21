/**
 * Version Selector Component
 * Allows users to select document versions for search and comparison
 */
class VersionSelector {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        this.options = {
            apiBasePath: '/api/v1',
            onVersionChange: null,
            onCompareRequest: null,
            ...options
        };
        
        this.documentId = null;
        this.versions = [];
        this.selectedVersion = 'latest';
        this.compareVersion = null;
        this.isCompareMode = false;
        
        this.init();
    }
    
    init() {
        // Create UI elements
        this.container.innerHTML = `
            <div class="version-selector">
                <div class="version-selector__main">
                    <label for="version-select">Document Version:</label>
                    <select id="version-select" class="form-select">
                        <option value="latest">Latest Version</option>
                        <option value="all">All Versions</option>
                    </select>
                    <button id="compare-toggle" class="btn btn-outline-primary">
                        <i class="fas fa-code-compare"></i> Compare
                    </button>
                </div>
                
                <div id="compare-container" class="version-selector__compare" style="display: none;">
                    <label for="compare-select">Compare with:</label>
                    <select id="compare-select" class="form-select"></select>
                    <button id="view-diff-btn" class="btn btn-primary">View Differences</button>
                </div>
                
                <div id="version-info" class="version-selector__info mt-2"></div>
            </div>
        `;
        
        // Get elements
        this.versionSelect = document.getElementById('version-select');
        this.compareToggle = document.getElementById('compare-toggle');
        this.compareContainer = document.getElementById('compare-container');
        this.compareSelect = document.getElementById('compare-select');
        this.viewDiffBtn = document.getElementById('view-diff-btn');
        this.versionInfo = document.getElementById('version-info');

        // Attach event listeners
        this.attachEventListeners();
    }

    attachEventListeners() {
        // Version selection change
        this.versionSelect.addEventListener('change', (e) => {
            this.selectedVersion = e.target.value;
            this.updateVersionInfo();
            this.notifyVersionChange();
        });

        // Compare toggle
        this.compareToggle.addEventListener('click', () => {
            this.toggleCompareMode();
        });

        // Compare version selection
        this.compareSelect.addEventListener('change', (e) => {
            this.compareVersion = e.target.value;
            this.updateCompareButton();
        });

        // View diff button
        this.viewDiffBtn.addEventListener('click', () => {
            this.viewDifferences();
        });
    }

    async loadDocumentVersions(documentId) {
        this.documentId = documentId;

        try {
            // First, determine if this is a family ID or document ID
            const response = await fetch(`${this.options.apiBasePath}/documents/family/${documentId}/versions`);

            if (!response.ok) {
                // Try as a single document
                const docResponse = await fetch(`${this.options.apiBasePath}/documents/${documentId}`);
                if (docResponse.ok) {
                    const docData = await docResponse.json();
                    this.versions = [{
                        id: docData.id,
                        filename: docData.filename,
                        version_number: 1,
                        is_latest_version: true,
                        upload_date: docData.upload_date,
                        processing_status: docData.processing_status
                    }];
                } else {
                    throw new Error('Document not found');
                }
            } else {
                const data = await response.json();
                this.versions = data.versions;
            }

            this.populateVersionSelects();
            this.updateVersionInfo();

        } catch (error) {
            console.error('Failed to load document versions:', error);
            this.showError('Failed to load document versions');
        }
    }

    populateVersionSelects() {
        // Clear existing options
        this.versionSelect.innerHTML = '';
        this.compareSelect.innerHTML = '';

        // Add default options
        this.versionSelect.innerHTML = `
            <option value="latest">Latest Version</option>
            <option value="all">All Versions</option>
        `;

        // Add specific version options
        this.versions.forEach(version => {
            const option = document.createElement('option');
            option.value = version.id;
            option.textContent = `v${version.version_number} - ${version.filename}`;
            if (version.is_latest_version) {
                option.textContent += ' (Latest)';
            }
            this.versionSelect.appendChild(option);

            // Also add to compare select
            const compareOption = option.cloneNode(true);
            this.compareSelect.appendChild(compareOption);
        });
    }

    toggleCompareMode() {
        this.isCompareMode = !this.isCompareMode;

        if (this.isCompareMode) {
            this.compareContainer.style.display = 'block';
            this.compareToggle.classList.add('active');
            this.compareToggle.innerHTML = '<i class="fas fa-times"></i> Cancel Compare';
        } else {
            this.compareContainer.style.display = 'none';
            this.compareToggle.classList.remove('active');
            this.compareToggle.innerHTML = '<i class="fas fa-code-compare"></i> Compare';
            this.compareVersion = null;
        }

        this.updateCompareButton();
    }

    updateVersionInfo() {
        if (!this.versions.length) {
            this.versionInfo.innerHTML = '<p class="text-muted">No versions available</p>';
            return;
        }

        let infoHtml = '';

        if (this.selectedVersion === 'latest') {
            const latestVersion = this.versions.find(v => v.is_latest_version);
            if (latestVersion) {
                infoHtml = `
                    <div class="version-details">
                        <strong>Latest Version:</strong> v${latestVersion.version_number}<br>
                        <strong>File:</strong> ${latestVersion.filename}<br>
                        <strong>Uploaded:</strong> ${new Date(latestVersion.upload_date).toLocaleDateString()}<br>
                        <strong>Status:</strong> <span class="status-${latestVersion.processing_status}">${latestVersion.processing_status}</span>
                    </div>
                `;
            }
        } else if (this.selectedVersion === 'all') {
            infoHtml = `
                <div class="version-summary">
                    <strong>All Versions:</strong> ${this.versions.length} version(s)<br>
                    <strong>Latest:</strong> v${Math.max(...this.versions.map(v => v.version_number))}
                </div>
            `;
        } else {
            const selectedVersion = this.versions.find(v => v.id === this.selectedVersion);
            if (selectedVersion) {
                infoHtml = `
                    <div class="version-details">
                        <strong>Version:</strong> v${selectedVersion.version_number}<br>
                        <strong>File:</strong> ${selectedVersion.filename}<br>
                        <strong>Uploaded:</strong> ${new Date(selectedVersion.upload_date).toLocaleDateString()}<br>
                        <strong>Status:</strong> <span class="status-${selectedVersion.processing_status}">${selectedVersion.processing_status}</span>
                        ${selectedVersion.version_description ? `<br><strong>Description:</strong> ${selectedVersion.version_description}` : ''}
                    </div>
                `;
            }
        }

        this.versionInfo.innerHTML = infoHtml;
    }

    updateCompareButton() {
        if (this.isCompareMode && this.compareVersion && this.selectedVersion !== 'all' && this.selectedVersion !== 'latest') {
            this.viewDiffBtn.disabled = false;
            this.viewDiffBtn.textContent = 'View Differences';
        } else {
            this.viewDiffBtn.disabled = true;
            this.viewDiffBtn.textContent = 'Select versions to compare';
        }
    }

    async viewDifferences() {
        if (!this.compareVersion || this.selectedVersion === 'all' || this.selectedVersion === 'latest') {
            this.showError('Please select specific versions to compare');
            return;
        }

        try {
            this.viewDiffBtn.disabled = true;
            this.viewDiffBtn.textContent = 'Generating diff...';

            const response = await fetch(
                `${this.options.apiBasePath}/history/documents/compare/${this.selectedVersion}/${this.compareVersion}`
            );

            if (!response.ok) {
                throw new Error('Failed to generate diff');
            }

            const diffData = await response.json();

            // Notify parent component about comparison request
            if (this.options.onCompareRequest) {
                this.options.onCompareRequest(diffData);
            } else {
                // Default behavior - show diff in a modal or new window
                this.showDiffModal(diffData);
            }

        } catch (error) {
            console.error('Failed to generate diff:', error);
            this.showError('Failed to generate document comparison');
        } finally {
            this.updateCompareButton();
        }
    }

    showDiffModal(diffData) {
        // Create a modal to display the diff
        const modal = document.createElement('div');
        modal.className = 'diff-modal';
        modal.innerHTML = `
            <div class="diff-modal-content">
                <div class="diff-modal-header">
                    <h3>Document Comparison</h3>
                    <button class="diff-modal-close">&times;</button>
                </div>
                <div class="diff-modal-body">
                    <div class="diff-info">
                        <div class="diff-document">
                            <strong>Document 1:</strong> ${diffData.document1.filename} (v${diffData.document1.version_number})
                        </div>
                        <div class="diff-document">
                            <strong>Document 2:</strong> ${diffData.document2.filename} (v${diffData.document2.version_number})
                        </div>
                    </div>
                    <div class="diff-stats">
                        <span class="diff-stat added">+${diffData.diff.stats.added_lines} lines</span>
                        <span class="diff-stat removed">-${diffData.diff.stats.removed_lines} lines</span>
                        <span class="diff-stat similarity">Similarity: ${(diffData.diff.similarity_score * 100).toFixed(1)}%</span>
                    </div>
                    <div class="diff-content">
                        ${diffData.diff.html_diff}
                    </div>
                </div>
            </div>
        `;

        // Add modal to page
        document.body.appendChild(modal);

        // Close modal functionality
        const closeBtn = modal.querySelector('.diff-modal-close');
        closeBtn.addEventListener('click', () => {
            document.body.removeChild(modal);
        });

        // Close on outside click
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                document.body.removeChild(modal);
            }
        });
    }

    notifyVersionChange() {
        if (this.options.onVersionChange) {
            this.options.onVersionChange({
                selectedVersion: this.selectedVersion,
                isCompareMode: this.isCompareMode,
                compareVersion: this.compareVersion,
                versions: this.versions
            });
        }
    }

    showError(message) {
        this.versionInfo.innerHTML = `<div class="alert alert-danger">${message}</div>`;
    }

    // Public methods
    getSelectedVersion() {
        return this.selectedVersion;
    }

    getCompareVersion() {
        return this.compareVersion;
    }

    isInCompareMode() {
        return this.isCompareMode;
    }

    getVersions() {
        return this.versions;
    }

    reset() {
        this.documentId = null;
        this.versions = [];
        this.selectedVersion = 'latest';
        this.compareVersion = null;
        this.isCompareMode = false;

        this.versionSelect.innerHTML = `
            <option value="latest">Latest Version</option>
            <option value="all">All Versions</option>
        `;
        this.compareSelect.innerHTML = '';
        this.versionInfo.innerHTML = '';
        this.compareContainer.style.display = 'none';
        this.compareToggle.classList.remove('active');
        this.compareToggle.innerHTML = '<i class="fas fa-code-compare"></i> Compare';
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = VersionSelector;
}