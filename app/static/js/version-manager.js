/**
 * Version Manager Component
 * Comprehensive interface for managing document versions
 */
class VersionManager {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        this.options = {
            apiBasePath: '/api/v1',
            onVersionSelect: null,
            onCompareRequest: null,
            ...options
        };
        
        this.currentFamily = null;
        this.versions = [];
        this.selectedVersions = new Set();
        this.viewMode = 'timeline'; // 'timeline', 'grid', 'list'
        
        this.init();
    }
    
    init() {
        this.container.innerHTML = `
            <div class="version-manager">
                <div class="version-manager__header">
                    <div class="header-controls">
                        <div class="view-mode-toggle">
                            <button class="view-btn active" data-mode="timeline">
                                <i class="fas fa-project-diagram"></i> Timeline
                            </button>
                            <button class="view-btn" data-mode="grid">
                                <i class="fas fa-th"></i> Grid
                            </button>
                            <button class="view-btn" data-mode="list">
                                <i class="fas fa-list"></i> List
                            </button>
                        </div>
                        <div class="action-buttons">
                            <button id="compare-selected" class="btn btn-primary" disabled>
                                <i class="fas fa-code-compare"></i> Compare Selected
                            </button>
                            <button id="upload-version" class="btn btn-success">
                                <i class="fas fa-upload"></i> Upload New Version
                            </button>
                        </div>
                    </div>
                </div>
                
                <div class="version-manager__content">
                    <div id="family-info" class="family-info"></div>
                    <div id="versions-display" class="versions-display"></div>
                </div>
                
                <div id="upload-modal" class="upload-modal" style="display: none;">
                    <div class="upload-modal-content">
                        <div class="upload-modal-header">
                            <h3>Upload New Version</h3>
                            <button class="upload-modal-close">&times;</button>
                        </div>
                        <div class="upload-modal-body">
                            <form id="version-upload-form">
                                <div class="form-group">
                                    <label for="version-file">Select File:</label>
                                    <input type="file" id="version-file" accept=".pdf,.docx,.txt,.rtf" required>
                                </div>
                                <div class="form-group">
                                    <label for="version-tag">Version Tag (optional):</label>
                                    <input type="text" id="version-tag" placeholder="e.g., v2.0, draft, final">
                                </div>
                                <div class="form-group">
                                    <label for="version-description">Description:</label>
                                    <textarea id="version-description" placeholder="Describe the changes in this version..."></textarea>
                                </div>
                                <div class="form-group">
                                    <label for="version-author">Author:</label>
                                    <input type="text" id="version-author" placeholder="Your name">
                                </div>
                                <div class="form-actions">
                                    <button type="button" class="btn btn-secondary" id="cancel-upload">Cancel</button>
                                    <button type="submit" class="btn btn-primary">Upload Version</button>
                                </div>
                            </form>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        this.attachEventListeners();
    }
    
    attachEventListeners() {
        // View mode toggle
        this.container.querySelectorAll('.view-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.setViewMode(e.target.dataset.mode);
            });
        });
        
        // Compare selected versions
        document.getElementById('compare-selected').addEventListener('click', () => {
            this.compareSelectedVersions();
        });
        
        // Upload new version
        document.getElementById('upload-version').addEventListener('click', () => {
            this.showUploadModal();
        });
        
        // Upload modal events
        document.getElementById('upload-modal').addEventListener('click', (e) => {
            if (e.target.id === 'upload-modal') {
                this.hideUploadModal();
            }
        });
        
        document.querySelector('.upload-modal-close').addEventListener('click', () => {
            this.hideUploadModal();
        });
        
        document.getElementById('cancel-upload').addEventListener('click', () => {
            this.hideUploadModal();
        });
        
        document.getElementById('version-upload-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.uploadNewVersion();
        });
    }
    
    async loadFamily(familyId) {
        this.currentFamily = familyId;
        
        try {
            // Load family timeline
            const response = await fetch(`${this.options.apiBasePath}/history/families/${familyId}/timeline`);
            
            if (!response.ok) {
                throw new Error('Failed to load family timeline');
            }
            
            const data = await response.json();
            this.versions = data.timeline;
            
            this.renderFamilyInfo(data);
            this.renderVersions();
            
        } catch (error) {
            console.error('Failed to load family:', error);
            this.showError('Failed to load document family');
        }
    }
    
    renderFamilyInfo(familyData) {
        const familyInfo = document.getElementById('family-info');
        const latestVersion = this.versions.find(v => v.version_number === Math.max(...this.versions.map(v => v.version_number)));
        
        familyInfo.innerHTML = `
            <div class="family-summary">
                <h3>Document Family: ${latestVersion ? latestVersion.filename : 'Unknown'}</h3>
                <div class="family-stats">
                    <span class="stat">
                        <i class="fas fa-file-alt"></i>
                        ${familyData.total_versions} version${familyData.total_versions !== 1 ? 's' : ''}
                    </span>
                    <span class="stat">
                        <i class="fas fa-clock"></i>
                        Latest: ${new Date(latestVersion.upload_date).toLocaleDateString()}
                    </span>
                    <span class="stat">
                        <i class="fas fa-edit"></i>
                        ${this.versions.reduce((sum, v) => sum + v.change_count, 0)} total changes
                    </span>
                </div>
            </div>
        `;
    }
    
    renderVersions() {
        const display = document.getElementById('versions-display');
        
        switch (this.viewMode) {
            case 'timeline':
                this.renderTimelineView(display);
                break;
            case 'grid':
                this.renderGridView(display);
                break;
            case 'list':
                this.renderListView(display);
                break;
        }
    }
    
    renderTimelineView(container) {
        const timelineHtml = this.versions.map((version, index) => {
            const isSelected = this.selectedVersions.has(version.document_id);
            const isLatest = index === 0; // Assuming sorted by version number desc
            
            return `
                <div class="timeline-item ${isSelected ? 'selected' : ''}" data-version-id="${version.document_id}">
                    <div class="timeline-marker ${isLatest ? 'latest' : ''}">
                        <span class="version-number">v${version.version_number}</span>
                    </div>
                    <div class="timeline-content">
                        <div class="version-header">
                            <h4>${version.filename}</h4>
                            <div class="version-meta">
                                <span class="upload-date">${new Date(version.upload_date).toLocaleDateString()}</span>
                                ${version.author ? `<span class="author">by ${version.author}</span>` : ''}
                                ${version.version_tag ? `<span class="version-tag">${version.version_tag}</span>` : ''}
                            </div>
                        </div>
                        <div class="version-details">
                            ${version.version_description || version.change_summary || 'No description available'}
                        </div>
                        <div class="version-stats">
                            <span class="stat-item">
                                <i class="fas fa-exchange-alt"></i>
                                ${version.change_count} changes
                            </span>
                            <span class="status status-${version.processing_status}">
                                ${version.processing_status}
                            </span>
                        </div>
                        <div class="version-actions">
                            <button class="btn-sm btn-outline" onclick="versionManager.selectVersion('${version.document_id}')">
                                ${isSelected ? 'Deselect' : 'Select'}
                            </button>
                            <button class="btn-sm btn-primary" onclick="versionManager.viewVersion('${version.document_id}')">
                                View
                            </button>
                        </div>
                    </div>
                </div>
            `;
        }).join('');
        
        container.innerHTML = `<div class="timeline-view">${timelineHtml}</div>`;
    }
    
    renderGridView(container) {
        const gridHtml = this.versions.map(version => {
            const isSelected = this.selectedVersions.has(version.document_id);
            
            return `
                <div class="grid-item ${isSelected ? 'selected' : ''}" data-version-id="${version.document_id}">
                    <div class="grid-header">
                        <span class="version-number">v${version.version_number}</span>
                        ${version.version_tag ? `<span class="version-tag">${version.version_tag}</span>` : ''}
                    </div>
                    <div class="grid-content">
                        <h5>${version.filename}</h5>
                        <p class="upload-date">${new Date(version.upload_date).toLocaleDateString()}</p>
                        <p class="change-count">${version.change_count} changes</p>
                    </div>
                    <div class="grid-actions">
                        <button class="btn-sm" onclick="versionManager.selectVersion('${version.document_id}')">
                            ${isSelected ? '✓' : '○'}
                        </button>
                    </div>
                </div>
            `;
        }).join('');
        
        container.innerHTML = `<div class="grid-view">${gridHtml}</div>`;
    }
    
    renderListView(container) {
        const listHtml = this.versions.map(version => {
            const isSelected = this.selectedVersions.has(version.document_id);
            
            return `
                <div class="list-item ${isSelected ? 'selected' : ''}" data-version-id="${version.document_id}">
                    <div class="list-checkbox">
                        <input type="checkbox" ${isSelected ? 'checked' : ''} 
                               onchange="versionManager.selectVersion('${version.document_id}')">
                    </div>
                    <div class="list-content">
                        <div class="list-main">
                            <span class="version-number">v${version.version_number}</span>
                            <span class="filename">${version.filename}</span>
                            ${version.version_tag ? `<span class="version-tag">${version.version_tag}</span>` : ''}
                        </div>
                        <div class="list-meta">
                            <span class="upload-date">${new Date(version.upload_date).toLocaleDateString()}</span>
                            ${version.author ? `<span class="author">${version.author}</span>` : ''}
                            <span class="change-count">${version.change_count} changes</span>
                            <span class="status status-${version.processing_status}">${version.processing_status}</span>
                        </div>
                    </div>
                    <div class="list-actions">
                        <button class="btn-sm btn-primary" onclick="versionManager.viewVersion('${version.document_id}')">
                            View
                        </button>
                    </div>
                </div>
            `;
        }).join('');
        
        container.innerHTML = `<div class="list-view">${listHtml}</div>`;
    }
    
    setViewMode(mode) {
        this.viewMode = mode;
        
        // Update active button
        this.container.querySelectorAll('.view-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.mode === mode);
        });
        
        this.renderVersions();
    }
    
    selectVersion(versionId) {
        if (this.selectedVersions.has(versionId)) {
            this.selectedVersions.delete(versionId);
        } else {
            this.selectedVersions.add(versionId);
        }
        
        this.updateSelectionUI();
        this.updateCompareButton();
    }
    
    updateSelectionUI() {
        this.container.querySelectorAll('[data-version-id]').forEach(item => {
            const versionId = item.dataset.versionId;
            item.classList.toggle('selected', this.selectedVersions.has(versionId));
        });
    }
    
    updateCompareButton() {
        const compareBtn = document.getElementById('compare-selected');
        const canCompare = this.selectedVersions.size === 2;
        
        compareBtn.disabled = !canCompare;
        compareBtn.textContent = canCompare ? 
            'Compare Selected' : 
            `Select ${2 - this.selectedVersions.size} more version${this.selectedVersions.size === 1 ? '' : 's'}`;
    }
    
    compareSelectedVersions() {
        if (this.selectedVersions.size !== 2) {
            alert('Please select exactly 2 versions to compare');
            return;
        }
        
        const versionIds = Array.from(this.selectedVersions);
        
        if (this.options.onCompareRequest) {
            this.options.onCompareRequest(versionIds[0], versionIds[1]);
        } else {
            // Default behavior - open comparison in new window
            window.open(`/compare/${versionIds[0]}/${versionIds[1]}`, '_blank');
        }
    }
    
    viewVersion(versionId) {
        if (this.options.onVersionSelect) {
            this.options.onVersionSelect(versionId);
        } else {
            // Default behavior - open version in new window
            window.open(`/document/${versionId}`, '_blank');
        }
    }
    
    showUploadModal() {
        document.getElementById('upload-modal').style.display = 'flex';
    }
    
    hideUploadModal() {
        document.getElementById('upload-modal').style.display = 'none';
        document.getElementById('version-upload-form').reset();
    }
    
    async uploadNewVersion() {
        const form = document.getElementById('version-upload-form');
        const formData = new FormData();
        
        const file = document.getElementById('version-file').files[0];
        const versionTag = document.getElementById('version-tag').value;
        const description = document.getElementById('version-description').value;
        const author = document.getElementById('version-author').value;
        
        if (!file) {
            alert('Please select a file');
            return;
        }
        
        formData.append('file', file);
        if (versionTag) formData.append('version_tag', versionTag);
        if (description) formData.append('version_description', description);
        if (author) formData.append('author', author);
        
        try {
            const response = await fetch(
                `${this.options.apiBasePath}/documents/upload-version/${this.currentFamily}`,
                {
                    method: 'POST',
                    body: formData
                }
            );
            
            if (!response.ok) {
                throw new Error('Failed to upload version');
            }
            
            const result = await response.json();
            
            this.hideUploadModal();
            this.showSuccess('Version uploaded successfully');
            
            // Reload the family to show the new version
            this.loadFamily(this.currentFamily);
            
        } catch (error) {
            console.error('Failed to upload version:', error);
            this.showError('Failed to upload new version');
        }
    }
    
    showError(message) {
        // Simple error display - could be enhanced with a proper notification system
        alert(`Error: ${message}`);
    }
    
    showSuccess(message) {
        // Simple success display - could be enhanced with a proper notification system
        alert(message);
    }
    
    // Public methods
    getSelectedVersions() {
        return Array.from(this.selectedVersions);
    }
    
    clearSelection() {
        this.selectedVersions.clear();
        this.updateSelectionUI();
        this.updateCompareButton();
    }
    
    getViewMode() {
        return this.viewMode;
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = VersionManager;
}
