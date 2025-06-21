/**
 * Version Toggle Controls Component
 * Provides UI controls for enabling/disabling version analysis mode and controlling LLM version awareness
 */
class VersionToggleControls {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        this.options = {
            apiBasePath: '/api/v1',
            onModeChange: null,
            onSettingsChange: null,
            defaultMode: 'latest',
            showAdvancedControls: true,
            persistSettings: true,
            ...options
        };
        
        this.settings = {
            versionAnalysisEnabled: true,
            llmVersionAwareness: true,
            searchMode: this.options.defaultMode,
            includeVersionContext: true,
            autoDetectVersions: true,
            showVersionMetadata: true,
            compareMode: false,
            selectedVersions: [],
            excludedVersions: []
        };
        
        this.loadSettings();
        this.init();
    }
    
    init() {
        this.container.innerHTML = `
            <div class="version-toggle-controls">
                <div class="controls-header">
                    <h3>Version Analysis Controls</h3>
                    <div class="master-toggle">
                        <label class="toggle-switch">
                            <input type="checkbox" id="master-toggle" ${this.settings.versionAnalysisEnabled ? 'checked' : ''}>
                            <span class="toggle-slider"></span>
                            <span class="toggle-label">Version Analysis</span>
                        </label>
                    </div>
                </div>
                
                <div id="controls-content" class="controls-content" ${!this.settings.versionAnalysisEnabled ? 'style="display: none;"' : ''}>
                    <!-- Core Settings -->
                    <div class="control-section">
                        <h4>Core Settings</h4>
                        <div class="control-grid">
                            <div class="control-item">
                                <label class="toggle-switch">
                                    <input type="checkbox" id="llm-awareness" ${this.settings.llmVersionAwareness ? 'checked' : ''}>
                                    <span class="toggle-slider"></span>
                                    <span class="toggle-label">LLM Version Awareness</span>
                                </label>
                                <small class="control-description">Enable AI to understand document versions in responses</small>
                            </div>
                            
                            <div class="control-item">
                                <label class="toggle-switch">
                                    <input type="checkbox" id="include-context" ${this.settings.includeVersionContext ? 'checked' : ''}>
                                    <span class="toggle-slider"></span>
                                    <span class="toggle-label">Include Version Context</span>
                                </label>
                                <small class="control-description">Show version information in AI responses</small>
                            </div>
                            
                            <div class="control-item">
                                <label class="toggle-switch">
                                    <input type="checkbox" id="auto-detect" ${this.settings.autoDetectVersions ? 'checked' : ''}>
                                    <span class="toggle-slider"></span>
                                    <span class="toggle-label">Auto-Detect Versions</span>
                                </label>
                                <small class="control-description">Automatically identify document versions during upload</small>
                            </div>
                            
                            <div class="control-item">
                                <label class="toggle-switch">
                                    <input type="checkbox" id="show-metadata" ${this.settings.showVersionMetadata ? 'checked' : ''}>
                                    <span class="toggle-slider"></span>
                                    <span class="toggle-label">Show Version Metadata</span>
                                </label>
                                <small class="control-description">Display version numbers, dates, and change summaries</small>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Search Mode -->
                    <div class="control-section">
                        <h4>Search Mode</h4>
                        <div class="search-mode-selector">
                            <div class="mode-options">
                                <label class="mode-option">
                                    <input type="radio" name="search-mode" value="latest" ${this.settings.searchMode === 'latest' ? 'checked' : ''}>
                                    <span class="mode-card">
                                        <i class="fas fa-star"></i>
                                        <strong>Latest Only</strong>
                                        <small>Search only the most recent versions</small>
                                    </span>
                                </label>
                                
                                <label class="mode-option">
                                    <input type="radio" name="search-mode" value="all" ${this.settings.searchMode === 'all' ? 'checked' : ''}>
                                    <span class="mode-card">
                                        <i class="fas fa-layer-group"></i>
                                        <strong>All Versions</strong>
                                        <small>Search across all document versions</small>
                                    </span>
                                </label>
                                
                                <label class="mode-option">
                                    <input type="radio" name="search-mode" value="specific" ${this.settings.searchMode === 'specific' ? 'checked' : ''}>
                                    <span class="mode-card">
                                        <i class="fas fa-bullseye"></i>
                                        <strong>Specific Versions</strong>
                                        <small>Search only selected versions</small>
                                    </span>
                                </label>
                                
                                <label class="mode-option">
                                    <input type="radio" name="search-mode" value="compare" ${this.settings.searchMode === 'compare' ? 'checked' : ''}>
                                    <span class="mode-card">
                                        <i class="fas fa-code-compare"></i>
                                        <strong>Compare Mode</strong>
                                        <small>Compare between versions</small>
                                    </span>
                                </label>
                            </div>
                        </div>
                    </div>
                    
                    ${this.options.showAdvancedControls ? this.renderAdvancedControls() : ''}
                    
                    <!-- Status Display -->
                    <div class="control-section">
                        <h4>Current Status</h4>
                        <div id="status-display" class="status-display">
                            ${this.renderStatusDisplay()}
                        </div>
                    </div>
                    
                    <!-- Action Buttons -->
                    <div class="control-actions">
                        <button id="apply-settings" class="btn btn-primary">Apply Settings</button>
                        <button id="reset-settings" class="btn btn-secondary">Reset to Defaults</button>
                        <button id="export-settings" class="btn btn-outline">Export Settings</button>
                        <button id="import-settings" class="btn btn-outline">Import Settings</button>
                    </div>
                </div>
            </div>
        `;
        
        this.attachEventListeners();
        this.updateUI();
    }
    
    renderAdvancedControls() {
        return `
            <!-- Advanced Settings -->
            <div class="control-section advanced-section">
                <h4>Advanced Settings</h4>
                <div class="control-grid">
                    <div class="control-item">
                        <label for="version-threshold">Version Similarity Threshold:</label>
                        <input type="range" id="version-threshold" min="0.1" max="1.0" step="0.1" value="0.7">
                        <span class="range-value">0.7</span>
                        <small class="control-description">Minimum similarity to consider documents as versions</small>
                    </div>
                    
                    <div class="control-item">
                        <label for="context-window">Version Context Window:</label>
                        <select id="context-window">
                            <option value="1">±1 version</option>
                            <option value="2" selected>±2 versions</option>
                            <option value="3">±3 versions</option>
                            <option value="all">All versions</option>
                        </select>
                        <small class="control-description">How many versions to include in context</small>
                    </div>
                    
                    <div class="control-item">
                        <label for="diff-granularity">Diff Granularity:</label>
                        <select id="diff-granularity">
                            <option value="line">Line-by-line</option>
                            <option value="word" selected>Word-by-word</option>
                            <option value="character">Character-by-character</option>
                            <option value="semantic">Semantic blocks</option>
                        </select>
                        <small class="control-description">Level of detail for version comparisons</small>
                    </div>
                    
                    <div class="control-item">
                        <label class="toggle-switch">
                            <input type="checkbox" id="cache-diffs" checked>
                            <span class="toggle-slider"></span>
                            <span class="toggle-label">Cache Diff Results</span>
                        </label>
                        <small class="control-description">Store comparison results for faster access</small>
                    </div>
                </div>
            </div>
            
            <!-- Version Selection -->
            <div class="control-section" id="version-selection-section" style="display: none;">
                <h4>Version Selection</h4>
                <div class="version-selection">
                    <div class="selection-controls">
                        <button id="select-all-versions" class="btn btn-sm">Select All</button>
                        <button id="clear-selection" class="btn btn-sm">Clear Selection</button>
                        <button id="invert-selection" class="btn btn-sm">Invert Selection</button>
                    </div>
                    <div id="version-list" class="version-list">
                        <!-- Version list will be populated dynamically -->
                    </div>
                </div>
            </div>
        `;
    }
    
    renderStatusDisplay() {
        const status = this.getSystemStatus();
        return `
            <div class="status-grid">
                <div class="status-item">
                    <span class="status-label">Version Analysis:</span>
                    <span class="status-value ${status.versionAnalysis ? 'enabled' : 'disabled'}">
                        ${status.versionAnalysis ? 'Enabled' : 'Disabled'}
                    </span>
                </div>
                <div class="status-item">
                    <span class="status-label">Search Mode:</span>
                    <span class="status-value">${status.searchMode}</span>
                </div>
                <div class="status-item">
                    <span class="status-label">LLM Awareness:</span>
                    <span class="status-value ${status.llmAwareness ? 'enabled' : 'disabled'}">
                        ${status.llmAwareness ? 'Enabled' : 'Disabled'}
                    </span>
                </div>
                <div class="status-item">
                    <span class="status-label">Active Versions:</span>
                    <span class="status-value">${status.activeVersions}</span>
                </div>
            </div>
        `;
    }
    
    attachEventListeners() {
        // Master toggle
        document.getElementById('master-toggle').addEventListener('change', (e) => {
            this.settings.versionAnalysisEnabled = e.target.checked;
            this.toggleControlsVisibility();
            this.notifyChange();
        });
        
        // Core settings toggles
        document.getElementById('llm-awareness').addEventListener('change', (e) => {
            this.settings.llmVersionAwareness = e.target.checked;
            this.notifyChange();
        });
        
        document.getElementById('include-context').addEventListener('change', (e) => {
            this.settings.includeVersionContext = e.target.checked;
            this.notifyChange();
        });
        
        document.getElementById('auto-detect').addEventListener('change', (e) => {
            this.settings.autoDetectVersions = e.target.checked;
            this.notifyChange();
        });
        
        document.getElementById('show-metadata').addEventListener('change', (e) => {
            this.settings.showVersionMetadata = e.target.checked;
            this.notifyChange();
        });
        
        // Search mode selection
        document.querySelectorAll('input[name="search-mode"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.settings.searchMode = e.target.value;
                this.updateSearchModeUI();
                this.notifyChange();
            });
        });
        
        // Advanced controls
        if (this.options.showAdvancedControls) {
            this.attachAdvancedListeners();
        }
        
        // Action buttons
        document.getElementById('apply-settings').addEventListener('click', () => {
            this.applySettings();
        });
        
        document.getElementById('reset-settings').addEventListener('click', () => {
            this.resetSettings();
        });
        
        document.getElementById('export-settings').addEventListener('click', () => {
            this.exportSettings();
        });
        
        document.getElementById('import-settings').addEventListener('click', () => {
            this.importSettings();
        });
    }
    
    attachAdvancedListeners() {
        // Version threshold
        const thresholdSlider = document.getElementById('version-threshold');
        thresholdSlider.addEventListener('input', (e) => {
            document.querySelector('.range-value').textContent = e.target.value;
        });
        
        // Version selection controls
        document.getElementById('select-all-versions')?.addEventListener('click', () => {
            this.selectAllVersions();
        });
        
        document.getElementById('clear-selection')?.addEventListener('click', () => {
            this.clearVersionSelection();
        });
        
        document.getElementById('invert-selection')?.addEventListener('click', () => {
            this.invertVersionSelection();
        });
    }
    
    toggleControlsVisibility() {
        const content = document.getElementById('controls-content');
        content.style.display = this.settings.versionAnalysisEnabled ? 'block' : 'none';
        this.updateStatusDisplay();
    }
    
    updateSearchModeUI() {
        const versionSection = document.getElementById('version-selection-section');
        if (versionSection) {
            versionSection.style.display = this.settings.searchMode === 'specific' ? 'block' : 'none';
        }
        this.updateStatusDisplay();
    }
    
    updateStatusDisplay() {
        const statusDisplay = document.getElementById('status-display');
        if (statusDisplay) {
            statusDisplay.innerHTML = this.renderStatusDisplay();
        }
    }
    
    updateUI() {
        this.toggleControlsVisibility();
        this.updateSearchModeUI();
        this.updateStatusDisplay();
    }
    
    getSystemStatus() {
        return {
            versionAnalysis: this.settings.versionAnalysisEnabled,
            searchMode: this.settings.searchMode,
            llmAwareness: this.settings.llmVersionAwareness,
            activeVersions: this.settings.selectedVersions.length || 'All'
        };
    }
    
    applySettings() {
        this.saveSettings();
        this.notifyChange();
        this.showNotification('Settings applied successfully', 'success');
    }
    
    resetSettings() {
        this.settings = {
            versionAnalysisEnabled: true,
            llmVersionAwareness: true,
            searchMode: this.options.defaultMode,
            includeVersionContext: true,
            autoDetectVersions: true,
            showVersionMetadata: true,
            compareMode: false,
            selectedVersions: [],
            excludedVersions: []
        };
        
        this.updateFormValues();
        this.updateUI();
        this.notifyChange();
        this.showNotification('Settings reset to defaults', 'info');
    }
    
    updateFormValues() {
        document.getElementById('master-toggle').checked = this.settings.versionAnalysisEnabled;
        document.getElementById('llm-awareness').checked = this.settings.llmVersionAwareness;
        document.getElementById('include-context').checked = this.settings.includeVersionContext;
        document.getElementById('auto-detect').checked = this.settings.autoDetectVersions;
        document.getElementById('show-metadata').checked = this.settings.showVersionMetadata;
        
        document.querySelector(`input[name="search-mode"][value="${this.settings.searchMode}"]`).checked = true;
    }
    
    exportSettings() {
        const settingsJson = JSON.stringify(this.settings, null, 2);
        const blob = new Blob([settingsJson], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = 'version-control-settings.json';
        a.click();
        
        URL.revokeObjectURL(url);
        this.showNotification('Settings exported successfully', 'success');
    }
    
    importSettings() {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.json';
        
        input.onchange = (e) => {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    try {
                        const importedSettings = JSON.parse(e.target.result);
                        this.settings = { ...this.settings, ...importedSettings };
                        this.updateFormValues();
                        this.updateUI();
                        this.notifyChange();
                        this.showNotification('Settings imported successfully', 'success');
                    } catch (error) {
                        this.showNotification('Failed to import settings: Invalid JSON', 'error');
                    }
                };
                reader.readAsText(file);
            }
        };
        
        input.click();
    }
    
    saveSettings() {
        if (this.options.persistSettings) {
            localStorage.setItem('versionControlSettings', JSON.stringify(this.settings));
        }
    }
    
    loadSettings() {
        if (this.options.persistSettings) {
            const saved = localStorage.getItem('versionControlSettings');
            if (saved) {
                try {
                    this.settings = { ...this.settings, ...JSON.parse(saved) };
                } catch (error) {
                    console.error('Failed to load saved settings:', error);
                }
            }
        }
    }
    
    notifyChange() {
        if (this.options.onSettingsChange) {
            this.options.onSettingsChange(this.settings);
        }
        
        if (this.options.onModeChange) {
            this.options.onModeChange(this.settings.searchMode, this.settings);
        }
        
        this.saveSettings();
    }
    
    showNotification(message, type = 'info') {
        // Simple notification - could be enhanced with a proper notification system
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 12px 20px;
            border-radius: 4px;
            color: white;
            background: ${type === 'success' ? '#28a745' : type === 'error' ? '#dc3545' : '#007bff'};
            z-index: 1000;
            animation: slideIn 0.3s ease;
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }
    
    // Public methods
    getSettings() {
        return { ...this.settings };
    }
    
    updateSettings(newSettings) {
        this.settings = { ...this.settings, ...newSettings };
        this.updateFormValues();
        this.updateUI();
        this.notifyChange();
    }
    
    isVersionAnalysisEnabled() {
        return this.settings.versionAnalysisEnabled;
    }
    
    getSearchMode() {
        return this.settings.searchMode;
    }
    
    isLLMVersionAware() {
        return this.settings.llmVersionAwareness;
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = VersionToggleControls;
}
