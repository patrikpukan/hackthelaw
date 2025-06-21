/**
 * Diff Viewer Component
 * Displays visual diffs between document versions with multiple view modes
 */
class DiffViewer {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        this.options = {
            apiBasePath: '/api/v1',
            defaultViewMode: 'unified', // 'unified', 'side-by-side', 'inline'
            showLineNumbers: true,
            contextLines: 3,
            ...options
        };
        
        this.diffData = null;
        this.viewMode = this.options.defaultViewMode;
        this.showStats = true;
        
        this.init();
    }
    
    init() {
        this.container.innerHTML = `
            <div class="diff-viewer">
                <div class="diff-viewer__header">
                    <div class="diff-viewer__controls">
                        <div class="view-mode-selector">
                            <label>View Mode:</label>
                            <select id="view-mode-select">
                                <option value="unified">Unified</option>
                                <option value="side-by-side">Side by Side</option>
                                <option value="inline">Inline</option>
                            </select>
                        </div>
                        <div class="diff-options">
                            <label>
                                <input type="checkbox" id="show-line-numbers" ${this.options.showLineNumbers ? 'checked' : ''}>
                                Line Numbers
                            </label>
                            <label>
                                <input type="checkbox" id="show-stats" ${this.showStats ? 'checked' : ''}>
                                Statistics
                            </label>
                        </div>
                    </div>
                </div>
                
                <div class="diff-viewer__content">
                    <div id="diff-stats" class="diff-stats" style="display: ${this.showStats ? 'block' : 'none'}"></div>
                    <div id="diff-display" class="diff-display"></div>
                </div>
            </div>
        `;
        
        this.attachEventListeners();
    }
    
    attachEventListeners() {
        const viewModeSelect = document.getElementById('view-mode-select');
        const showLineNumbers = document.getElementById('show-line-numbers');
        const showStats = document.getElementById('show-stats');
        
        viewModeSelect.value = this.viewMode;
        viewModeSelect.addEventListener('change', (e) => {
            this.viewMode = e.target.value;
            this.renderDiff();
        });
        
        showLineNumbers.addEventListener('change', (e) => {
            this.options.showLineNumbers = e.target.checked;
            this.renderDiff();
        });
        
        showStats.addEventListener('change', (e) => {
            this.showStats = e.target.checked;
            document.getElementById('diff-stats').style.display = e.target.checked ? 'block' : 'none';
        });
    }
    
    async loadDiff(document1Id, document2Id, comparisonType = 'full') {
        try {
            this.showLoading();
            
            const response = await fetch(
                `${this.options.apiBasePath}/history/documents/compare/${document1Id}/${document2Id}?comparison_type=${comparisonType}`
            );
            
            if (!response.ok) {
                throw new Error('Failed to load diff');
            }
            
            this.diffData = await response.json();
            this.renderDiff();
            
        } catch (error) {
            console.error('Failed to load diff:', error);
            this.showError('Failed to load document comparison');
        }
    }
    
    setDiffData(diffData) {
        this.diffData = diffData;
        this.renderDiff();
    }
    
    renderDiff() {
        if (!this.diffData) {
            this.showMessage('No diff data available');
            return;
        }
        
        this.renderStats();
        
        switch (this.viewMode) {
            case 'unified':
                this.renderUnifiedDiff();
                break;
            case 'side-by-side':
                this.renderSideBySideDiff();
                break;
            case 'inline':
                this.renderInlineDiff();
                break;
            default:
                this.renderUnifiedDiff();
        }
    }
    
    renderStats() {
        const statsContainer = document.getElementById('diff-stats');
        const stats = this.diffData.diff.stats;
        
        const changePercentage = stats.change_percentage || 0;
        const similarity = this.diffData.diff.similarity_score || 0;
        
        statsContainer.innerHTML = `
            <div class="diff-stats-grid">
                <div class="stat-item">
                    <span class="stat-label">Added Lines:</span>
                    <span class="stat-value added">+${stats.added_lines}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Removed Lines:</span>
                    <span class="stat-value removed">-${stats.removed_lines}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Changed Sections:</span>
                    <span class="stat-value">${stats.changed_sections}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Similarity:</span>
                    <span class="stat-value">${(similarity * 100).toFixed(1)}%</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Change Impact:</span>
                    <span class="stat-value">${changePercentage.toFixed(1)}%</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Word Difference:</span>
                    <span class="stat-value">${stats.word_difference > 0 ? '+' : ''}${stats.word_difference}</span>
                </div>
            </div>
        `;
    }
    
    renderUnifiedDiff() {
        const displayContainer = document.getElementById('diff-display');
        const htmlDiff = this.diffData.diff.html_diff;
        
        displayContainer.innerHTML = `
            <div class="unified-diff">
                <div class="diff-header">
                    <div class="file-info">
                        <span class="file-label">From:</span> ${this.diffData.document1.filename} (v${this.diffData.document1.version_number})
                    </div>
                    <div class="file-info">
                        <span class="file-label">To:</span> ${this.diffData.document2.filename} (v${this.diffData.document2.version_number})
                    </div>
                </div>
                <div class="diff-content unified-content">
                    ${this.addLineNumbers(htmlDiff)}
                </div>
            </div>
        `;
    }
    
    renderSideBySideDiff() {
        const displayContainer = document.getElementById('diff-display');
        const sideBySideData = this.diffData.diff.side_by_side;
        
        if (!sideBySideData) {
            this.renderUnifiedDiff();
            return;
        }
        
        let leftContent = '';
        let rightContent = '';
        
        sideBySideData.line_pairs.forEach(pair => {
            const leftLine = this.formatSideBySideLine(pair.left, 'left');
            const rightLine = this.formatSideBySideLine(pair.right, 'right');
            
            leftContent += leftLine;
            rightContent += rightLine;
        });
        
        displayContainer.innerHTML = `
            <div class="side-by-side-diff">
                <div class="diff-header">
                    <div class="file-info left">
                        ${this.diffData.document1.filename} (v${this.diffData.document1.version_number})
                    </div>
                    <div class="file-info right">
                        ${this.diffData.document2.filename} (v${this.diffData.document2.version_number})
                    </div>
                </div>
                <div class="diff-content side-by-side-content">
                    <div class="diff-pane left-pane">
                        ${leftContent}
                    </div>
                    <div class="diff-pane right-pane">
                        ${rightContent}
                    </div>
                </div>
            </div>
        `;
    }
    
    renderInlineDiff() {
        const displayContainer = document.getElementById('diff-display');
        const inlineData = this.diffData.diff.inline_diff;
        
        if (!inlineData) {
            this.renderUnifiedDiff();
            return;
        }
        
        displayContainer.innerHTML = `
            <div class="inline-diff">
                <div class="diff-header">
                    <div class="file-info">
                        Comparing: ${this.diffData.document1.filename} (v${this.diffData.document1.version_number}) 
                        ↔ ${this.diffData.document2.filename} (v${this.diffData.document2.version_number})
                    </div>
                </div>
                <div class="diff-content inline-content">
                    ${inlineData.html}
                </div>
            </div>
        `;
    }
    
    formatSideBySideLine(lineData, side) {
        const lineNum = this.options.showLineNumbers && lineData.line_num ? 
            `<span class="line-number">${lineData.line_num}</span>` : '';
        
        const cssClass = `diff-line ${lineData.type}`;
        const escapedText = this.escapeHtml(lineData.text);
        
        return `<div class="${cssClass}">${lineNum}<span class="line-content">${escapedText}</span></div>`;
    }
    
    addLineNumbers(htmlContent) {
        if (!this.options.showLineNumbers) {
            return htmlContent;
        }
        
        const lines = htmlContent.split('\n');
        let lineNumber = 1;
        
        return lines.map(line => {
            if (line.trim() === '') return line;
            
            const lineNumSpan = `<span class="line-number">${lineNumber}</span>`;
            lineNumber++;
            
            // Insert line number at the beginning of each diff line
            if (line.includes('class="diff-')) {
                return line.replace('>', `>${lineNumSpan}`);
            }
            
            return line;
        }).join('\n');
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    showLoading() {
        document.getElementById('diff-display').innerHTML = `
            <div class="diff-loading">
                <div class="loading-spinner"></div>
                <p>Generating diff...</p>
            </div>
        `;
    }
    
    showError(message) {
        document.getElementById('diff-display').innerHTML = `
            <div class="diff-error">
                <i class="fas fa-exclamation-triangle"></i>
                <p>${message}</p>
            </div>
        `;
    }
    
    showMessage(message) {
        document.getElementById('diff-display').innerHTML = `
            <div class="diff-message">
                <p>${message}</p>
            </div>
        `;
    }
    
    // Public methods
    getViewMode() {
        return this.viewMode;
    }
    
    setViewMode(mode) {
        this.viewMode = mode;
        document.getElementById('view-mode-select').value = mode;
        this.renderDiff();
    }
    
    exportDiff(format = 'html') {
        if (!this.diffData) {
            throw new Error('No diff data to export');
        }
        
        switch (format) {
            case 'html':
                return this.exportAsHtml();
            case 'text':
                return this.diffData.diff.diff_text;
            case 'json':
                return JSON.stringify(this.diffData, null, 2);
            default:
                throw new Error(`Unsupported export format: ${format}`);
        }
    }
    
    exportAsHtml() {
        const currentContent = document.getElementById('diff-display').innerHTML;
        const stats = document.getElementById('diff-stats').innerHTML;
        
        return `
            <!DOCTYPE html>
            <html>
            <head>
                <title>Document Diff Export</title>
                <link rel="stylesheet" href="/static/css/diff-viewer.css">
            </head>
            <body>
                <div class="diff-export">
                    <h1>Document Comparison</h1>
                    <div class="export-stats">${stats}</div>
                    <div class="export-content">${currentContent}</div>
                </div>
            </body>
            </html>
        `;
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DiffViewer;
}
