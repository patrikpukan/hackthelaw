/**
 * Version-Aware Search Component
 * Provides search functionality with version filtering and context
 */
class VersionAwareSearch {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        this.options = {
            apiBasePath: '/api/v1',
            onSearchResults: null,
            onVersionChange: null,
            defaultSearchType: 'latest',
            showAdvancedOptions: true,
            ...options
        };
        
        this.searchHistory = [];
        this.currentResults = [];
        this.searchType = this.options.defaultSearchType;
        this.selectedFamilies = [];
        this.versionRange = null;
        
        this.init();
    }
    
    init() {
        this.container.innerHTML = `
            <div class="version-aware-search">
                <div class="search-header">
                    <div class="search-input-group">
                        <input type="text" id="search-input" class="search-input" 
                               placeholder="Search across document versions...">
                        <button id="search-btn" class="search-btn">
                            <i class="fas fa-search"></i>
                        </button>
                    </div>
                    
                    <div class="search-options">
                        <div class="search-type-selector">
                            <label>Search in:</label>
                            <select id="search-type">
                                <option value="latest">Latest Versions Only</option>
                                <option value="all">All Versions</option>
                                <option value="specific">Specific Versions</option>
                                <option value="family">Document Families</option>
                            </select>
                        </div>
                        
                        ${this.options.showAdvancedOptions ? `
                        <button id="advanced-toggle" class="advanced-toggle">
                            <i class="fas fa-cog"></i> Advanced
                        </button>
                        ` : ''}
                    </div>
                </div>
                
                ${this.options.showAdvancedOptions ? `
                <div id="advanced-options" class="advanced-options" style="display: none;">
                    <div class="advanced-grid">
                        <div class="option-group">
                            <label for="family-selector">Document Families:</label>
                            <select id="family-selector" multiple class="family-selector">
                                <option value="">Loading families...</option>
                            </select>
                        </div>
                        
                        <div class="option-group">
                            <label for="version-range">Version Range:</label>
                            <input type="text" id="version-range" placeholder="e.g., 1-3, latest, 2">
                        </div>
                        
                        <div class="option-group">
                            <label for="search-limit">Results Limit:</label>
                            <select id="search-limit">
                                <option value="25">25 results</option>
                                <option value="50" selected>50 results</option>
                                <option value="100">100 results</option>
                            </select>
                        </div>
                        
                        <div class="option-group">
                            <label>
                                <input type="checkbox" id="include-snippets" checked>
                                Include text snippets
                            </label>
                        </div>
                    </div>
                </div>
                ` : ''}
                
                <div class="search-results">
                    <div id="search-status" class="search-status"></div>
                    <div id="results-container" class="results-container"></div>
                </div>
                
                <div id="search-history" class="search-history" style="display: none;">
                    <h4>Recent Searches</h4>
                    <div id="history-list" class="history-list"></div>
                </div>
            </div>
        `;
        
        this.attachEventListeners();
        this.loadDocumentFamilies();
    }
    
    attachEventListeners() {
        // Search functionality
        document.getElementById('search-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.performSearch();
            }
        });
        
        document.getElementById('search-btn').addEventListener('click', () => {
            this.performSearch();
        });
        
        // Search type change
        document.getElementById('search-type').addEventListener('change', (e) => {
            this.searchType = e.target.value;
            this.updateAdvancedOptions();
            if (this.options.onVersionChange) {
                this.options.onVersionChange(this.searchType);
            }
        });
        
        // Advanced options toggle
        if (this.options.showAdvancedOptions) {
            document.getElementById('advanced-toggle').addEventListener('click', () => {
                this.toggleAdvancedOptions();
            });
            
            // Family selector
            document.getElementById('family-selector').addEventListener('change', (e) => {
                this.selectedFamilies = Array.from(e.target.selectedOptions).map(opt => opt.value);
            });
            
            // Version range
            document.getElementById('version-range').addEventListener('change', (e) => {
                this.versionRange = e.target.value;
            });
        }
    }
    
    async loadDocumentFamilies() {
        if (!this.options.showAdvancedOptions) return;
        
        try {
            const response = await fetch(`${this.options.apiBasePath}/history/families`);
            if (!response.ok) throw new Error('Failed to load families');
            
            const data = await response.json();
            const familySelector = document.getElementById('family-selector');
            
            familySelector.innerHTML = data.families.map(family => `
                <option value="${family.family_id}">
                    ${family.filenames} (${family.version_count} versions)
                </option>
            `).join('');
            
        } catch (error) {
            console.error('Failed to load document families:', error);
        }
    }
    
    async performSearch() {
        const query = document.getElementById('search-input').value.trim();
        if (!query) return;
        
        this.showSearchStatus('Searching...');
        
        try {
            const searchParams = this.buildSearchParams(query);
            const response = await fetch(
                `${this.options.apiBasePath}/history/search/versions?${new URLSearchParams(searchParams)}`
            );
            
            if (!response.ok) throw new Error('Search failed');
            
            const results = await response.json();
            this.currentResults = results.results;
            
            this.displayResults(results);
            this.addToSearchHistory(query, results);
            
            if (this.options.onSearchResults) {
                this.options.onSearchResults(results);
            }
            
        } catch (error) {
            console.error('Search error:', error);
            this.showSearchStatus('Search failed. Please try again.', 'error');
        }
    }
    
    buildSearchParams(query) {
        const params = {
            query: query,
            search_type: this.searchType,
            limit: this.options.showAdvancedOptions ? 
                   document.getElementById('search-limit').value : '50'
        };
        
        if (this.selectedFamilies.length > 0) {
            params.family_id = this.selectedFamilies[0]; // API supports single family for now
        }
        
        if (this.versionRange) {
            params.version_range = this.versionRange;
        }
        
        return params;
    }
    
    displayResults(results) {
        const container = document.getElementById('results-container');
        
        if (results.results.length === 0) {
            this.showSearchStatus('No results found.', 'info');
            container.innerHTML = '';
            return;
        }
        
        this.showSearchStatus(`Found ${results.total_matches} result(s) in ${results.search_type} mode`);
        
        const resultsHtml = results.results.map(result => this.renderSearchResult(result)).join('');
        container.innerHTML = `<div class="search-results-list">${resultsHtml}</div>`;
    }
    
    renderSearchResult(result) {
        const snippetsHtml = result.snippets.map(snippet => `
            <div class="result-snippet">
                <span class="snippet-text">${this.highlightQuery(snippet.text, result.query)}</span>
                <span class="snippet-position">Position: ${snippet.position}</span>
            </div>
        `).join('');
        
        return `
            <div class="search-result-item" data-document-id="${result.document_id}">
                <div class="result-header">
                    <h4 class="result-title">${result.filename}</h4>
                    <div class="result-meta">
                        <span class="version-info">
                            v${result.version_number}
                            ${result.is_latest_version ? '<span class="latest-badge">Latest</span>' : ''}
                        </span>
                        <span class="upload-date">${new Date(result.upload_date).toLocaleDateString()}</span>
                        <span class="match-count">${result.match_count} matches</span>
                    </div>
                </div>
                
                ${snippetsHtml ? `
                <div class="result-snippets">
                    ${snippetsHtml}
                </div>
                ` : ''}
                
                <div class="result-actions">
                    <button class="btn-sm btn-primary" onclick="versionAwareSearch.viewDocument('${result.document_id}')">
                        View Document
                    </button>
                    ${result.document_family_id ? `
                    <button class="btn-sm btn-outline" onclick="versionAwareSearch.viewFamily('${result.document_family_id}')">
                        View All Versions
                    </button>
                    ` : ''}
                </div>
            </div>
        `;
    }
    
    highlightQuery(text, query) {
        if (!query) return text;
        
        const regex = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
        return text.replace(regex, '<mark>$1</mark>');
    }
    
    showSearchStatus(message, type = 'info') {
        const statusEl = document.getElementById('search-status');
        statusEl.innerHTML = `<div class="status-message status-${type}">${message}</div>`;
    }
    
    toggleAdvancedOptions() {
        const advancedOptions = document.getElementById('advanced-options');
        const isVisible = advancedOptions.style.display !== 'none';
        
        advancedOptions.style.display = isVisible ? 'none' : 'block';
        
        const toggleBtn = document.getElementById('advanced-toggle');
        toggleBtn.classList.toggle('active', !isVisible);
    }
    
    updateAdvancedOptions() {
        // Update advanced options based on search type
        const familyGroup = document.querySelector('.option-group:has(#family-selector)');
        const versionGroup = document.querySelector('.option-group:has(#version-range)');
        
        if (familyGroup && versionGroup) {
            const showFamilyOptions = this.searchType === 'family' || this.searchType === 'specific';
            const showVersionOptions = this.searchType === 'specific';
            
            familyGroup.style.display = showFamilyOptions ? 'block' : 'none';
            versionGroup.style.display = showVersionOptions ? 'block' : 'none';
        }
    }
    
    addToSearchHistory(query, results) {
        const historyItem = {
            query: query,
            timestamp: new Date().toISOString(),
            resultCount: results.total_matches,
            searchType: this.searchType
        };
        
        this.searchHistory.unshift(historyItem);
        this.searchHistory = this.searchHistory.slice(0, 10); // Keep last 10 searches
        
        this.updateSearchHistoryDisplay();
    }
    
    updateSearchHistoryDisplay() {
        if (this.searchHistory.length === 0) return;
        
        const historyContainer = document.getElementById('search-history');
        const historyList = document.getElementById('history-list');
        
        historyList.innerHTML = this.searchHistory.map(item => `
            <div class="history-item" onclick="versionAwareSearch.repeatSearch('${item.query}')">
                <span class="history-query">${item.query}</span>
                <span class="history-meta">${item.resultCount} results • ${item.searchType}</span>
                <span class="history-time">${new Date(item.timestamp).toLocaleTimeString()}</span>
            </div>
        `).join('');
        
        historyContainer.style.display = 'block';
    }
    
    repeatSearch(query) {
        document.getElementById('search-input').value = query;
        this.performSearch();
    }
    
    viewDocument(documentId) {
        // Default behavior - can be overridden by options
        window.open(`/document/${documentId}`, '_blank');
    }
    
    viewFamily(familyId) {
        // Default behavior - can be overridden by options
        window.open(`/family/${familyId}`, '_blank');
    }
    
    // Public methods
    setQuery(query) {
        document.getElementById('search-input').value = query;
    }
    
    setSearchType(type) {
        this.searchType = type;
        document.getElementById('search-type').value = type;
        this.updateAdvancedOptions();
    }
    
    getResults() {
        return this.currentResults;
    }
    
    clearResults() {
        document.getElementById('results-container').innerHTML = '';
        document.getElementById('search-status').innerHTML = '';
        this.currentResults = [];
    }
    
    getSearchHistory() {
        return this.searchHistory;
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = VersionAwareSearch;
}
