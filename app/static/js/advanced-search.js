/**
 * Advanced Cross-Version Search Component
 * Provides comprehensive search functionality across document versions with advanced filtering
 */
class AdvancedSearch {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        this.options = {
            apiBasePath: '/api/v1',
            onSearchResults: null,
            onDocumentSelect: null,
            defaultScope: 'all',
            showFilters: true,
            showGrouping: true,
            ...options
        };
        
        this.searchParams = {
            query: '',
            search_scope: this.options.defaultScope,
            family_ids: null,
            version_ids: null,
            date_from: null,
            date_to: null,
            version_range: null,
            include_metadata: true,
            include_snippets: true,
            snippet_length: 200,
            max_snippets_per_doc: 3,
            group_by_family: false,
            sort_by: 'relevance',
            sort_order: 'desc',
            limit: 50,
            offset: 0
        };
        
        this.currentResults = null;
        this.searchHistory = [];
        this.availableFamilies = [];
        
        this.init();
    }
    
    init() {
        this.container.innerHTML = `
            <div class="advanced-search">
                <div class="search-header">
                    <h3>Advanced Cross-Version Search</h3>
                    <div class="search-actions">
                        <button id="save-search" class="btn btn-outline">
                            <i class="fas fa-save"></i> Save Search
                        </button>
                        <button id="load-search" class="btn btn-outline">
                            <i class="fas fa-folder-open"></i> Load Search
                        </button>
                    </div>
                </div>
                
                <div class="search-form">
                    <div class="search-input-section">
                        <div class="input-group">
                            <input type="text" id="search-query" class="search-input" 
                                   placeholder="Enter your search query...">
                            <button id="search-btn" class="search-btn">
                                <i class="fas fa-search"></i> Search
                            </button>
                        </div>
                    </div>
                    
                    ${this.options.showFilters ? this.renderFilters() : ''}
                    
                    <div class="search-options">
                        <div class="options-grid">
                            <div class="option-group">
                                <label>Sort by:</label>
                                <select id="sort-by">
                                    <option value="relevance">Relevance</option>
                                    <option value="date">Upload Date</option>
                                    <option value="version">Version Number</option>
                                    <option value="family">Document Family</option>
                                </select>
                            </div>
                            
                            <div class="option-group">
                                <label>Sort order:</label>
                                <select id="sort-order">
                                    <option value="desc">Descending</option>
                                    <option value="asc">Ascending</option>
                                </select>
                            </div>
                            
                            <div class="option-group">
                                <label>Results per page:</label>
                                <select id="results-limit">
                                    <option value="25">25</option>
                                    <option value="50" selected>50</option>
                                    <option value="100">100</option>
                                </select>
                            </div>
                            
                            ${this.options.showGrouping ? `
                            <div class="option-group">
                                <label class="checkbox-label">
                                    <input type="checkbox" id="group-by-family">
                                    Group by family
                                </label>
                            </div>
                            ` : ''}
                        </div>
                    </div>
                </div>
                
                <div class="search-results">
                    <div id="search-status" class="search-status"></div>
                    <div id="results-container" class="results-container"></div>
                    <div id="pagination" class="pagination"></div>
                </div>
                
                <div id="search-history-panel" class="search-history-panel" style="display: none;">
                    <h4>Search History</h4>
                    <div id="history-list" class="history-list"></div>
                </div>
            </div>
        `;
        
        this.attachEventListeners();
        this.loadDocumentFamilies();
    }
    
    renderFilters() {
        return `
            <div class="search-filters">
                <div class="filter-section">
                    <h4>Search Scope</h4>
                    <div class="scope-options">
                        <label class="scope-option">
                            <input type="radio" name="search-scope" value="all" checked>
                            <span>All Versions</span>
                        </label>
                        <label class="scope-option">
                            <input type="radio" name="search-scope" value="latest">
                            <span>Latest Versions Only</span>
                        </label>
                        <label class="scope-option">
                            <input type="radio" name="search-scope" value="specific_versions">
                            <span>Specific Versions</span>
                        </label>
                        <label class="scope-option">
                            <input type="radio" name="search-scope" value="family">
                            <span>Document Families</span>
                        </label>
                        <label class="scope-option">
                            <input type="radio" name="search-scope" value="date_range">
                            <span>Date Range</span>
                        </label>
                    </div>
                </div>
                
                <div class="filter-section">
                    <h4>Filters</h4>
                    <div class="filters-grid">
                        <div id="family-filter" class="filter-group" style="display: none;">
                            <label for="family-selector">Document Families:</label>
                            <select id="family-selector" multiple>
                                <option value="">Loading families...</option>
                            </select>
                        </div>
                        
                        <div id="version-filter" class="filter-group" style="display: none;">
                            <label for="version-range">Version Range:</label>
                            <input type="text" id="version-range" placeholder="e.g., 1-3, latest-2, 5">
                            <small>Examples: "1-3", "latest-2", "5"</small>
                        </div>
                        
                        <div id="date-filter" class="filter-group" style="display: none;">
                            <div class="date-inputs">
                                <div>
                                    <label for="date-from">From:</label>
                                    <input type="date" id="date-from">
                                </div>
                                <div>
                                    <label for="date-to">To:</label>
                                    <input type="date" id="date-to">
                                </div>
                            </div>
                        </div>
                        
                        <div class="filter-group">
                            <label for="snippet-length">Snippet Length:</label>
                            <input type="range" id="snippet-length" min="100" max="500" value="200" step="50">
                            <span class="range-value">200</span>
                        </div>
                        
                        <div class="filter-group">
                            <label for="max-snippets">Max Snippets per Document:</label>
                            <select id="max-snippets">
                                <option value="1">1</option>
                                <option value="3" selected>3</option>
                                <option value="5">5</option>
                                <option value="10">10</option>
                            </select>
                        </div>
                    </div>
                </div>
                
                <div class="filter-actions">
                    <button id="apply-filters" class="btn btn-primary">Apply Filters</button>
                    <button id="clear-filters" class="btn btn-secondary">Clear Filters</button>
                    <button id="toggle-history" class="btn btn-outline">
                        <i class="fas fa-history"></i> History
                    </button>
                </div>
            </div>
        `;
    }
    
    attachEventListeners() {
        // Search functionality
        document.getElementById('search-query').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.performSearch();
            }
        });
        
        document.getElementById('search-btn').addEventListener('click', () => {
            this.performSearch();
        });
        
        // Search scope changes
        document.querySelectorAll('input[name="search-scope"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.updateSearchScope(e.target.value);
            });
        });
        
        // Filter controls
        if (this.options.showFilters) {
            this.attachFilterListeners();
        }
        
        // Options
        document.getElementById('sort-by').addEventListener('change', (e) => {
            this.searchParams.sort_by = e.target.value;
        });
        
        document.getElementById('sort-order').addEventListener('change', (e) => {
            this.searchParams.sort_order = e.target.value;
        });
        
        document.getElementById('results-limit').addEventListener('change', (e) => {
            this.searchParams.limit = parseInt(e.target.value);
        });
        
        if (this.options.showGrouping) {
            document.getElementById('group-by-family').addEventListener('change', (e) => {
                this.searchParams.group_by_family = e.target.checked;
            });
        }
        
        // Action buttons
        document.getElementById('save-search').addEventListener('click', () => {
            this.saveCurrentSearch();
        });
        
        document.getElementById('load-search').addEventListener('click', () => {
            this.showSearchHistory();
        });
        
        document.getElementById('toggle-history').addEventListener('click', () => {
            this.toggleSearchHistory();
        });
    }
    
    attachFilterListeners() {
        // Apply filters
        document.getElementById('apply-filters').addEventListener('click', () => {
            this.applyFilters();
        });
        
        // Clear filters
        document.getElementById('clear-filters').addEventListener('click', () => {
            this.clearFilters();
        });
        
        // Family selector
        document.getElementById('family-selector').addEventListener('change', (e) => {
            const selected = Array.from(e.target.selectedOptions).map(opt => opt.value);
            this.searchParams.family_ids = selected.length > 0 ? selected.join(',') : null;
        });
        
        // Version range
        document.getElementById('version-range').addEventListener('change', (e) => {
            this.searchParams.version_range = e.target.value || null;
        });
        
        // Date filters
        document.getElementById('date-from').addEventListener('change', (e) => {
            this.searchParams.date_from = e.target.value || null;
        });
        
        document.getElementById('date-to').addEventListener('change', (e) => {
            this.searchParams.date_to = e.target.value || null;
        });
        
        // Snippet controls
        document.getElementById('snippet-length').addEventListener('input', (e) => {
            this.searchParams.snippet_length = parseInt(e.target.value);
            document.querySelector('.range-value').textContent = e.target.value;
        });
        
        document.getElementById('max-snippets').addEventListener('change', (e) => {
            this.searchParams.max_snippets_per_doc = parseInt(e.target.value);
        });
    }
    
    async loadDocumentFamilies() {
        try {
            const response = await fetch(`${this.options.apiBasePath}/history/families`);
            if (!response.ok) throw new Error('Failed to load families');
            
            const data = await response.json();
            this.availableFamilies = data.families;
            
            if (this.options.showFilters) {
                this.populateFamilySelector();
            }
            
        } catch (error) {
            console.error('Failed to load document families:', error);
        }
    }
    
    populateFamilySelector() {
        const selector = document.getElementById('family-selector');
        if (!selector) return;
        
        selector.innerHTML = this.availableFamilies.map(family => `
            <option value="${family.family_id}">
                ${family.filenames} (${family.version_count} versions)
            </option>
        `).join('');
    }
    
    updateSearchScope(scope) {
        this.searchParams.search_scope = scope;
        
        // Show/hide relevant filter sections
        const familyFilter = document.getElementById('family-filter');
        const versionFilter = document.getElementById('version-filter');
        const dateFilter = document.getElementById('date-filter');
        
        if (familyFilter && versionFilter && dateFilter) {
            familyFilter.style.display = (scope === 'family' || scope === 'specific_versions') ? 'block' : 'none';
            versionFilter.style.display = scope === 'specific_versions' ? 'block' : 'none';
            dateFilter.style.display = scope === 'date_range' ? 'block' : 'none';
        }
    }
    
    applyFilters() {
        // Filters are applied automatically through event listeners
        this.performSearch();
    }
    
    clearFilters() {
        // Reset all filter values
        this.searchParams.family_ids = null;
        this.searchParams.version_ids = null;
        this.searchParams.date_from = null;
        this.searchParams.date_to = null;
        this.searchParams.version_range = null;
        
        // Reset form elements
        if (document.getElementById('family-selector')) {
            document.getElementById('family-selector').selectedIndex = -1;
        }
        if (document.getElementById('version-range')) {
            document.getElementById('version-range').value = '';
        }
        if (document.getElementById('date-from')) {
            document.getElementById('date-from').value = '';
        }
        if (document.getElementById('date-to')) {
            document.getElementById('date-to').value = '';
        }
        
        // Reset search scope to default
        document.querySelector('input[name="search-scope"][value="all"]').checked = true;
        this.updateSearchScope('all');
    }
    
    async performSearch() {
        const query = document.getElementById('search-query').value.trim();
        if (!query) {
            this.showStatus('Please enter a search query', 'warning');
            return;
        }
        
        this.searchParams.query = query;
        this.searchParams.offset = 0; // Reset pagination
        
        this.showStatus('Searching...', 'info');
        
        try {
            const searchUrl = new URL(`${this.options.apiBasePath}/history/search/advanced`, window.location.origin);
            
            // Add all search parameters
            Object.entries(this.searchParams).forEach(([key, value]) => {
                if (value !== null && value !== undefined && value !== '') {
                    searchUrl.searchParams.append(key, value);
                }
            });
            
            const response = await fetch(searchUrl);
            if (!response.ok) throw new Error('Search failed');
            
            const results = await response.json();
            this.currentResults = results;
            
            this.displayResults(results);
            this.addToSearchHistory(query, results);
            
            if (this.options.onSearchResults) {
                this.options.onSearchResults(results);
            }
            
        } catch (error) {
            console.error('Search error:', error);
            this.showStatus('Search failed. Please try again.', 'error');
        }
    }
    
    displayResults(results) {
        const container = document.getElementById('results-container');
        
        if (results.total_results === 0) {
            this.showStatus('No results found', 'info');
            container.innerHTML = '';
            return;
        }
        
        this.showStatus(`Found ${results.total_results} result(s)`, 'success');
        
        if (results.family_groups) {
            this.displayGroupedResults(results.family_groups, container);
        } else {
            this.displayFlatResults(results.results, container);
        }
        
        this.updatePagination(results);
    }
    
    displayFlatResults(results, container) {
        const resultsHtml = results.map(result => this.renderSearchResult(result)).join('');
        container.innerHTML = `<div class="search-results-list">${resultsHtml}</div>`;
    }
    
    displayGroupedResults(familyGroups, container) {
        const groupsHtml = familyGroups.map(group => `
            <div class="family-group">
                <div class="family-group-header">
                    <h4>${group.family_name}</h4>
                    <div class="family-stats">
                        ${group.total_versions} versions • 
                        ${group.total_snippets} matches • 
                        Avg relevance: ${(group.avg_relevance * 100).toFixed(1)}%
                    </div>
                </div>
                <div class="family-documents">
                    ${group.documents.map(doc => this.renderSearchResult(doc)).join('')}
                </div>
            </div>
        `).join('');
        
        container.innerHTML = `<div class="grouped-results">${groupsHtml}</div>`;
    }
    
    renderSearchResult(result) {
        const snippetsHtml = result.snippets.map(snippet => `
            <div class="result-snippet">
                <div class="snippet-text">${snippet.highlighted_text}</div>
                <div class="snippet-meta">Position: ${snippet.position}</div>
            </div>
        `).join('');
        
        return `
            <div class="search-result-item" data-document-id="${result.document_id}">
                <div class="result-header">
                    <h5 class="result-title">${result.filename}</h5>
                    <div class="result-meta">
                        <span class="version-info">v${result.version_number}</span>
                        ${result.is_latest_version ? '<span class="latest-badge">Latest</span>' : ''}
                        <span class="relevance-score">${(result.relevance_score * 100).toFixed(1)}% relevant</span>
                        <span class="upload-date">${new Date(result.upload_date).toLocaleDateString()}</span>
                    </div>
                </div>
                
                ${snippetsHtml ? `
                <div class="result-snippets">
                    ${snippetsHtml}
                </div>
                ` : ''}
                
                <div class="result-actions">
                    <button class="btn-sm btn-primary" onclick="advancedSearch.selectDocument('${result.document_id}')">
                        View Document
                    </button>
                    ${result.document_family_id ? `
                    <button class="btn-sm btn-outline" onclick="advancedSearch.viewFamily('${result.document_family_id}')">
                        View Family
                    </button>
                    ` : ''}
                </div>
            </div>
        `;
    }
    
    updatePagination(results) {
        const pagination = document.getElementById('pagination');
        const totalPages = Math.ceil(results.total_results / this.searchParams.limit);
        const currentPage = Math.floor(this.searchParams.offset / this.searchParams.limit) + 1;
        
        if (totalPages <= 1) {
            pagination.innerHTML = '';
            return;
        }
        
        let paginationHtml = '<div class="pagination-controls">';
        
        // Previous button
        if (currentPage > 1) {
            paginationHtml += `<button class="page-btn" onclick="advancedSearch.goToPage(${currentPage - 1})">Previous</button>`;
        }
        
        // Page numbers
        for (let i = Math.max(1, currentPage - 2); i <= Math.min(totalPages, currentPage + 2); i++) {
            const activeClass = i === currentPage ? 'active' : '';
            paginationHtml += `<button class="page-btn ${activeClass}" onclick="advancedSearch.goToPage(${i})">${i}</button>`;
        }
        
        // Next button
        if (currentPage < totalPages) {
            paginationHtml += `<button class="page-btn" onclick="advancedSearch.goToPage(${currentPage + 1})">Next</button>`;
        }
        
        paginationHtml += '</div>';
        pagination.innerHTML = paginationHtml;
    }
    
    goToPage(page) {
        this.searchParams.offset = (page - 1) * this.searchParams.limit;
        this.performSearch();
    }
    
    showStatus(message, type = 'info') {
        const statusEl = document.getElementById('search-status');
        statusEl.innerHTML = `<div class="status-message status-${type}">${message}</div>`;
    }
    
    addToSearchHistory(query, results) {
        const historyItem = {
            query: query,
            params: { ...this.searchParams },
            timestamp: new Date().toISOString(),
            resultCount: results.total_results
        };
        
        this.searchHistory.unshift(historyItem);
        this.searchHistory = this.searchHistory.slice(0, 20); // Keep last 20 searches
        
        // Save to localStorage
        localStorage.setItem('advancedSearchHistory', JSON.stringify(this.searchHistory));
    }
    
    loadSearchHistory() {
        const saved = localStorage.getItem('advancedSearchHistory');
        if (saved) {
            try {
                this.searchHistory = JSON.parse(saved);
            } catch (error) {
                console.error('Failed to load search history:', error);
            }
        }
    }
    
    toggleSearchHistory() {
        const panel = document.getElementById('search-history-panel');
        const isVisible = panel.style.display !== 'none';
        
        panel.style.display = isVisible ? 'none' : 'block';
        
        if (!isVisible) {
            this.renderSearchHistory();
        }
    }
    
    renderSearchHistory() {
        const historyList = document.getElementById('history-list');
        
        if (this.searchHistory.length === 0) {
            historyList.innerHTML = '<p class="no-history">No search history available</p>';
            return;
        }
        
        const historyHtml = this.searchHistory.map((item, index) => `
            <div class="history-item" onclick="advancedSearch.loadSearchFromHistory(${index})">
                <div class="history-query">${item.query}</div>
                <div class="history-meta">
                    ${item.resultCount} results • 
                    ${new Date(item.timestamp).toLocaleDateString()} • 
                    Scope: ${item.params.search_scope}
                </div>
            </div>
        `).join('');
        
        historyList.innerHTML = historyHtml;
    }
    
    loadSearchFromHistory(index) {
        const historyItem = this.searchHistory[index];
        if (!historyItem) return;
        
        // Restore search parameters
        this.searchParams = { ...historyItem.params };
        
        // Update form fields
        document.getElementById('search-query').value = historyItem.query;
        document.querySelector(`input[name="search-scope"][value="${historyItem.params.search_scope}"]`).checked = true;
        
        // Update other form fields as needed
        this.updateSearchScope(historyItem.params.search_scope);
        
        // Perform the search
        this.performSearch();
        
        // Hide history panel
        document.getElementById('search-history-panel').style.display = 'none';
    }
    
    saveCurrentSearch() {
        const searchName = prompt('Enter a name for this search:');
        if (!searchName) return;
        
        const savedSearch = {
            name: searchName,
            query: this.searchParams.query,
            params: { ...this.searchParams },
            timestamp: new Date().toISOString()
        };
        
        // Save to localStorage (could be enhanced to save to server)
        const savedSearches = JSON.parse(localStorage.getItem('savedAdvancedSearches') || '[]');
        savedSearches.push(savedSearch);
        localStorage.setItem('savedAdvancedSearches', JSON.stringify(savedSearches));
        
        alert('Search saved successfully!');
    }
    
    selectDocument(documentId) {
        if (this.options.onDocumentSelect) {
            this.options.onDocumentSelect(documentId);
        } else {
            // Default behavior
            window.open(`/document/${documentId}`, '_blank');
        }
    }
    
    viewFamily(familyId) {
        // Default behavior - could be customized
        window.open(`/family/${familyId}`, '_blank');
    }
    
    // Public methods
    getSearchParams() {
        return { ...this.searchParams };
    }
    
    setSearchParams(params) {
        this.searchParams = { ...this.searchParams, ...params };
    }
    
    getCurrentResults() {
        return this.currentResults;
    }
    
    clearResults() {
        document.getElementById('results-container').innerHTML = '';
        document.getElementById('search-status').innerHTML = '';
        document.getElementById('pagination').innerHTML = '';
        this.currentResults = null;
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AdvancedSearch;
}
