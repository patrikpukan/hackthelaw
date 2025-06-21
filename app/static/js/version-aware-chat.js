/**
 * Version-Aware Chat Component
 * Chat interface with document version awareness and context
 */
class VersionAwareChat {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        this.options = {
            apiBasePath: '/api/v1',
            sessionId: null,
            defaultVersionMode: 'latest',
            showVersionControls: true,
            maxMessages: 100,
            ...options
        };
        
        this.sessionId = this.options.sessionId || this.generateSessionId();
        this.messages = [];
        this.versionMode = this.options.defaultVersionMode;
        this.selectedFamilies = [];
        this.specificVersions = [];
        this.includeVersionContext = true;
        this.isTyping = false;
        
        this.init();
    }
    
    init() {
        this.container.innerHTML = `
            <div class="version-aware-chat">
                ${this.options.showVersionControls ? this.renderVersionControls() : ''}
                
                <div class="chat-container">
                    <div id="chat-messages" class="chat-messages"></div>
                    
                    <div class="chat-input-container">
                        <div class="input-group">
                            <textarea id="chat-input" class="chat-input" 
                                    placeholder="Ask a question about your documents..." 
                                    rows="2"></textarea>
                            <button id="send-btn" class="send-btn">
                                <i class="fas fa-paper-plane"></i>
                            </button>
                        </div>
                        
                        <div class="chat-options">
                            <label class="option-label">
                                <input type="checkbox" id="iterative-rag" checked>
                                Use advanced analysis
                            </label>
                            <label class="option-label">
                                <input type="checkbox" id="include-context" ${this.includeVersionContext ? 'checked' : ''}>
                                Include version context
                            </label>
                            <span class="version-indicator">
                                Mode: <strong id="current-mode">${this.versionMode}</strong>
                            </span>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        this.attachEventListeners();
        this.loadChatHistory();
    }
    
    renderVersionControls() {
        return `
            <div class="version-controls">
                <div class="controls-header">
                    <h4>Document Version Settings</h4>
                    <button id="toggle-controls" class="toggle-btn">
                        <i class="fas fa-chevron-up"></i>
                    </button>
                </div>
                
                <div id="controls-content" class="controls-content">
                    <div class="control-group">
                        <label for="version-mode">Search Mode:</label>
                        <select id="version-mode">
                            <option value="latest">Latest Versions Only</option>
                            <option value="all">All Versions</option>
                            <option value="specific">Specific Versions</option>
                            <option value="family">Document Families</option>
                        </select>
                    </div>
                    
                    <div id="family-selector-group" class="control-group" style="display: none;">
                        <label for="family-selector">Select Families:</label>
                        <select id="family-selector" multiple>
                            <option value="">Loading families...</option>
                        </select>
                    </div>
                    
                    <div id="version-selector-group" class="control-group" style="display: none;">
                        <label for="version-selector">Select Versions:</label>
                        <select id="version-selector" multiple>
                            <option value="">Select families first...</option>
                        </select>
                    </div>
                    
                    <div class="control-actions">
                        <button id="apply-settings" class="btn btn-primary">Apply Settings</button>
                        <button id="reset-settings" class="btn btn-secondary">Reset</button>
                    </div>
                </div>
            </div>
        `;
    }
    
    attachEventListeners() {
        // Chat input
        const chatInput = document.getElementById('chat-input');
        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Send button
        document.getElementById('send-btn').addEventListener('click', () => {
            this.sendMessage();
        });
        
        // Version context toggle
        document.getElementById('include-context').addEventListener('change', (e) => {
            this.includeVersionContext = e.target.checked;
        });
        
        if (this.options.showVersionControls) {
            this.attachVersionControlListeners();
        }
    }
    
    attachVersionControlListeners() {
        // Toggle controls
        document.getElementById('toggle-controls').addEventListener('click', () => {
            this.toggleVersionControls();
        });
        
        // Version mode change
        document.getElementById('version-mode').addEventListener('change', (e) => {
            this.updateVersionMode(e.target.value);
        });
        
        // Family selector
        document.getElementById('family-selector').addEventListener('change', (e) => {
            this.selectedFamilies = Array.from(e.target.selectedOptions).map(opt => opt.value);
            this.loadVersionsForFamilies();
        });
        
        // Version selector
        document.getElementById('version-selector').addEventListener('change', (e) => {
            this.specificVersions = Array.from(e.target.selectedOptions).map(opt => opt.value);
        });
        
        // Apply settings
        document.getElementById('apply-settings').addEventListener('click', () => {
            this.applyVersionSettings();
        });
        
        // Reset settings
        document.getElementById('reset-settings').addEventListener('click', () => {
            this.resetVersionSettings();
        });
        
        this.loadDocumentFamilies();
    }
    
    async loadDocumentFamilies() {
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
    
    async loadVersionsForFamilies() {
        if (this.selectedFamilies.length === 0) {
            document.getElementById('version-selector').innerHTML = '<option value="">Select families first...</option>';
            return;
        }
        
        try {
            const versionSelector = document.getElementById('version-selector');
            versionSelector.innerHTML = '<option value="">Loading versions...</option>';
            
            // Load versions for each selected family
            const allVersions = [];
            for (const familyId of this.selectedFamilies) {
                const response = await fetch(`${this.options.apiBasePath}/documents/family/${familyId}/versions`);
                if (response.ok) {
                    const data = await response.json();
                    allVersions.push(...data.versions);
                }
            }
            
            versionSelector.innerHTML = allVersions.map(version => `
                <option value="${version.id}">
                    ${version.filename} v${version.version_number}
                    ${version.is_latest_version ? ' (Latest)' : ''}
                </option>
            `).join('');
            
        } catch (error) {
            console.error('Failed to load versions:', error);
            document.getElementById('version-selector').innerHTML = '<option value="">Error loading versions</option>';
        }
    }
    
    updateVersionMode(mode) {
        this.versionMode = mode;
        document.getElementById('current-mode').textContent = mode;
        
        // Show/hide relevant controls
        const familyGroup = document.getElementById('family-selector-group');
        const versionGroup = document.getElementById('version-selector-group');
        
        familyGroup.style.display = (mode === 'family' || mode === 'specific') ? 'block' : 'none';
        versionGroup.style.display = mode === 'specific' ? 'block' : 'none';
    }
    
    applyVersionSettings() {
        // Update the current mode indicator
        document.getElementById('current-mode').textContent = this.versionMode;
        
        // Show confirmation
        this.showSystemMessage(`Version settings applied: ${this.versionMode} mode`);
    }
    
    resetVersionSettings() {
        this.versionMode = this.options.defaultVersionMode;
        this.selectedFamilies = [];
        this.specificVersions = [];
        
        document.getElementById('version-mode').value = this.versionMode;
        document.getElementById('family-selector').selectedIndex = -1;
        document.getElementById('version-selector').selectedIndex = -1;
        
        this.updateVersionMode(this.versionMode);
        this.showSystemMessage('Version settings reset to defaults');
    }
    
    toggleVersionControls() {
        const content = document.getElementById('controls-content');
        const toggleBtn = document.getElementById('toggle-controls');
        const isVisible = content.style.display !== 'none';
        
        content.style.display = isVisible ? 'none' : 'block';
        toggleBtn.innerHTML = isVisible ? 
            '<i class="fas fa-chevron-down"></i>' : 
            '<i class="fas fa-chevron-up"></i>';
    }
    
    async sendMessage() {
        const input = document.getElementById('chat-input');
        const message = input.value.trim();
        
        if (!message || this.isTyping) return;
        
        // Add user message to chat
        this.addMessage('user', message);
        input.value = '';
        
        // Show typing indicator
        this.showTypingIndicator();
        
        try {
            const response = await this.callChatAPI(message);
            this.hideTypingIndicator();
            
            // Add assistant response
            this.addMessage('assistant', response.response, {
                sources: response.sources,
                metadata: response.metadata
            });
            
        } catch (error) {
            this.hideTypingIndicator();
            this.addMessage('system', 'Sorry, I encountered an error processing your request. Please try again.');
            console.error('Chat error:', error);
        }
    }
    
    async callChatAPI(message) {
        const requestData = {
            message: message,
            session_id: this.sessionId,
            use_iterative_rag: document.getElementById('iterative-rag').checked,
            version_mode: this.versionMode,
            include_version_context: this.includeVersionContext
        };
        
        // Add version-specific parameters
        if (this.versionMode === 'family' && this.selectedFamilies.length > 0) {
            requestData.family_ids = this.selectedFamilies;
        } else if (this.versionMode === 'specific' && this.specificVersions.length > 0) {
            requestData.specific_versions = this.specificVersions;
        }
        
        const response = await fetch(`${this.options.apiBasePath}/chat/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        return await response.json();
    }
    
    addMessage(type, content, metadata = {}) {
        const message = {
            id: Date.now(),
            type: type,
            content: content,
            timestamp: new Date(),
            metadata: metadata
        };
        
        this.messages.push(message);
        this.renderMessage(message);
        this.scrollToBottom();
        
        // Limit message history
        if (this.messages.length > this.options.maxMessages) {
            this.messages = this.messages.slice(-this.options.maxMessages);
            this.rerenderMessages();
        }
    }
    
    renderMessage(message) {
        const messagesContainer = document.getElementById('chat-messages');
        const messageEl = document.createElement('div');
        messageEl.className = `message message-${message.type}`;
        messageEl.innerHTML = this.formatMessage(message);
        
        messagesContainer.appendChild(messageEl);
    }
    
    formatMessage(message) {
        let html = `
            <div class="message-content">
                ${this.formatMessageContent(message.content)}
            </div>
            <div class="message-timestamp">
                ${message.timestamp.toLocaleTimeString()}
            </div>
        `;
        
        if (message.type === 'assistant' && message.metadata) {
            html += this.formatMessageMetadata(message.metadata);
        }
        
        return html;
    }
    
    formatMessageContent(content) {
        // Convert markdown-like formatting to HTML
        return content
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/\n/g, '<br>');
    }
    
    formatMessageMetadata(metadata) {
        let html = '<div class="message-metadata">';
        
        // Version information
        if (metadata.version_info) {
            const versionInfo = metadata.version_info;
            html += `
                <div class="version-context">
                    <strong>Version Context:</strong>
                    ${versionInfo.documents_resolved} document(s) searched in ${versionInfo.version_mode} mode
                    ${versionInfo.version_context && versionInfo.version_context.documents_used ? 
                        `<br><small>Documents: ${versionInfo.version_context.documents_used.map(d => `${d.filename} v${d.version_number}`).join(', ')}</small>` : ''}
                </div>
            `;
        }
        
        // Sources
        if (metadata.sources && metadata.sources.length > 0) {
            html += `
                <div class="sources">
                    <strong>Sources:</strong>
                    <ul>
                        ${metadata.sources.map(source => `
                            <li>
                                ${source.document_name || 'Document'} 
                                ${source.chunk_index ? `(chunk ${source.chunk_index})` : ''}
                                <span class="relevance">Relevance: ${(source.relevance_score * 100).toFixed(1)}%</span>
                            </li>
                        `).join('')}
                    </ul>
                </div>
            `;
        }
        
        html += '</div>';
        return html;
    }
    
    showTypingIndicator() {
        this.isTyping = true;
        const indicator = document.createElement('div');
        indicator.id = 'typing-indicator';
        indicator.className = 'message message-system typing';
        indicator.innerHTML = `
            <div class="message-content">
                <div class="typing-dots">
                    <span></span><span></span><span></span>
                </div>
                AI is thinking...
            </div>
        `;
        
        document.getElementById('chat-messages').appendChild(indicator);
        this.scrollToBottom();
    }
    
    hideTypingIndicator() {
        this.isTyping = false;
        const indicator = document.getElementById('typing-indicator');
        if (indicator) {
            indicator.remove();
        }
    }
    
    showSystemMessage(message) {
        this.addMessage('system', message);
    }
    
    scrollToBottom() {
        const messagesContainer = document.getElementById('chat-messages');
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
    
    rerenderMessages() {
        const messagesContainer = document.getElementById('chat-messages');
        messagesContainer.innerHTML = '';
        this.messages.forEach(message => this.renderMessage(message));
    }
    
    loadChatHistory() {
        // Load chat history from localStorage or API
        const savedHistory = localStorage.getItem(`chat_history_${this.sessionId}`);
        if (savedHistory) {
            try {
                this.messages = JSON.parse(savedHistory).map(msg => ({
                    ...msg,
                    timestamp: new Date(msg.timestamp)
                }));
                this.rerenderMessages();
            } catch (error) {
                console.error('Failed to load chat history:', error);
            }
        }
    }
    
    saveChatHistory() {
        localStorage.setItem(`chat_history_${this.sessionId}`, JSON.stringify(this.messages));
    }
    
    generateSessionId() {
        return 'chat_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
    
    // Public methods
    clearChat() {
        this.messages = [];
        document.getElementById('chat-messages').innerHTML = '';
        localStorage.removeItem(`chat_history_${this.sessionId}`);
    }
    
    getSessionId() {
        return this.sessionId;
    }
    
    getMessages() {
        return this.messages;
    }
    
    setVersionMode(mode) {
        this.versionMode = mode;
        if (this.options.showVersionControls) {
            document.getElementById('version-mode').value = mode;
            this.updateVersionMode(mode);
        }
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = VersionAwareChat;
}
