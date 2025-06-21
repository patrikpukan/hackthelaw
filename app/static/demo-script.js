// Legal RAG Agent Frontend - Demo Version
class LegalRAGAppDemo {
    constructor() {
        this.selectedDocument = null;
        this.currentSessionId = 'demo-session-123';
        this.llmProvider = 'vertexai';
        this.demoMode = true;
        
        this.initializeElements();
        this.attachEventListeners();
        this.loadDemoDocuments();
        this.setDemoStatus();
        this.showDemoWelcome();
        this.loadDemoLLMProviders();
    }

    initializeElements() {
        // Document elements
        this.uploadZone = document.getElementById('uploadZone');
        this.fileInput = document.getElementById('fileInput');
        this.uploadBtn = document.getElementById('uploadBtn');
        this.documentsList = document.getElementById('documentsList');

        // Chat elements
        this.chatMessages = document.getElementById('chatMessages');
        this.messageInput = document.getElementById('messageInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.llmSelector = document.getElementById('llmProvider');

        // UI elements
        this.loadingOverlay = document.getElementById('loadingOverlay');
        this.loadingText = document.getElementById('loadingText');
        this.status = document.getElementById('status');
        this.toastContainer = document.getElementById('toastContainer');
    }

    attachEventListeners() {
        // File upload (demo)
        this.uploadZone.addEventListener('click', () => this.showDemoUpload());
        this.uploadBtn.addEventListener('click', () => this.demoUploadDocument());

        // Chat
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // LLM provider selection
        this.llmSelector.addEventListener('change', (e) => {
            this.llmProvider = e.target.value;
            this.showToast(`Switched to ${e.target.options[e.target.selectedIndex].text}`, 'info');
        });

        // Quick questions
        document.querySelectorAll('.quick-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const question = e.currentTarget.dataset.question;
                this.messageInput.value = question;
                this.sendMessage();
            });
        });
    }

    setDemoStatus() {
        this.updateStatus('DEMO MODE', 'demo');
    }

    updateStatus(message, type) {
        const statusElement = this.status.querySelector('span');
        const dotElement = this.status.querySelector('.status-dot');

        statusElement.textContent = message;

        if (type === 'demo') {
            dotElement.style.color = '#ffd700';
        } else if (type === 'online') {
            dotElement.style.color = '#48bb78';
        } else {
            dotElement.style.color = '#f56565';
        }
    }

    // Load demo LLM providers
    loadDemoLLMProviders() {
        const demoProviders = [
            { id: 'groq', name: 'Groq (DEMO)', available: true },
            { id: 'openai', name: 'OpenAI (DEMO)', available: true },
            { id: 'vertexai', name: 'Vertex AI (DEMO)', available: true },
            { id: 'mock', name: 'Mock (DEMO)', available: true }
        ];

        // Clear existing options
        this.llmSelector.innerHTML = '';

        // Add demo providers
        demoProviders.forEach(provider => {
            const option = document.createElement('option');
            option.value = provider.id;
            option.textContent = provider.name;
            this.llmSelector.appendChild(option);
        });

        // Set default to vertexai
        this.llmSelector.value = 'vertexai';
        this.llmProvider = 'vertexai';
    }

    showDemoWelcome() {
        setTimeout(() => {
            this.showToast('🎭 DEMO MODE: API simulation enabled!', 'info');
        }, 1000);
    }

    showDemoUpload() {
        this.showToast('📁 In demo mode use ready documents below', 'info');
    }

    demoUploadDocument() {
        this.showLoading('Simulating upload...');
        
        setTimeout(() => {
            this.hideLoading();
            this.showToast('✅ Demo document "added"!', 'success');
            this.loadDemoDocuments();
        }, 2000);
    }

    loadDemoDocuments() {
        const demoDocuments = [
            {
                id: 'demo-1',
                filename: 'Employment_Contract_TechInnovations.txt',
                upload_date: new Date().toISOString(),
                file_size: 15600,
                processing_status: 'processed'
            },
            {
                id: 'demo-2', 
                filename: 'Office_Lease_Agreement.pdf',
                upload_date: new Date(Date.now() - 86400000).toISOString(),
                file_size: 890000,
                processing_status: 'processed'
            },
            {
                id: 'demo-3',
                filename: 'NDA_Agreement.docx',
                upload_date: new Date(Date.now() - 172800000).toISOString(),
                file_size: 45000,
                processing_status: 'processing'
            }
        ];
        
        this.renderDocuments(demoDocuments);
    }

    renderDocuments(documents) {
        const container = this.documentsList;
        
        container.innerHTML = documents.map(doc => `
            <div class="document-item" data-id="${doc.id}" onclick="demoApp.selectDocument('${doc.id}', '${doc.filename}')">
                <div class="document-name">${doc.filename}</div>
                <div class="document-meta">
                    ${new Date(doc.upload_date).toLocaleDateString('en-US')} • ${this.formatFileSize(doc.file_size)}
                    <span class="document-status status-${doc.processing_status}">
                        ${this.getStatusText(doc.processing_status)}
                    </span>
                </div>
            </div>
        `).join('');

        // Enable chat
        this.enableChat();
    }

    getStatusText(status) {
        const statusMap = {
            'processed': 'Processed',
            'processing': 'Processing',
            'pending': 'Pending'
        };
        return statusMap[status] || status;
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    selectDocument(documentId, filename) {
        // Remove previous selection
        document.querySelectorAll('.document-item').forEach(item => {
            item.classList.remove('selected');
        });

        // Add selection to current item
        document.querySelector(`[data-id="${documentId}"]`).classList.add('selected');

        this.selectedDocument = { id: documentId, filename };
        this.showToast(`Selected document: ${filename}`, 'info');
    }

    enableChat() {
        this.messageInput.disabled = false;
        this.sendBtn.disabled = false;
        this.messageInput.placeholder = "Ask a question about the document (DEMO mode)...";
        
        document.querySelectorAll('.quick-btn').forEach(btn => {
            btn.disabled = false;
        });
    }

    // Demo Chat Responses
    getDemoResponse(message) {
        const responses = {
            'analyze the main terms of this contract': `
**Analysis of Main Employment Contract Terms:**

🏢 **Employer:** TechInnovations LLC
👤 **Employee:** John Smith  
📍 **Position:** Senior Software Developer

💰 **Compensation:**
- Base salary: $120,000/year
- Bonus: up to 50% of salary per quarter
- Payment: 10th and 25th of each month

⏰ **Working hours:**
- 40 hours per week
- Vacation: 28 days + 7 days for overtime

🔐 **Confidentiality:**
- 3-year obligation after termination
- Prohibition on using proprietary information

💼 **Additional benefits:**
- Health insurance: $25,000/year
- Meal allowance: $1,200/month
- Company car
            `,
            'what are the risks in this document?': `
**Identified risks in the employment contract:**

⚠️ **High risks:**
- Short termination notice period (14 days)
- Long confidentiality period (3 years)
- Lack of clear bonus criteria

🟡 **Medium risks:**
- Overtime hours not limited
- Compensation only for "no cause" termination
- Company car without usage restrictions

💡 **Recommendations:**
1. Clarify bonus calculation criteria
2. Reduce confidentiality period to 1-2 years
3. Detail overtime conditions
4. Add company car usage terms
            `,
            'explain the termination conditions': `
**Employment contract termination conditions:**

📋 **General conditions:**
- Termination by either party's initiative
- 14-day notice period (short term!)

💰 **Termination compensation:**
- Employer termination WITHOUT cause: 2 months' salary
- Employer termination WITH cause: standard payments
- Employee resignation: standard payments

⚖️ **Legal assessment:**
- 14 days - less than standard month notice
- 2-salary compensation - above minimum requirement
- No definition of "justified reasons"

🔴 **Risks:**
- Short notice may complicate job search
- Unclear criteria for "justified reasons"
- Possible disputes over compensation amount
            `,
            'what salary is specified in the contract?': `
**Salary information:**

💰 **Base salary:** $120,000 per year

🎯 **Additional payments:**
- Bonus: up to 50% of salary (up to $60,000)
- Bonus frequency: quarterly

📅 **Payment schedule:**
- 10th of month - partial payment
- 25th of month - remaining salary

🧮 **Potential income calculation:**
- Monthly minimum: $10,000
- Monthly maximum: $15,000 (with bonus)
- Annual income: $120,000 - $180,000

💼 **Additional compensation:**
- Meal allowance: $1,200/month
- Health insurance: $25,000/year
- Company car (transportation savings)
            `
        };

        // Find most suitable response
        const lowerMessage = message.toLowerCase();
        for (const [key, response] of Object.entries(responses)) {
            if (lowerMessage.includes(key.toLowerCase()) || 
                key.toLowerCase().includes(lowerMessage)) {
                return response;
            }
        }

        // Default response
        return `
**Demo response to query: "${message}"**

🤖 In demo mode I can answer the following questions:

1. "Analyze the main terms of this contract"
2. "What are the risks in this document?"  
3. "Explain the termination conditions"
4. "What salary is specified in the contract?"

💡 **Try one of the suggested questions or use the quick buttons above!**

🔗 **For full functionality connect real API:**
- Groq API key: configure in environment variables
- Run: \`docker-compose -f docker-compose.frontend.yml up\`
        `;
    }

    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message) return;

        if (!this.selectedDocument) {
            this.showToast('Please select a document first', 'error');
            return;
        }

        // Add user message
        this.addMessage(message, 'user');
        this.messageInput.value = '';

        // Show loading
        const loadingId = this.addLoadingMessage();
        
        // Simulate API delay
        setTimeout(() => {
            this.removeLoadingMessage(loadingId);
            
            const response = this.getDemoResponse(message);
            const metadata = {
                model_used: `${this.llmProvider.toUpperCase()} (DEMO)`,
                usage: { total_tokens: Math.floor(Math.random() * 500) + 100 }
            };
            
            this.addMessage(response, 'bot', [], metadata);
        }, 1500 + Math.random() * 1000);
    }

    addMessage(content, type, sources = [], metadata = null, isError = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message`;
        
        const time = new Date().toLocaleTimeString('en-US', { 
            hour: '2-digit', 
            minute: '2-digit' 
        });

        let sourcesHtml = '';
        if (sources && sources.length > 0) {
            sourcesHtml = `
                <div class="message-sources">
                    <h4><i class="fas fa-link"></i> Sources:</h4>
                    ${sources.map(source => `<div class="source-item">${source.title || source.text}</div>`).join('')}
                </div>
            `;
        }

        let metadataHtml = '';
        if (metadata && metadata.model_used) {
            metadataHtml = `
                <div class="message-time">
                    ${time} • ${metadata.model_used}
                    ${metadata.usage ? ` • ${metadata.usage.total_tokens} tokens` : ''}
                </div>
            `;
        } else {
            metadataHtml = `<div class="message-time">${time}</div>`;
        }

        messageDiv.innerHTML = `
            <div class="message-avatar">
                <i class="fas fa-${type === 'user' ? 'user' : 'robot'}"></i>
            </div>
            <div class="message-content ${isError ? 'error' : ''}">
                <p style="white-space: pre-line;">${content}</p>
                ${sourcesHtml}
                ${metadataHtml}
            </div>
        `;

        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
    }

    addLoadingMessage() {
        const loadingId = Date.now();
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message bot-message';
        messageDiv.id = `loading-${loadingId}`;
        
        messageDiv.innerHTML = `
            <div class="message-avatar">
                <i class="fas fa-robot"></i>
            </div>
            <div class="message-content">
                <div class="spinner" style="width: 20px; height: 20px; margin: 0;"></div>
                <p style="margin-left: 30px; margin-top: -20px;">Analyzing document... (DEMO)</p>
            </div>
        `;

        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
        
        return loadingId;
    }

    removeLoadingMessage(loadingId) {
        const loadingElement = document.getElementById(`loading-${loadingId}`);
        if (loadingElement) {
            loadingElement.remove();
        }
    }

    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }

    showLoading(text = 'Loading...') {
        this.loadingText.textContent = text;
        this.loadingOverlay.classList.add('show');
    }

    hideLoading() {
        this.loadingOverlay.classList.remove('show');
    }

    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        const icon = {
            'success': 'check-circle',
            'error': 'exclamation-circle',
            'info': 'info-circle'
        }[type] || 'info-circle';

        toast.innerHTML = `
            <i class="fas fa-${icon}"></i>
            <span>${message}</span>
        `;

        this.toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.style.animation = 'slideInRight 0.3s ease reverse';
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    }
}

// Initialize demo app
document.addEventListener('DOMContentLoaded', () => {
    window.demoApp = new LegalRAGAppDemo();
}); 