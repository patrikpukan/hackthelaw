// Legal RAG Agent Frontend
class LegalRAGApp {
  constructor() {
    this.apiBase = "http://localhost:8000";
    this.selectedDocuments = new Set();
    this.currentSessionId = null;
    this.llmProvider = "vertexai"; // Will be updated from API
    this.searchAllMode = false;

    // Detailed analysis mode settings
    this.detailedAnalysisMode = false;
    this.enableContradictionDetection = true;
    this.enableTemporalTracking = true;
    this.enableCrossDocumentReasoning = true;
    this.maxAnalysisTokens = 300000;

    // Pagination and filtering
    this.currentPage = 1;
    this.pageSize = 25;
    this.totalDocuments = 0;
    this.filteredDocuments = [];
    this.allDocuments = [];
    this.searchQuery = "";
    this.statusFilter = "";
    this.typeFilter = "";
    this.sortBy = "date_desc";

    // Selection state
    this.lastSelectedIndex = -1;

    this.initializeElements();
    this.attachEventListeners();
    this.loadDocuments();
    this.checkAPIStatus();
    this.loadLLMProviders();
  }

  initializeElements() {
    // Document elements
    this.uploadZone = document.getElementById("uploadZone");
    this.fileInput = document.getElementById("fileInput");
    this.uploadBtn = document.getElementById("uploadBtn");
    this.importWebBtn = document.getElementById("importWebBtn");
    this.documentsList = document.getElementById("documentsList");
    this.documentCounter = document.getElementById("documentCounter");

    // Upload progress elements
    this.uploadProgress = document.getElementById("uploadProgress");
    this.progressFill = document.getElementById("progressFill");
    this.progressText = document.getElementById("progressText");

    // Search and filter elements
    this.documentSearch = document.getElementById("documentSearch");
    this.clearSearch = document.getElementById("clearSearch");
    this.statusFilterEl = document.getElementById("statusFilter");
    this.typeFilterEl = document.getElementById("typeFilter");
    this.sortByEl = document.getElementById("sortBy");

    // Selection elements
    this.selectAllBtn = document.getElementById("selectAllBtn");
    this.clearSelectionBtn = document.getElementById("clearSelectionBtn");
    this.bulkActions = document.getElementById("bulkActions");
    this.deleteSelected = document.getElementById("deleteSelected");
    this.versionManagementBtn = document.getElementById("versionManagementBtn");

    // Pagination elements
    this.pagination = document.getElementById("pagination");
    this.prevPage = document.getElementById("prevPage");
    this.nextPage = document.getElementById("nextPage");
    this.pageInfo = document.getElementById("pageInfo");
    this.pageSizeEl = document.getElementById("pageSize");

    // Update versions button
    this.updateVersionsBtn = document.getElementById("updateVersionsBtn");

    // Chat elements
    this.chatMessages = document.getElementById("chatMessages");
    this.messageInput = document.getElementById("messageInput");
    this.sendBtn = document.getElementById("sendBtn");
    this.llmSelector = document.getElementById("llmProvider");

    // Analysis mode elements
    this.detailedAnalysisModeCheckbox = document.getElementById(
      "detailedAnalysisMode"
    );
    this.analysisConfig = document.getElementById("analysisConfig");
    this.enableContradictionDetectionCheckbox = document.getElementById(
      "enableContradictionDetection"
    );
    this.enableTemporalTrackingCheckbox = document.getElementById(
      "enableTemporalTracking"
    );
    this.enableCrossDocumentReasoningCheckbox = document.getElementById(
      "enableCrossDocumentReasoning"
    );
    this.maxAnalysisTokensInput = document.getElementById("maxAnalysisTokens");

    // UI elements
    this.loadingOverlay = document.getElementById("loadingOverlay");
    this.loadingText = document.getElementById("loadingText");
    this.status = document.getElementById("status");
    this.toastContainer = document.getElementById("toastContainer");

    // Prolong duration modal elements
    this.prolongDurationBtn = document.getElementById("prolongDurationBtn");
    this.prolongDurationModal = document.getElementById("prolongDurationModal");
    this.prolongDurationModalClose = document.getElementById(
      "prolongDurationModalClose"
    );
    this.prolongDatePicker = document.getElementById("prolongDatePicker");
    this.prolongCancelBtn = document.getElementById("prolongCancelBtn");
    this.prolongConfirmBtn = document.getElementById("prolongConfirmBtn");

    // Find client modal elements
    this.findClientBtn = document.getElementById("findClientBtn");
    this.findClientModal = document.getElementById("findClientModal");
    this.findClientModalClose = document.getElementById("findClientModalClose");
    this.clientNameInput = document.getElementById("clientNameInput");
    this.clientSearchCancelBtn = document.getElementById(
      "clientSearchCancelBtn"
    );
    this.clientSearchBtn = document.getElementById("clientSearchBtn");
  }

  attachEventListeners() {
    // File upload
    this.uploadZone.addEventListener("click", () => this.fileInput.click());
    this.uploadZone.addEventListener("dragover", (e) => this.handleDragOver(e));
    this.uploadZone.addEventListener("drop", (e) => this.handleDrop(e));
    this.fileInput.addEventListener("change", (e) => this.handleFileSelect(e));
    this.uploadBtn.addEventListener("click", () => this.uploadDocuments());
    this.importWebBtn.addEventListener("click", () =>
      this.showImportWebModal()
    );

    // Search and filter
    this.documentSearch.addEventListener("input", (e) =>
      this.handleSearch(e.target.value)
    );
    this.clearSearch.addEventListener("click", () => this.clearSearchInput());
    this.statusFilterEl.addEventListener("change", (e) =>
      this.handleFilterChange("status", e.target.value)
    );
    this.typeFilterEl.addEventListener("change", (e) =>
      this.handleFilterChange("type", e.target.value)
    );
    this.sortByEl.addEventListener("change", (e) =>
      this.handleSortChange(e.target.value)
    );

    // Selection
    this.selectAllBtn.addEventListener("click", () =>
      this.selectAllDocuments()
    );
    this.clearSelectionBtn.addEventListener("click", () =>
      this.clearSelection()
    );
    this.deleteSelected.addEventListener("click", () =>
      this.deleteSelectedDocuments()
    );
    this.versionManagementBtn.addEventListener("click", () =>
      this.openVersionManagement()
    );

    // Pagination
    this.prevPage.addEventListener("click", () =>
      this.changePage(this.currentPage - 1)
    );
    this.nextPage.addEventListener("click", () =>
      this.changePage(this.currentPage + 1)
    );
    this.pageSizeEl.addEventListener("change", (e) =>
      this.changePageSize(parseInt(e.target.value))
    );

    // Update versions
    this.updateVersionsBtn.addEventListener("click", () =>
      this.showUpdateVersionsModal()
    );

    // Chat
    this.sendBtn.addEventListener("click", () => this.sendMessage());
    this.messageInput.addEventListener("keypress", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        this.sendMessage();
      }
    });

    // LLM provider selection
    this.llmSelector.addEventListener("change", (e) => {
      this.llmProvider = e.target.value;
      this.showToast(
        `Switched to ${e.target.options[e.target.selectedIndex].text}`,
        "info"
      );
    });

    // Analysis mode controls
    this.detailedAnalysisModeCheckbox.addEventListener("change", (e) => {
      this.detailedAnalysisMode = e.target.checked;
      this.toggleAnalysisConfig(e.target.checked);
      this.updateChatPlaceholder();

      if (e.target.checked) {
        this.showToast(
          "Detailed analysis mode enabled - responses will be more comprehensive but slower",
          "info"
        );
      } else {
        this.showToast("Switched to fast mode", "info");
      }
    });

    this.enableContradictionDetectionCheckbox.addEventListener(
      "change",
      (e) => {
        this.enableContradictionDetection = e.target.checked;
      }
    );

    this.enableTemporalTrackingCheckbox.addEventListener("change", (e) => {
      this.enableTemporalTracking = e.target.checked;
    });

    this.enableCrossDocumentReasoningCheckbox.addEventListener(
      "change",
      (e) => {
        this.enableCrossDocumentReasoning = e.target.checked;
      }
    );

    this.maxAnalysisTokensInput.addEventListener("change", (e) => {
      this.maxAnalysisTokens = parseInt(e.target.value);
    });

    // Quick questions
    document.querySelectorAll(".quick-btn").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        // Special handling for prolong duration button
        if (e.currentTarget.id === "prolongDurationBtn") {
          this.showProlongDurationModal();
          return;
        }

        // Special handling for find client button
        if (e.currentTarget.id === "findClientBtn") {
          this.showFindClientModal();
          return;
        }

        const question = e.currentTarget.dataset.question;
        if (question) {
          // Extract button text (remove icon and trim)
          const buttonText = e.currentTarget.textContent.trim();

          // Special handling for check due date button - inject today's date
          let finalQuestion = question;
          if (e.currentTarget.id === "checkDueDateBtn") {
            const today = new Date().toLocaleDateString("en-US", {
              year: "numeric",
              month: "long",
              day: "numeric",
            });
            finalQuestion = question.replace("{todayDate}", today);
          }

          // Send the full question to LLM but display only button text in chat
          this.sendMessage(finalQuestion, buttonText);
        }
      });
    });

    // Prolong duration modal event listeners
    this.prolongDurationBtn.addEventListener("click", () =>
      this.showProlongDurationModal()
    );
    this.prolongDurationModalClose.addEventListener("click", () =>
      this.hideProlongDurationModal()
    );
    this.prolongCancelBtn.addEventListener("click", () =>
      this.hideProlongDurationModal()
    );
    this.prolongConfirmBtn.addEventListener("click", () =>
      this.confirmProlongDuration()
    );

    // Date picker change event
    this.prolongDatePicker.addEventListener("change", (e) => {
      this.prolongConfirmBtn.disabled = !e.target.value;
    });

    // Close modal when clicking outside
    this.prolongDurationModal.addEventListener("click", (e) => {
      if (e.target === this.prolongDurationModal) {
        this.hideProlongDurationModal();
      }
    });

    // Find client modal event listeners
    this.findClientModalClose.addEventListener("click", () =>
      this.hideFindClientModal()
    );
    this.clientSearchCancelBtn.addEventListener("click", () =>
      this.hideFindClientModal()
    );
    this.clientSearchBtn.addEventListener("click", () =>
      this.searchForClient()
    );

    // Client name input change event
    this.clientNameInput.addEventListener("input", (e) => {
      this.clientSearchBtn.disabled = !e.target.value.trim();
    });

    // Close modal when clicking outside
    this.findClientModal.addEventListener("click", (e) => {
      if (e.target === this.findClientModal) {
        this.hideFindClientModal();
      }
    });

    // Enter key support for client search
    this.clientNameInput.addEventListener("keypress", (e) => {
      if (e.key === "Enter" && !this.clientSearchBtn.disabled) {
        this.searchForClient();
      }
    });
  }

  // API Status Check
  async checkAPIStatus() {
    try {
      const response = await fetch(`${this.apiBase}/health`);
      if (response.ok) {
        this.updateStatus("Ready to work", "online");
      } else {
        this.updateStatus("Server issues", "offline");
      }
    } catch (error) {
      this.updateStatus("Server unavailable", "offline");
    }
  }

  updateStatus(message, type) {
    // Since we now use a logo instead of status text, we can optionally
    // update the image opacity or add a visual indicator for status
    const logoElement = this.status.querySelector("img");

    if (logoElement) {
      if (type === "online") {
        logoElement.style.opacity = "1";
        logoElement.style.filter = "none";
      } else {
        logoElement.style.opacity = "0.5";
        logoElement.style.filter = "grayscale(100%)";
      }
    }
  }

  toggleAnalysisConfig(show) {
    if (show) {
      this.analysisConfig.style.display = "block";
    } else {
      this.analysisConfig.style.display = "none";
    }
  }

  updateChatPlaceholder() {
    if (this.messageInput) {
      const baseText =
        this.selectedDocuments.size > 0
          ? `Ask a question about ${this.selectedDocuments.size} selected document(s)`
          : "Ask a question about the documents";

      if (this.detailedAnalysisMode) {
        this.messageInput.placeholder = `${baseText} (Detailed Analysis Mode)...`;
      } else {
        this.messageInput.placeholder = `${baseText}...`;
      }
    }
  }

  // Load available LLM providers
  async loadLLMProviders() {
    try {
      const response = await fetch(`${this.apiBase}/api/v1/chat/providers`);
      if (response.ok) {
        const data = await response.json();
        this.updateLLMSelector(data.providers, data.default);
      } else {
        console.warn("Failed to load LLM providers, using defaults");
      }
    } catch (error) {
      console.warn("Error loading LLM providers:", error);
    }
  }

  updateLLMSelector(providers, defaultProvider) {
    // Clear existing options
    this.llmSelector.innerHTML = "";

    // Add available providers
    providers.forEach((provider) => {
      const option = document.createElement("option");
      option.value = provider.id;

      if (provider.available) {
        option.textContent = provider.name;
        option.title = provider.description;
      } else {
        option.textContent = `${provider.name} (Not configured)`;
        option.title = `${provider.description} - ${provider.reason}`;
        option.disabled = true;
      }

      // Set as selected if it's the default and available
      if (provider.id === defaultProvider && provider.available) {
        option.selected = true;
        this.llmProvider = provider.id;
      }

      this.llmSelector.appendChild(option);
    });

    // If no provider is selected, select the first available one
    if (!this.llmSelector.value) {
      const firstAvailable = providers.find((p) => p.available);
      if (firstAvailable) {
        this.llmSelector.value = firstAvailable.id;
        this.llmProvider = firstAvailable.id;

        // Show informative message about fallback if default was not available
        const defaultProvider = providers.find((p) => p.id === defaultProvider);
        if (defaultProvider && !defaultProvider.available) {
          this.showToast(
            `Default provider (${defaultProvider.name}) not available: ${defaultProvider.reason}. Using ${firstAvailable.name} instead.`,
            "warning"
          );
        }
      } else {
        this.showToast(
          "No LLM providers available. Please configure API keys.",
          "error"
        );
      }
    }
  }

  // File Upload Handlers
  handleDragOver(e) {
    e.preventDefault();
    this.uploadZone.classList.add("dragover");
  }

  handleDrop(e) {
    e.preventDefault();
    this.uploadZone.classList.remove("dragover");

    const files = e.dataTransfer.files;
    if (files.length > 0) {
      this.fileInput.files = files;
      this.handleFileSelect({ target: { files } });
    }
  }

  handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
      this.uploadBtn.disabled = false;
      const fileText =
        files.length === 1
          ? `File selected: <strong>${
              files[0].name
            }</strong> (${this.formatFileSize(files[0].size)})`
          : `<strong>${
              files.length
            } files</strong> selected (${this.formatTotalFileSize(files)})`;
      this.uploadZone.querySelector("p").innerHTML = fileText;
    }
  }

  formatFileSize(bytes) {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  }

  formatTotalFileSize(files) {
    let totalSize = 0;
    for (let file of files) {
      totalSize += file.size;
    }
    return this.formatFileSize(totalSize);
  }

  async uploadDocuments() {
    const files = this.fileInput.files;
    if (!files || files.length === 0) return;

    // Show upload progress
    this.showUploadProgress();
    this.uploadBtn.disabled = true;

    let completed = 0;
    const total = files.length;

    try {
      const uploadPromises = [];
      for (let file of files) {
        const formData = new FormData();
        formData.append("file", file);

        const uploadPromise = this.uploadSingleDocument(formData, file.name)
          .then((result) => {
            completed++;
            this.updateUploadProgress(completed, total);
            return result;
          })
          .catch((error) => {
            completed++;
            this.updateUploadProgress(completed, total);
            throw error;
          });

        uploadPromises.push(uploadPromise);
      }

      const results = await Promise.allSettled(uploadPromises);
      const successful = results.filter((r) => r.status === "fulfilled").length;
      const failed = results.filter((r) => r.status === "rejected").length;

      if (successful > 0) {
        this.showToast(
          `Successfully uploaded ${successful} document(s)!`,
          "success"
        );
        this.resetUploadForm();
        this.loadDocuments();
      }

      if (failed > 0) {
        this.showToast(`Failed to upload ${failed} document(s)`, "error");
      }
    } catch (error) {
      this.showToast(`Upload error: ${error.message}`, "error");
    } finally {
      this.hideUploadProgress();
      this.uploadBtn.disabled = false;
    }
  }

  async uploadSingleDocument(formData, filename) {
    const response = await fetch(`${this.apiBase}/api/v1/documents/upload`, {
      method: "POST",
      body: formData,
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error(`${filename}: ${result.detail || "Upload error"}`);
    }

    return result;
  }

  resetUploadForm() {
    this.fileInput.value = "";
    this.uploadBtn.disabled = true;
    this.uploadZone.querySelector("p").innerHTML =
      'Drag documents here or <span class="upload-text">choose files</span>';
  }

  showUploadProgress() {
    this.uploadProgress.style.display = "block";
    this.progressFill.style.width = "0%";
    this.progressText.textContent = "Starting upload...";
  }

  updateUploadProgress(completed, total) {
    const percentage = Math.round((completed / total) * 100);
    this.progressFill.style.width = `${percentage}%`;
    this.progressText.textContent = `Uploading documents... ${completed}/${total} (${percentage}%)`;
  }

  hideUploadProgress() {
    this.uploadProgress.style.display = "none";
  }

  // Import from Web Modal
  showImportWebModal() {
    // Create modal
    const modal = document.createElement("div");
    modal.className = "modal";
    modal.innerHTML = `
      <div class="modal-content">
        <div class="modal-header">
          <h3>Import from Web</h3>
          <button class="modal-close">&times;</button>
        </div>
        <div class="modal-body">
          <p>Newest documents already present</p>
        </div>
      </div>
    `;

    // Add modal to page
    document.body.appendChild(modal);

    // Close modal functionality
    const closeBtn = modal.querySelector(".modal-close");
    closeBtn.addEventListener("click", () => {
      document.body.removeChild(modal);
    });

    // Close on outside click
    modal.addEventListener("click", (e) => {
      if (e.target === modal) {
        document.body.removeChild(modal);
      }
    });
  }

  // Update Versions Modal
  showUpdateVersionsModal() {
    // Create modal
    const modal = document.createElement("div");
    modal.className = "modal";
    modal.innerHTML = `
      <div class="modal-content">
        <div class="modal-header">
          <h3>Update Versions</h3>
          <button class="modal-close">&times;</button>
        </div>
        <div class="modal-body">
          <p>Newest document versions already present</p>
        </div>
      </div>
    `;

    // Add modal to page
    document.body.appendChild(modal);

    // Close modal functionality
    const closeBtn = modal.querySelector(".modal-close");
    closeBtn.addEventListener("click", () => {
      document.body.removeChild(modal);
    });

    // Close on outside click
    modal.addEventListener("click", (e) => {
      if (e.target === modal) {
        document.body.removeChild(modal);
      }
    });
  }

  // Search and Filter Functions
  handleSearch(query) {
    this.searchQuery = query.toLowerCase();
    this.clearSearch.style.display = query ? "block" : "none";
    this.currentPage = 1;
    this.applyFiltersAndPagination();
  }

  clearSearchInput() {
    this.documentSearch.value = "";
    this.searchQuery = "";
    this.clearSearch.style.display = "none";
    this.currentPage = 1;
    this.applyFiltersAndPagination();
  }

  handleFilterChange(type, value) {
    if (type === "status") {
      this.statusFilter = value;
    } else if (type === "type") {
      this.typeFilter = value;
    }
    this.currentPage = 1;
    this.applyFiltersAndPagination();
  }

  handleSortChange(sortBy) {
    this.sortBy = sortBy;
    this.applyFiltersAndPagination();
  }

  applyFiltersAndPagination() {
    // Filter documents
    this.filteredDocuments = this.allDocuments.filter((doc) => {
      // Search filter
      if (
        this.searchQuery &&
        !doc.filename.toLowerCase().includes(this.searchQuery)
      ) {
        return false;
      }

      // Status filter
      if (this.statusFilter && doc.processing_status !== this.statusFilter) {
        return false;
      }

      // Type filter
      if (this.typeFilter) {
        const extension = doc.filename.split(".").pop().toLowerCase();
        if (extension !== this.typeFilter) {
          return false;
        }
      }

      return true;
    });

    // Sort documents
    this.filteredDocuments.sort((a, b) => {
      switch (this.sortBy) {
        case "date_desc":
          return new Date(b.uploaded_at) - new Date(a.uploaded_at);
        case "date_asc":
          return new Date(a.uploaded_at) - new Date(b.uploaded_at);
        case "name_asc":
          return a.filename.localeCompare(b.filename);
        case "name_desc":
          return b.filename.localeCompare(a.filename);
        case "size_desc":
          return (b.file_size || 0) - (a.file_size || 0);
        case "size_asc":
          return (a.file_size || 0) - (b.file_size || 0);
        default:
          return 0;
      }
    });

    this.totalDocuments = this.filteredDocuments.length;
    this.renderDocuments();
    this.updatePagination();
  }

  // Pagination Functions
  changePage(page) {
    const maxPage = Math.ceil(this.totalDocuments / this.pageSize);
    if (page >= 1 && page <= maxPage) {
      this.currentPage = page;
      this.renderDocuments();
      this.updatePagination();
    }
  }

  changePageSize(newSize) {
    this.pageSize = newSize;
    this.currentPage = 1;
    this.renderDocuments();
    this.updatePagination();
  }

  updatePagination() {
    const maxPage = Math.ceil(this.totalDocuments / this.pageSize);

    this.prevPage.disabled = this.currentPage <= 1;
    this.nextPage.disabled = this.currentPage >= maxPage;

    this.pageInfo.textContent = `Page ${this.currentPage} of ${maxPage}`;

    this.pagination.style.display = maxPage > 1 ? "flex" : "none";
  }

  // Documents Management
  async loadDocuments() {
    try {
      // Load all documents (no pagination on API level for now)
      const response = await fetch(
        `${this.apiBase}/api/v1/documents/?limit=1000`
      );
      const data = await response.json();

      if (response.ok) {
        this.allDocuments = data.documents || [];
        this.applyFiltersAndPagination();
        this.updateDocumentCounter();

        // Check if there are documents still processing
        const hasProcessingDocs = this.allDocuments.some(
          (doc) =>
            doc.processing_status === "processing" ||
            doc.processing_status === "pending"
        );

        if (hasProcessingDocs) {
          // Auto-refresh every 3 seconds if documents are still processing
          setTimeout(() => this.loadDocuments(), 3000);
        }

        // Auto-enable chat if documents are available
        if (this.allDocuments.length > 0) {
          setTimeout(() => {
            if (this.selectedDocuments.size === 0) {
              // Select first document if none selected
              this.selectedDocuments.add(this.allDocuments[0].id);
              this.updateSelectionDisplay();
            }
            this.enableChat();
          }, 500);
        }
      } else {
        throw new Error(
          `API Error: ${response.status} - ${data.detail || "Unknown error"}`
        );
      }
    } catch (error) {
      console.error("Error loading documents:", error);
      this.showToast(`Error loading documents: ${error.message}`, "error");
    }
  }

  updateDocumentCounter() {
    const total = this.allDocuments.length;
    const text = total === 1 ? "1 document" : `${total} documents`;
    this.documentCounter.querySelector("span").textContent = text;
  }

  renderDocuments() {
    const startIndex = (this.currentPage - 1) * this.pageSize;
    const endIndex = startIndex + this.pageSize;
    const documentsToShow = this.filteredDocuments.slice(startIndex, endIndex);

    if (documentsToShow.length === 0) {
      this.documentsList.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-folder-open"></i>
                    <p>${
                      this.allDocuments.length === 0
                        ? "No documents uploaded"
                        : "No documents match your filters"
                    }</p>
                </div>
            `;
      return;
    }

    this.documentsList.innerHTML = documentsToShow
      .map((doc, index) => {
        const actualIndex = startIndex + index;
        const isSelected = this.selectedDocuments.has(doc.id);

        return `
                <div class="document-item ${isSelected ? "selected" : ""}"
                     data-document-id="${doc.id}"
                     data-index="${actualIndex}">
                    <input type="checkbox" ${isSelected ? "checked" : ""}
                           onclick="event.stopPropagation()">
                    <div class="document-content">
                        <div class="document-name">${doc.filename}</div>
                        <div class="document-meta">
                            <div>
                                <span class="document-status ${this.getStatusClass(
                                  doc.processing_status
                                )}">
                                    ${this.getStatusText(doc.processing_status)}
                                </span>
                                ${
                                  doc.file_size
                                    ? `• ${this.formatFileSize(doc.file_size)}`
                                    : ""
                                }
                            </div>
                            <div>
                                Uploaded: ${new Date(
                                  doc.uploaded_at
                                ).toLocaleString()}
                                ${
                                  doc.chunk_count
                                    ? `• ${doc.chunk_count} chunks`
                                    : ""
                                }
                            </div>
                        </div>
                    </div>
                    <div class="document-actions">
                        <button class="btn-delete" onclick="app.deleteDocument('${
                          doc.id
                        }', '${doc.filename}')">
                            <i class="fas fa-trash"></i>
                        </button>
                    </div>
                </div>
            `;
      })
      .join("");

    // Add click event listeners to document items
    this.documentsList.querySelectorAll(".document-item").forEach((item) => {
      item.addEventListener("click", (e) => this.handleDocumentClick(e));

      const checkbox = item.querySelector('input[type="checkbox"]');
      checkbox.addEventListener("change", (e) => this.handleCheckboxChange(e));
    });
  }

  handleDocumentClick(e) {
    // Don't handle click if it's on a button or checkbox
    if (
      e.target.tagName === "BUTTON" ||
      e.target.tagName === "INPUT" ||
      e.target.closest(".btn-delete") ||
      e.target.closest('input[type="checkbox"]')
    ) {
      return;
    }

    const item = e.currentTarget;
    const documentId = item.dataset.documentId;
    const index = parseInt(item.dataset.index);
    const isCtrlPressed = e.ctrlKey || e.metaKey;
    const isShiftPressed = e.shiftKey;

    if (isShiftPressed && this.lastSelectedIndex !== -1) {
      // Range selection
      const start = Math.min(this.lastSelectedIndex, index);
      const end = Math.max(this.lastSelectedIndex, index);

      for (let i = start; i <= end; i++) {
        if (i < this.filteredDocuments.length) {
          this.selectedDocuments.add(this.filteredDocuments[i].id);
        }
      }
    } else if (isCtrlPressed) {
      // Toggle selection (allows multiple documents to be selected)
      if (this.selectedDocuments.has(documentId)) {
        this.selectedDocuments.delete(documentId);
      } else {
        this.selectedDocuments.add(documentId);
      }
    } else {
      // For normal clicks, toggle selection (changed from single selection to toggle)
      if (
        this.selectedDocuments.has(documentId) &&
        this.selectedDocuments.size === 1
      ) {
        // If only this document is selected, deselect it
        this.selectedDocuments.delete(documentId);
      } else {
        // Otherwise, toggle this document's selection
        if (this.selectedDocuments.has(documentId)) {
          this.selectedDocuments.delete(documentId);
        } else {
          this.selectedDocuments.add(documentId);
        }
      }
    }

    this.lastSelectedIndex = index;
    this.updateSelectionDisplay();
  }

  handleCheckboxChange(e) {
    const item = e.target.closest(".document-item");
    const documentId = item.dataset.documentId;

    if (e.target.checked) {
      this.selectedDocuments.add(documentId);
    } else {
      this.selectedDocuments.delete(documentId);
    }

    this.updateSelectionDisplay();
  }

  updateSelectionDisplay() {
    // Update visual selection
    this.documentsList.querySelectorAll(".document-item").forEach((item) => {
      const documentId = item.dataset.documentId;
      const checkbox = item.querySelector('input[type="checkbox"]');
      const isSelected = this.selectedDocuments.has(documentId);

      item.classList.toggle("selected", isSelected);
      checkbox.checked = isSelected;
    });

    // Update selection controls
    const selectedCount = this.selectedDocuments.size;
    const totalVisibleCount = this.filteredDocuments.slice(
      (this.currentPage - 1) * this.pageSize,
      this.currentPage * this.pageSize
    ).length;

    this.bulkActions.style.display = selectedCount > 0 ? "flex" : "none";
    this.clearSelectionBtn.style.display =
      selectedCount > 0 ? "inline-flex" : "none";

    // Update select all button text
    const documentsOnPage = this.filteredDocuments.slice(
      (this.currentPage - 1) * this.pageSize,
      this.currentPage * this.pageSize
    );
    const allOnPageSelected =
      documentsOnPage.length > 0 &&
      documentsOnPage.every((doc) => this.selectedDocuments.has(doc.id));

    if (allOnPageSelected) {
      this.selectAllBtn.innerHTML =
        '<i class="fas fa-square"></i> Deselect All';
    } else if (selectedCount === 0) {
      this.selectAllBtn.innerHTML =
        '<i class="fas fa-check-square"></i> Select All';
    } else {
      this.selectAllBtn.innerHTML = `<i class="fas fa-check-square"></i> Select All (${selectedCount} selected)`;
    }

    // Update selection info with count
    const selectionInfo = document.querySelector(".selection-info span");
    if (selectedCount > 0) {
      selectionInfo.textContent = `${selectedCount} document${
        selectedCount === 1 ? "" : "s"
      } selected • Click to select/deselect • Hold Ctrl/Cmd for multi-select • Shift for range`;
    } else {
      selectionInfo.textContent =
        "Click to select/deselect • Hold Ctrl/Cmd for multi-select • Shift for range";
    }

    // Update chat placeholder
    this.enableChat();
    this.updateChatPlaceholder();
  }

  selectAllDocuments() {
    const documentsOnPage = this.filteredDocuments.slice(
      (this.currentPage - 1) * this.pageSize,
      this.currentPage * this.pageSize
    );

    // Check if all documents on the current page are selected
    const allSelected = documentsOnPage.every((doc) =>
      this.selectedDocuments.has(doc.id)
    );

    if (allSelected && documentsOnPage.length > 0) {
      // If all are selected, deselect all on this page
      documentsOnPage.forEach((doc) => {
        this.selectedDocuments.delete(doc.id);
      });
    } else {
      // Otherwise, select all on this page
      documentsOnPage.forEach((doc) => {
        this.selectedDocuments.add(doc.id);
      });
    }

    this.updateSelectionDisplay();
  }

  clearSelection() {
    this.selectedDocuments.clear();
    this.updateSelectionDisplay();
  }

  async deleteSelectedDocuments() {
    if (this.selectedDocuments.size === 0) return;

    const confirmed = confirm(
      `Are you sure you want to delete ${this.selectedDocuments.size} document(s)?`
    );
    if (!confirmed) return;

    this.showLoading("Deleting documents...");

    try {
      const deletePromises = Array.from(this.selectedDocuments).map((docId) =>
        fetch(`${this.apiBase}/api/v1/documents/${docId}`, { method: "DELETE" })
      );

      await Promise.all(deletePromises);

      this.showToast(
        `Deleted ${this.selectedDocuments.size} document(s)`,
        "success"
      );
      this.selectedDocuments.clear();
      this.loadDocuments();
    } catch (error) {
      this.showToast("Error deleting documents", "error");
    } finally {
      this.hideLoading();
    }
  }

  openVersionManagement() {
    // Navigate to the version management page
    window.open(
      `${this.apiBase}/api/v1/documents/version-management`,
      "_blank"
    );
  }

  getStatusClass(status) {
    switch (status) {
      case "completed":
      case "ready":
        return "status-completed";
      case "processing":
        return "status-processing";
      case "pending":
        return "status-pending";
      case "failed":
        return "status-failed";
      default:
        return "status-pending";
    }
  }

  getStatusText(status) {
    switch (status) {
      case "completed":
        return "Ready";
      case "processing":
        return "Processing...";
      case "pending":
        return "Pending";
      case "failed":
        return "Failed";
      default:
        return "Unknown";
    }
  }

  async deleteDocument(documentId, filename) {
    const confirmed = confirm(`Are you sure you want to delete "${filename}"?`);
    if (!confirmed) return;

    this.showLoading("Deleting document...");

    try {
      const response = await fetch(
        `${this.apiBase}/api/v1/documents/${documentId}`,
        {
          method: "DELETE",
        }
      );

      if (response.ok) {
        this.showToast("Document deleted successfully!", "success");
        this.selectedDocuments.delete(documentId);
        this.loadDocuments();
      } else {
        const error = await response.json();
        throw new Error(error.detail || "Delete error");
      }
    } catch (error) {
      this.showToast(`Delete error: ${error.message}`, "error");
    } finally {
      this.hideLoading();
    }
  }

  enableChat() {
    const hasDocuments = this.allDocuments.length > 0;
    const hasSelected = this.selectedDocuments.size > 0 || this.searchAllMode;

    this.messageInput.disabled = !hasDocuments;
    this.sendBtn.disabled = !hasDocuments;

    if (hasDocuments) {
      this.updateChatPlaceholder();
    }
  }

  async sendMessage(apiMessage = null, displayMessage = null) {
    // Use provided messages or fall back to input value
    const message = apiMessage || this.messageInput.value.trim();
    const chatDisplayMessage = displayMessage || message;
    if (!message) return;

    // Determine which documents to search
    let documentIds = [];
    if (this.selectedDocuments.size > 0) {
      documentIds = Array.from(this.selectedDocuments);
    } else {
      // If no documents selected, use all completed documents
      documentIds = this.allDocuments
        .filter((doc) => doc.processing_status === "completed")
        .map((doc) => doc.id);
    }

    if (documentIds.length === 0) {
      this.showToast("No documents available for search", "error");
      return;
    }

    // Add user message (use display message for chat, API message for backend)
    this.addMessage(chatDisplayMessage, "user");
    this.messageInput.value = "";

    // Add loading message
    const loadingId = this.addLoadingMessage();

    try {
      // Choose endpoint based on analysis mode
      const endpoint = this.detailedAnalysisMode
        ? "/api/v1/chat/query"
        : "/api/v1/chat/iterative";

      const requestBody = {
        message: message,
        document_ids: documentIds,
        llm_provider: this.llmProvider,
        session_id: this.currentSessionId,
      };

      // Add detailed analysis parameters if enabled
      if (this.detailedAnalysisMode) {
        requestBody.detailed_analysis_mode = true;
        requestBody.enable_contradiction_detection =
          this.enableContradictionDetection;
        requestBody.enable_temporal_tracking = this.enableTemporalTracking;
        requestBody.enable_cross_document_reasoning =
          this.enableCrossDocumentReasoning;
        requestBody.max_analysis_tokens = this.maxAnalysisTokens;
        requestBody.use_iterative_rag = false; // Use multi-stage instead
      }

      const response = await fetch(`${this.apiBase}${endpoint}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      });

      const data = await response.json();

      if (response.ok) {
        this.currentSessionId = data.session_id;

        // Remove loading message
        this.removeLoadingMessage(loadingId);

        // Add bot response with enhanced analysis if available
        this.addMessage(
          data.response,
          "bot",
          data.sources || [],
          data.metadata,
          false,
          data
        );

        // Show success toast with stats
        if (data.metadata) {
          const stats = data.metadata.search_stats || {};
          this.showToast(
            `Response generated using ${
              stats.total_chunks || "unknown"
            } chunks from ${documentIds.length} document(s)`,
            "info"
          );
        }
      } else {
        throw new Error(data.detail || "Chat error");
      }
    } catch (error) {
      this.removeLoadingMessage(loadingId);
      this.addMessage(`Error: ${error.message}`, "bot", [], null, true);
      this.showToast(`Chat error: ${error.message}`, "error");
    }
  }

  addMessage(
    content,
    type,
    sources = [],
    metadata = null,
    isError = false,
    analysisData = null
  ) {
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${type}-message`;

    const timestamp = new Date().toLocaleTimeString();
    const avatarIcon = type === "user" ? "fas fa-user" : "fas fa-robot";

    let sourcesHtml = "";
    if (sources && sources.length > 0) {
      sourcesHtml = `
                <div class="message-sources">
                    <h4><i class="fas fa-link"></i> Sources (${
                      sources.length
                    })</h4>
                    ${sources
                      .map((source) => this.formatSource(source))
                      .join("")}
                </div>
            `;
    }

    // Format metadata if available
    let metadataHtml = "";
    if (metadata && metadata.search_stats) {
      const stats = metadata.search_stats;
      metadataHtml = `
                <div class="search-metadata">
                    <small>
                        <i class="fas fa-chart-bar"></i>
                        Search: ${stats.total_chunks || 0} chunks from ${
        stats.documents_searched || 0
      } documents
                        ${
                          stats.iterations_used
                            ? `• ${stats.iterations_used} iterations`
                            : ""
                        }
                        ${
                          stats.search_time
                            ? `• ${(stats.search_time * 1000).toFixed(0)}ms`
                            : ""
                        }
                    </small>
                </div>
            `;
    }

    // Format enhanced analysis results if available
    let analysisHtml = "";
    if (
      analysisData &&
      (analysisData.legal_analysis ||
        analysisData.contradictions ||
        analysisData.processing_stages)
    ) {
      analysisHtml = this.formatLegalAnalysisResults(analysisData);
    }

    messageDiv.innerHTML = `
            <div class="message-avatar">
                <i class="${avatarIcon}"></i>
            </div>
            <div class="message-content">
                <div class="markdown-content">${
                  type === "user"
                    ? this.escapeHtml(content)
                    : marked.parse(content)
                }</div>
                ${sourcesHtml}
                ${analysisHtml}
                ${metadataHtml}
                <div class="message-time">${timestamp}</div>
            </div>
        `;

    if (isError) {
      messageDiv.classList.add("error-message");
    }

    this.chatMessages.appendChild(messageDiv);
    this.scrollToBottom();

    // Highlight code blocks
    messageDiv.querySelectorAll("pre code").forEach((block) => {
      hljs.highlightElement(block);
    });
  }

  formatSource(source) {
    // Handle both old and new source formats
    const doc = source.document || {};
    const chunk = source.chunk || {};

    // Get document name from various possible fields
    const documentName =
      source.document_name ||
      doc.filename ||
      doc.name ||
      doc.title ||
      "Unknown Document";

    const relevanceScore = Math.round((source.similarity_score || 0) * 100);

    // Get chunk content from various possible fields
    const chunkContent =
      source.chunk_preview || chunk.content || chunk.text || "";

    const chunkType = source.chunk_type || chunk.chunk_type || "content";
    const pageNumber = source.page_number || chunk.page_number;
    const sectionTitle = source.section_title || chunk.section_title;

    return `
            <div class="source-item">
                <div class="source-header">
                    <div class="source-doc">
                        <i class="fas fa-file-text"></i>
                        <span>${documentName}</span>
                    </div>
                    <div class="source-relevance">${relevanceScore}% match</div>
                </div>
                <div class="source-meta">
                    <span class="source-type">${chunkType}</span>
                    ${pageNumber ? `Page ${pageNumber}` : ""}
                    ${sectionTitle ? `• ${sectionTitle}` : ""}
                </div>
                ${
                  chunkContent
                    ? `<div class="source-preview">${this.truncateText(
                        chunkContent,
                        200
                      )}</div>`
                    : ""
                }
            </div>
        `;
  }

  truncateText(text, maxLength) {
    if (text.length <= maxLength) return this.escapeHtml(text);
    return this.escapeHtml(text.substring(0, maxLength)) + "...";
  }

  escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  addLoadingMessage() {
    const loadingId = "loading-" + Date.now();
    const messageDiv = document.createElement("div");
    messageDiv.className = "message bot-message loading-message";
    messageDiv.id = loadingId;

    messageDiv.innerHTML = `
            <div class="message-avatar">
                <i class="fas fa-robot"></i>
            </div>
            <div class="message-content">
                <div class="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
                <div class="message-time">Thinking...</div>
            </div>
        `;

    this.chatMessages.appendChild(messageDiv);
    this.scrollToBottom();

    return loadingId;
  }

  removeLoadingMessage(loadingId) {
    const loadingMessage = document.getElementById(loadingId);
    if (loadingMessage) {
      loadingMessage.remove();
    }
  }

  scrollToBottom() {
    this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
  }

  showLoading(text = "Loading...") {
    this.loadingText.textContent = text;
    this.loadingOverlay.classList.add("show");
  }

  hideLoading() {
    this.loadingOverlay.classList.remove("show");
  }

  formatLegalAnalysisResults(analysisData) {
    let html = '<div class="legal-analysis-results">';

    // Analysis mode indicator
    if (analysisData.rag_approach === "multi_stage") {
      html += `
                <div class="analysis-mode-indicator">
                    <i class="fas fa-brain"></i>
                    Multi-Stage Legal Analysis
                </div>
            `;
    }

    // Risk Assessment
    if (
      analysisData.legal_analysis &&
      analysisData.legal_analysis.risk_assessment
    ) {
      const risk = analysisData.legal_analysis.risk_assessment;
      const riskLevel = risk.overall_risk_level || "unknown";
      const riskClass = `risk-${riskLevel}`;

      html += `
                <div class="analysis-section">
                    <div class="analysis-header">
                        <i class="fas fa-shield-alt"></i>
                        Risk Assessment
                    </div>
                    <div class="risk-assessment ${riskClass}">
                        <i class="fas fa-exclamation-triangle"></i>
                        <strong>${riskLevel.toUpperCase()} RISK</strong>
                        ${
                          risk.risk_score
                            ? `(Score: ${(risk.risk_score * 100).toFixed(0)}%)`
                            : ""
                        }
                    </div>
                    ${
                      risk.risk_factors && risk.risk_factors.length > 0
                        ? `
                        <div style="margin-top: 0.5rem;">
                            <small><strong>Risk Factors:</strong> ${risk.risk_factors.join(
                              ", "
                            )}</small>
                        </div>
                    `
                        : ""
                    }
                </div>
            `;
    }

    // Contradictions
    if (analysisData.contradictions && analysisData.contradictions.length > 0) {
      html += `
                <div class="analysis-section">
                    <div class="analysis-header">
                        <i class="fas fa-exclamation-triangle"></i>
                        Contradictions Found (${analysisData.contradictions.length})
                    </div>
                    <div class="contradictions-list">
            `;

      analysisData.contradictions.slice(0, 3).forEach((contradiction) => {
        const severityClass = `severity-${contradiction.severity || "minor"}`;
        html += `
                    <div class="contradiction-item">
                        <div class="contradiction-header">
                            <span class="contradiction-type">${
                              contradiction.type || "conflict"
                            }</span>
                            <span class="contradiction-severity ${severityClass}">${
          contradiction.severity || "minor"
        }</span>
                        </div>
                        <div class="contradiction-description">${this.escapeHtml(
                          contradiction.description ||
                            "No description available"
                        )}</div>
                        ${
                          contradiction.confidence
                            ? `<div class="contradiction-confidence">Confidence: ${(
                                contradiction.confidence * 100
                              ).toFixed(0)}%</div>`
                            : ""
                        }
                    </div>
                `;
      });

      if (analysisData.contradictions.length > 3) {
        html += `<div style="text-align: center; margin-top: 0.5rem;"><small>... and ${
          analysisData.contradictions.length - 3
        } more</small></div>`;
      }

      html += "</div></div>";
    }

    // Processing Stages
    if (
      analysisData.processing_stages &&
      analysisData.processing_stages.length > 0
    ) {
      html += `
                <div class="analysis-section">
                    <div class="analysis-header">
                        <i class="fas fa-cogs"></i>
                        Processing Pipeline
                    </div>
                    <div class="processing-stages">
            `;

      analysisData.processing_stages.forEach((stage) => {
        const statusClass = stage.success ? "stage-success" : "stage-error";
        const statusIcon = stage.success ? "fas fa-check" : "fas fa-times";

        html += `
                    <div class="stage-item">
                        <div class="stage-status ${statusClass}">
                            <i class="${statusIcon}"></i>
                        </div>
                        <div class="stage-info">
                            <div class="stage-name">${this.formatStageName(
                              stage.stage_name
                            )}</div>
                            <div class="stage-details">
                                ${
                                  stage.duration_seconds
                                    ? `${stage.duration_seconds.toFixed(1)}s`
                                    : ""
                                }
                                ${
                                  stage.token_count
                                    ? `• ${stage.token_count.toLocaleString()} tokens`
                                    : ""
                                }
                                ${
                                  stage.error_message
                                    ? `• Error: ${stage.error_message}`
                                    : ""
                                }
                            </div>
                        </div>
                    </div>
                `;
      });

      html += "</div></div>";
    }

    // Token Usage
    if (analysisData.token_usage) {
      const usage = analysisData.token_usage;
      html += `
                <div class="analysis-section">
                    <div class="analysis-header">
                        <i class="fas fa-memory"></i>
                        Token Usage
                    </div>
                    <div class="token-usage">
                        <div class="token-stat">
                            <div class="token-value">${
                              usage.total_tokens
                                ? usage.total_tokens.toLocaleString()
                                : "N/A"
                            }</div>
                            <div class="token-label">Total</div>
                        </div>
                        <div class="token-stat">
                            <div class="token-value">${
                              usage.token_limit
                                ? usage.token_limit.toLocaleString()
                                : "N/A"
                            }</div>
                            <div class="token-label">Limit</div>
                        </div>
                        <div class="token-stat">
                            <div class="token-value">${
                              usage.utilization
                                ? (usage.utilization * 100).toFixed(1) + "%"
                                : "N/A"
                            }</div>
                            <div class="token-label">Used</div>
                        </div>
                    </div>
                </div>
            `;
    }

    // Recommendations
    if (
      analysisData.legal_analysis &&
      analysisData.legal_analysis.recommendations &&
      analysisData.legal_analysis.recommendations.length > 0
    ) {
      html += `
                <div class="analysis-section">
                    <div class="analysis-header">
                        <i class="fas fa-lightbulb"></i>
                        Recommendations
                    </div>
                    <div class="recommendations-list">
            `;

      analysisData.legal_analysis.recommendations
        .slice(0, 5)
        .forEach((recommendation) => {
          html += `
                    <div class="recommendation-item">
                        <i class="fas fa-arrow-right"></i>
                        <div class="recommendation-text">${this.escapeHtml(
                          recommendation
                        )}</div>
                    </div>
                `;
        });

      html += "</div></div>";
    }

    html += "</div>";
    return html;
  }

  formatStageName(stageName) {
    const stageNames = {
      document_processing: "Document Processing",
      intelligent_compression: "Intelligent Compression",
      legal_analysis: "Legal Analysis",
      final_response_generation: "Response Generation",
      pipeline_error: "Pipeline Error",
    };
    return (
      stageNames[stageName] ||
      stageName.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase())
    );
  }

  showProlongDurationModal() {
    // Set minimum date to today
    const today = new Date().toISOString().split("T")[0];
    this.prolongDatePicker.min = today;
    this.prolongDatePicker.value = "";
    this.prolongConfirmBtn.disabled = true;

    // Show modal (use flex to maintain centering)
    this.prolongDurationModal.style.display = "flex";

    // Focus on date picker
    setTimeout(() => {
      this.prolongDatePicker.focus();
    }, 100);
  }

  hideProlongDurationModal() {
    this.prolongDurationModal.style.display = "none";
    this.prolongDatePicker.value = "";
    this.prolongConfirmBtn.disabled = true;
  }

  showFindClientModal() {
    // Clear input and disable search button
    this.clientNameInput.value = "";
    this.clientSearchBtn.disabled = true;

    // Show modal (use flex to maintain centering)
    this.findClientModal.style.display = "flex";

    // Focus on client name input
    setTimeout(() => {
      this.clientNameInput.focus();
    }, 100);
  }

  hideFindClientModal() {
    this.findClientModal.style.display = "none";
    this.clientNameInput.value = "";
    this.clientSearchBtn.disabled = true;
  }

  searchForClient() {
    const clientName = this.clientNameInput.value.trim();
    if (!clientName) {
      this.showToast("Please enter a client name", "error");
      return;
    }

    // Build the question using the specified format
    const question = `Find and return all the documents for ${clientName}. Only return the document names and titles, nothing else.`;
    const displayMessage = `Find related to Client: ${clientName}`;

    // Hide modal
    this.hideFindClientModal();

    // Send message to LLM
    this.sendMessage(question, displayMessage);

    this.showToast(
      `Searching for documents related to ${clientName}...`,
      "info"
    );
  }

  confirmProlongDuration() {
    const selectedDate = this.prolongDatePicker.value;
    if (!selectedDate) {
      this.showToast("Please select a date", "error");
      return;
    }

    // Format the date for the LLM query
    const dateObj = new Date(selectedDate);
    const formattedDate = dateObj.toLocaleDateString("en-US", {
      year: "numeric",
      month: "long",
      day: "numeric",
    });

    // Build the question using the specified format
    const question = `Create a new amendment to this document. Only return the amendment, do not include any other text. This new amendment should contain the due date of this contract prolonged to this date: ${formattedDate}`;

    // Hide modal
    this.hideProlongDurationModal();

    // Send message for PDF generation instead of regular chat
    this.sendMessageForPDF(question, formattedDate);

    this.showToast(
      `Generating contract amendment document for ${formattedDate}...`,
      "info"
    );
  }

  async sendMessageForPDF(message, selectedDate) {
    // Determine which documents to search
    let documentIds = [];
    if (this.selectedDocuments.size > 0) {
      documentIds = Array.from(this.selectedDocuments);
    } else {
      // If no documents selected, use all completed documents
      documentIds = this.allDocuments
        .filter((doc) => doc.processing_status === "completed")
        .map((doc) => doc.id);
    }

    if (documentIds.length === 0) {
      this.showToast(
        "No documents available for amendment generation",
        "error"
      );
      return;
    }

    this.showLoading("Generating contract amendment...");

    try {
      // Choose endpoint based on analysis mode
      const endpoint = this.detailedAnalysisMode
        ? "/api/v1/chat/query"
        : "/api/v1/chat/iterative";

      const requestBody = {
        message: message,
        document_ids: documentIds,
        llm_provider: this.llmProvider,
        session_id: this.currentSessionId,
      };

      // Add detailed analysis parameters if enabled
      if (this.detailedAnalysisMode) {
        requestBody.detailed_analysis_mode = true;
        requestBody.enable_contradiction_detection =
          this.enableContradictionDetection;
        requestBody.enable_temporal_tracking = this.enableTemporalTracking;
        requestBody.enable_cross_document_reasoning =
          this.enableCrossDocumentReasoning;
        requestBody.max_analysis_tokens = this.maxAnalysisTokens;
        requestBody.use_iterative_rag = false; // Use multi-stage instead
      }

      // DEBUG: Log the request being sent
      console.log("PDF Generation Request:", {
        endpoint: `${this.apiBase}${endpoint}`,
        requestBody: requestBody,
        selectedDate: selectedDate,
        documentIds: documentIds,
      });

      const response = await fetch(`${this.apiBase}${endpoint}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      });

      const data = await response.json();

      // DEBUG: Log the response received
      console.log("PDF Generation Response:", {
        status: response.status,
        ok: response.ok,
        data: data,
        responseType: typeof data.response,
        responseLength: data.response ? data.response.length : 0,
      });

      if (response.ok) {
        this.currentSessionId = data.session_id;

        // DEBUG: Log the content before PDF generation
        console.log("Content for PDF generation:", {
          selectedDate: selectedDate,
          content: data.response,
          contentPreview: data.response
            ? data.response.substring(0, 200) + "..."
            : "No content",
        });

        // Generate and download PDF instead of showing in chat
        this.generateAndDownloadPDF(data.response, selectedDate);

        this.showToast(
          "Contract amendment generated and downloaded successfully!",
          "success"
        );
      } else {
        console.error("API Error:", data);
        throw new Error(data.detail || "Amendment generation error");
      }
    } catch (error) {
      console.error("PDF Generation Error:", error);
      this.showToast(`Amendment generation error: ${error.message}`, "error");
    } finally {
      this.hideLoading();
    }
  }

  generateAndDownloadPDF(content, selectedDate) {
    // DEBUG: Log PDF generation start
    console.log("Starting PDF generation:", {
      content: content,
      contentType: typeof content,
      contentLength: content ? content.length : 0,
      selectedDate: selectedDate,
    });

    // Create a new window for PDF generation
    const printWindow = window.open("", "_blank");

    if (!printWindow) {
      console.error("Failed to open print window - popup blocked?");
      this.showToast(
        "PDF generation failed - please allow popups for this site",
        "error"
      );
      return;
    }

    // Get current date for filename and document
    const today = new Date();
    const dateString = today.toISOString().split("T")[0];
    const timeString = today.toLocaleTimeString("en-US", {
      hour12: false,
      hour: "2-digit",
      minute: "2-digit",
    });

    // Format content for PDF
    const formattedContent = this.formatContentForPDF(content);
    console.log("Formatted content for PDF:", {
      originalContent: content,
      formattedContent: formattedContent,
      formattedLength: formattedContent ? formattedContent.length : 0,
    });

    // Create HTML content for PDF
    const htmlContent = `
      <!DOCTYPE html>
      <html>
        <head>
          <title>Contract Amendment - ${selectedDate}</title>
          <meta charset="UTF-8">
          <style>
            @page {
              margin: 1in;
              size: letter;
            }
            body {
              font-family: 'Times New Roman', serif;
              font-size: 12pt;
              line-height: 1.5;
              color: #000;
              max-width: none;
              margin: 0;
              padding: 0;
            }
            .header {
              text-align: center;
              margin-bottom: 30px;
              border-bottom: 2px solid #000;
              padding-bottom: 15px;
            }
            .header h1 {
              font-size: 16pt;
              font-weight: bold;
              margin: 0;
              text-transform: uppercase;
            }
            .document-info {
              margin-bottom: 25px;
              font-size: 10pt;
              text-align: right;
            }
            .content {
              text-align: justify;
              margin-bottom: 40px;
            }
            .content h2, .content h3, .content h4 {
              font-weight: bold;
              margin-top: 20px;
              margin-bottom: 10px;
            }
            .content h2 { font-size: 14pt; }
            .content h3 { font-size: 13pt; }
            .content h4 { font-size: 12pt; }
            .content p {
              margin-bottom: 12px;
              text-indent: 0.5in;
            }
            .content p:first-child {
              text-indent: 0;
            }
            .signature-section {
              margin-top: 50px;
              page-break-inside: avoid;
            }
            .signature-line {
              border-bottom: 1px solid #000;
              width: 300px;
              height: 40px;
              margin: 20px 0 5px 0;
              display: inline-block;
            }
            .signature-label {
              font-size: 10pt;
              margin-left: 10px;
            }
            .footer {
              margin-top: 30px;
              font-size: 10pt;
              text-align: center;
              border-top: 1px solid #ccc;
              padding-top: 10px;
            }
            @media print {
              body { -webkit-print-color-adjust: exact; }
              .no-print { display: none; }
            }
          </style>
        </head>
        <body>
          <div class="header">
            <h1>Contract Amendment</h1>
          </div>
          
          <div class="document-info">
            <strong>Generated:</strong> ${dateString} at ${timeString}<br>
            <strong>Amendment Date:</strong> ${selectedDate}
          </div>
          
          <div class="content">
            ${formattedContent}
          </div>
          
          <div class="signature-section">
            <p><strong>SIGNATURES:</strong></p>
            <br>
            <div class="signature-line"></div>
            <div class="signature-label">Party 1 Signature &nbsp;&nbsp;&nbsp;&nbsp; Date: ___________</div>
            <br><br>
            <div class="signature-line"></div>
            <div class="signature-label">Party 2 Signature &nbsp;&nbsp;&nbsp;&nbsp; Date: ___________</div>
          </div>
          
          <div class="footer">
            <p>This document was generated automatically. Please review carefully before signing.</p>
          </div>
          
          <script>
            // Auto-print when loaded
            window.onload = function() {
              setTimeout(function() {
                window.print();
                window.close();
              }, 500);
            };
          </script>
        </body>
      </html>
    `;

    // DEBUG: Log final HTML content
    console.log("Generated HTML for PDF:", {
      htmlContentLength: htmlContent.length,
      htmlPreview: htmlContent.substring(0, 500) + "...",
      printWindowExists: !!printWindow,
    });

    try {
      // Write content to new window and trigger print
      printWindow.document.write(htmlContent);
      printWindow.document.close();
      console.log("Successfully wrote content to print window");
    } catch (error) {
      console.error("Error writing to print window:", error);
      this.showToast("Error generating PDF document", "error");
    }
  }

  formatContentForPDF(content) {
    // Convert markdown-like content to HTML suitable for PDF
    let formattedContent = content;

    // Convert markdown headers to HTML
    formattedContent = formattedContent.replace(/^### (.*$)/gim, "<h3>$1</h3>");
    formattedContent = formattedContent.replace(/^## (.*$)/gim, "<h2>$1</h2>");
    formattedContent = formattedContent.replace(/^# (.*$)/gim, "<h2>$1</h2>");

    // Convert bold text
    formattedContent = formattedContent.replace(
      /\*\*(.*?)\*\*/g,
      "<strong>$1</strong>"
    );

    // Convert italic text
    formattedContent = formattedContent.replace(/\*(.*?)\*/g, "<em>$1</em>");

    // Convert line breaks to paragraphs
    const paragraphs = formattedContent.split("\n\n");
    formattedContent = paragraphs
      .filter((p) => p.trim().length > 0)
      .map((p) => `<p>${p.trim().replace(/\n/g, "<br>")}</p>`)
      .join("");

    return formattedContent;
  }

  showToast(message, type = "info") {
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.textContent = message;

    this.toastContainer.appendChild(toast);

    // Auto remove after 5 seconds
    setTimeout(() => {
      if (toast.parentNode) {
        toast.parentNode.removeChild(toast);
      }
    }, 5000);

    // Remove on click
    toast.addEventListener("click", () => {
      if (toast.parentNode) {
        toast.parentNode.removeChild(toast);
      }
    });
  }
}

// Initialize the app
const app = new LegalRAGApp();

// Demo mode function for testing
function enableDemoMode() {
  console.log("Demo mode enabled");
  app.showToast("Demo mode enabled - using mock data", "info");
}
