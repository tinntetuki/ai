// AI Content Creator Frontend JavaScript

class AIContentCreator {
    constructor() {
        this.apiBase = window.location.origin;
        this.currentTasks = new Map();
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupDragAndDrop();
        this.updateRangeDisplays();
        this.loadModels();
    }

    setupEventListeners() {
        // File input change events
        document.getElementById('upscale-file').addEventListener('change', (e) => this.handleFileSelect(e, 'upscale'));
        document.getElementById('speech-file').addEventListener('change', (e) => this.handleFileSelect(e, 'speech'));
        document.getElementById('detection-file').addEventListener('change', (e) => this.handleFileSelect(e, 'detection'));

        // Range input updates
        document.getElementById('tts-speed').addEventListener('input', (e) => this.updateRangeDisplay(e.target, 'x'));
        document.getElementById('tts-pitch').addEventListener('input', (e) => this.updateRangeDisplay(e.target, 'x'));
        document.getElementById('tts-volume').addEventListener('input', (e) => this.updateRangeDisplay(e.target, 'x'));
        document.getElementById('detection-confidence').addEventListener('input', (e) => this.updateRangeDisplay(e.target));

        // Provider change event
        document.getElementById('tts-provider').addEventListener('change', (e) => this.updateVoices(e.target.value));
        document.getElementById('tts-language').addEventListener('change', (e) => this.updateVoicesForLanguage(e.target.value));
    }

    setupDragAndDrop() {
        const uploadAreas = document.querySelectorAll('.upload-area');

        uploadAreas.forEach(area => {
            area.addEventListener('dragover', (e) => {
                e.preventDefault();
                area.classList.add('dragover');
            });

            area.addEventListener('dragleave', (e) => {
                e.preventDefault();
                area.classList.remove('dragover');
            });

            area.addEventListener('drop', (e) => {
                e.preventDefault();
                area.classList.remove('dragover');

                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    const tabId = area.closest('.tab-pane').id;
                    const type = tabId.replace('-tab', '');
                    this.handleFileSelect({ target: { files } }, type);
                }
            });
        });
    }

    updateRangeDisplays() {
        // Initialize range displays
        this.updateRangeDisplay(document.getElementById('tts-speed'), 'x');
        this.updateRangeDisplay(document.getElementById('tts-pitch'), 'x');
        this.updateRangeDisplay(document.getElementById('tts-volume'), 'x');
        this.updateRangeDisplay(document.getElementById('detection-confidence'));
    }

    updateRangeDisplay(element, suffix = '') {
        const value = parseFloat(element.value);
        const display = element.parentElement.querySelector('.text-muted');
        if (display) {
            display.textContent = `${value}${suffix}`;
        }
    }

    handleFileSelect(event, type) {
        const file = event.target.files[0];
        if (!file) return;

        const uploadArea = document.querySelector(`#${type}-tab .upload-area`);
        const icon = uploadArea.querySelector('i');
        const title = uploadArea.querySelector('h5');
        const description = uploadArea.querySelector('p');

        // Update upload area to show selected file
        icon.className = 'fas fa-file fa-3x mb-3 text-success';
        title.textContent = 'File Selected';
        description.textContent = `${file.name} (${this.formatFileSize(file.size)})`;

        // Store the file for processing
        this[`${type}File`] = file;
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    async loadModels() {
        try {
            // Load available voices for TTS
            await this.updateVoices('edge');
        } catch (error) {
            console.warn('Failed to load models:', error);
        }
    }

    async updateVoices(provider) {
        try {
            const response = await fetch(`${this.apiBase}/api/v1/tts/voices?provider=${provider}`);
            const voices = await response.json();

            const voiceSelect = document.getElementById('tts-voice');
            voiceSelect.innerHTML = '';

            voices.forEach(voice => {
                const option = document.createElement('option');
                option.value = voice.name || voice.short_name || voice.code;
                option.textContent = voice.name || voice.short_name || voice.code;
                voiceSelect.appendChild(option);
            });
        } catch (error) {
            console.warn('Failed to load voices:', error);
        }
    }

    async updateVoicesForLanguage(language) {
        // Update voices based on selected language
        const provider = document.getElementById('tts-provider').value;
        await this.updateVoices(provider);
    }

    // Processing Functions
    async startUpscaling() {
        if (!this.upscaleFile) {
            this.showError('Please select a video file first.');
            return;
        }

        const formData = new FormData();
        formData.append('file', this.upscaleFile);
        formData.append('scale', document.getElementById('upscale-factor').value);
        formData.append('model', document.getElementById('upscale-model').value);

        try {
            const response = await fetch(`${this.apiBase}/api/v1/upscale/upload`, {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.task_id) {
                this.addTaskToStatus(result.task_id, 'Video Upscaling', 'processing');
                this.pollTaskStatus(result.task_id, 'upscale');
            }
        } catch (error) {
            this.showError('Failed to start upscaling: ' + error.message);
        }
    }

    async startTranscription() {
        if (!this.speechFile) {
            this.showError('Please select an audio/video file first.');
            return;
        }

        const formData = new FormData();
        formData.append('file', this.speechFile);
        formData.append('model', document.getElementById('speech-model').value);
        formData.append('language', document.getElementById('speech-language').value);
        formData.append('format', document.getElementById('speech-format').value);

        const keywords = document.getElementById('speech-keywords').value;
        if (keywords) {
            formData.append('keywords', keywords);
        }

        try {
            const response = await fetch(`${this.apiBase}/api/v1/speech/transcribe`, {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.task_id) {
                this.addTaskToStatus(result.task_id, 'Speech Transcription', 'processing');
                this.pollTaskStatus(result.task_id, 'speech');
            }
        } catch (error) {
            this.showError('Failed to start transcription: ' + error.message);
        }
    }

    async startTTS() {
        const text = document.getElementById('tts-text').value.trim();
        if (!text) {
            this.showError('Please enter text to convert to speech.');
            return;
        }

        const requestData = {
            text: text,
            voice_profile: {
                provider: document.getElementById('tts-provider').value,
                voice_name: document.getElementById('tts-voice').value,
                language: document.getElementById('tts-language').value,
                style: document.getElementById('tts-style').value,
                speed: parseFloat(document.getElementById('tts-speed').value),
                pitch: parseFloat(document.getElementById('tts-pitch').value),
                volume: parseFloat(document.getElementById('tts-volume').value)
            }
        };

        try {
            const response = await fetch(`${this.apiBase}/api/v1/tts/synthesize`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            });

            const result = await response.json();

            if (result.task_id) {
                this.addTaskToStatus(result.task_id, 'Text-to-Speech', 'processing');
                this.pollTaskStatus(result.task_id, 'tts');
            }
        } catch (error) {
            this.showError('Failed to start TTS: ' + error.message);
        }
    }

    async startDetection() {
        if (!this.detectionFile) {
            this.showError('Please select a video file first.');
            return;
        }

        const formData = new FormData();
        formData.append('file', this.detectionFile);
        formData.append('model', document.getElementById('detection-model').value);
        formData.append('confidence', document.getElementById('detection-confidence').value);
        formData.append('sample_rate', document.getElementById('detection-fps').value);
        formData.append('annotate', document.getElementById('detection-annotate').checked);
        formData.append('speech_analysis', document.getElementById('detection-speech').checked);

        const keywords = document.getElementById('detection-keywords').value;
        if (keywords) {
            formData.append('keywords', keywords);
        }

        try {
            const response = await fetch(`${this.apiBase}/api/v1/products/detect`, {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.task_id) {
                this.addTaskToStatus(result.task_id, 'Product Detection', 'processing');
                this.pollTaskStatus(result.task_id, 'products');
            }
        } catch (error) {
            this.showError('Failed to start detection: ' + error.message);
        }
    }

    // Task Management
    addTaskToStatus(taskId, taskName, status) {
        const statusSection = document.getElementById('processing-status');
        const statusCards = document.getElementById('status-cards');

        statusSection.style.display = 'block';

        const taskCard = document.createElement('div');
        taskCard.className = 'processing-card';
        taskCard.id = `task-${taskId}`;

        taskCard.innerHTML = `
            <div class="d-flex justify-content-between align-items-center">
                <div>
                    <h6 class="mb-1">${taskName}</h6>
                    <small class="text-muted">Task ID: ${taskId}</small>
                </div>
                <div class="text-end">
                    <span class="status-badge status-${status}" id="status-${taskId}">${status.toUpperCase()}</span>
                    <div class="progress mt-2" style="width: 200px;">
                        <div class="progress-bar" id="progress-${taskId}" role="progressbar" style="width: 0%"></div>
                    </div>
                </div>
            </div>
        `;

        statusCards.appendChild(taskCard);
        this.currentTasks.set(taskId, { name: taskName, status, element: taskCard });
    }

    updateTaskStatus(taskId, status, progress = 0) {
        const statusBadge = document.getElementById(`status-${taskId}`);
        const progressBar = document.getElementById(`progress-${taskId}`);

        if (statusBadge) {
            statusBadge.className = `status-badge status-${status}`;
            statusBadge.textContent = status.toUpperCase();
        }

        if (progressBar) {
            progressBar.style.width = `${progress}%`;
        }

        if (this.currentTasks.has(taskId)) {
            this.currentTasks.get(taskId).status = status;
        }
    }

    async pollTaskStatus(taskId, endpoint) {
        const poll = async () => {
            try {
                const response = await fetch(`${this.apiBase}/api/v1/${endpoint}/status/${taskId}`);
                const result = await response.json();

                this.updateTaskStatus(taskId, result.status, result.progress || 0);

                if (result.status === 'completed') {
                    this.handleTaskCompleted(taskId, endpoint, result);
                } else if (result.status === 'failed') {
                    this.handleTaskFailed(taskId, result.error);
                } else {
                    // Continue polling
                    setTimeout(poll, 2000);
                }
            } catch (error) {
                console.error('Failed to poll task status:', error);
                setTimeout(poll, 5000); // Retry after longer delay
            }
        };

        poll();
    }

    handleTaskCompleted(taskId, endpoint, result) {
        this.updateTaskStatus(taskId, 'completed', 100);
        this.addResultToGrid(taskId, endpoint, result);
        this.showSuccess(`Task ${taskId} completed successfully!`);
    }

    handleTaskFailed(taskId, error) {
        this.updateTaskStatus(taskId, 'failed', 0);
        this.showError(`Task ${taskId} failed: ${error}`);
    }

    addResultToGrid(taskId, endpoint, result) {
        const resultsSection = document.getElementById('results-section');
        const resultsGrid = document.getElementById('results-grid');

        resultsSection.style.display = 'block';

        const task = this.currentTasks.get(taskId);
        const resultCard = document.createElement('div');
        resultCard.className = 'result-card';

        let downloadLinks = '';
        if (result.download_urls) {
            Object.entries(result.download_urls).forEach(([type, url]) => {
                downloadLinks += `
                    <a href="${url}" class="download-btn me-2 mb-2" download>
                        <i class="fas fa-download me-1"></i>Download ${type.toUpperCase()}
                    </a>
                `;
            });
        }

        resultCard.innerHTML = `
            <h6><i class="fas fa-check-circle text-success me-2"></i>${task.name}</h6>
            <p class="text-muted">Task ID: ${taskId}</p>
            ${result.summary ? `<p><strong>Summary:</strong> ${result.summary}</p>` : ''}
            ${result.duration ? `<p><strong>Duration:</strong> ${result.duration}s</p>` : ''}
            ${result.file_size ? `<p><strong>File Size:</strong> ${this.formatFileSize(result.file_size)}</p>` : ''}
            <div class="mt-3">
                ${downloadLinks}
            </div>
        `;

        resultsGrid.appendChild(resultCard);
    }

    // Utility Functions
    showSuccess(message) {
        this.showNotification(message, 'success');
    }

    showError(message) {
        this.showNotification(message, 'danger');
    }

    showNotification(message, type) {
        // Create Bootstrap toast notification
        const toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container position-fixed top-0 end-0 p-3';
        toastContainer.style.zIndex = '9999';

        const toast = document.createElement('div');
        toast.className = `toast align-items-center text-white bg-${type} border-0`;
        toast.setAttribute('role', 'alert');

        toast.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        `;

        toastContainer.appendChild(toast);
        document.body.appendChild(toastContainer);

        const bsToast = new bootstrap.Toast(toast);
        bsToast.show();

        // Remove toast container after it's hidden
        toast.addEventListener('hidden.bs.toast', () => {
            document.body.removeChild(toastContainer);
        });
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.aiContentCreator = new AIContentCreator();
});

// Global functions for button onclick events
function startUpscaling() {
    window.aiContentCreator.startUpscaling();
}

function startTranscription() {
    window.aiContentCreator.startTranscription();
}

function startTTS() {
    window.aiContentCreator.startTTS();
}

function startDetection() {
    window.aiContentCreator.startDetection();
}

// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Navbar scroll effect
window.addEventListener('scroll', () => {
    const navbar = document.querySelector('.navbar');
    if (window.scrollY > 50) {
        navbar.style.background = 'rgba(26, 26, 46, 0.98)';
    } else {
        navbar.style.background = 'rgba(26, 26, 46, 0.95)';
    }
});