// AI Content Creator Frontend JavaScript

class AIContentCreator {
    constructor() {
        this.apiBase = window.location.origin;
        this.currentTasks = new Map();
        this.processingHistory = JSON.parse(localStorage.getItem('processingHistory')) || [];
        this.settings = JSON.parse(localStorage.getItem('appSettings')) || this.getDefaultSettings();
        this.analytics = {
            charts: {},
            updateInterval: null,
            systemMetrics: {}
        };
        this.init();
    }

    getDefaultSettings() {
        return {
            defaultQuality: 'balanced',
            concurrentLimit: 2,
            autoDownload: true,
            notifications: true,
            outputFormat: 'mp4',
            sizeWarning: 100,
            cleanupTemp: true,
            processingPriority: 'normal',
            gpuUsage: 'auto',
            memoryLimit: 8,
            apiTimeout: 300,
            retryAttempts: 3
        };
    }

    init() {
        this.setupEventListeners();
        this.setupDragAndDrop();
        this.updateRangeDisplays();
        this.loadModels();
        this.initializeAnalytics();
        this.startSystemMonitoring();
        this.generateDemoData();
    }

    setupEventListeners() {
        // File input change events
        document.getElementById('upscale-file').addEventListener('change', (e) => this.handleFileSelect(e, 'upscale'));
        document.getElementById('speech-file').addEventListener('change', (e) => this.handleFileSelect(e, 'speech'));
        document.getElementById('detection-file').addEventListener('change', (e) => this.handleFileSelect(e, 'detection'));

        // Add multiple file support
        document.getElementById('upscale-file').setAttribute('multiple', 'true');
        document.getElementById('speech-file').setAttribute('multiple', 'true');
        document.getElementById('detection-file').setAttribute('multiple', 'true');

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
        const files = Array.from(event.target.files);
        if (!files.length) return;

        const uploadArea = document.querySelector(`#${type}-tab .upload-area`);
        const icon = uploadArea.querySelector('i');
        const title = uploadArea.querySelector('h5');
        const description = uploadArea.querySelector('p');

        // Validate all files
        const validFiles = [];
        const errors = [];

        for (const file of files) {
            const validationResult = this.validateFile(file, type);
            if (validationResult.valid) {
                validFiles.push(file);
            } else {
                errors.push(`${file.name}: ${validationResult.message}`);
            }
        }

        if (errors.length > 0) {
            this.showError(`File validation errors:\n${errors.join('\n')}`);
        }

        if (validFiles.length === 0) {
            event.target.value = '';
            return;
        }

        // Update upload area display
        if (validFiles.length === 1) {
            const file = validFiles[0];
            icon.className = 'fas fa-file fa-3x mb-3 text-success';
            title.textContent = 'File Selected';
            description.innerHTML = `
                <strong>${file.name}</strong><br>
                <small class="text-muted">${this.formatFileSize(file.size)} • ${file.type}</small><br>
                <small class="text-success"><i class="fas fa-check-circle"></i> File validated successfully</small>
            `;

            // Generate file preview for single file
            this.generateFilePreview(file, type);
        } else {
            icon.className = 'fas fa-files fa-3x mb-3 text-success';
            title.textContent = `${validFiles.length} Files Selected`;

            const totalSize = validFiles.reduce((sum, file) => sum + file.size, 0);
            description.innerHTML = `
                <strong>${validFiles.length} files ready for processing</strong><br>
                <small class="text-muted">Total size: ${this.formatFileSize(totalSize)}</small><br>
                <small class="text-success"><i class="fas fa-check-circle"></i> All files validated</small>
            `;

            // Show file list
            this.showFileList(validFiles, type);
        }

        // Store validated files
        this[`${type}Files`] = validFiles;

        // Show batch processing options if multiple files
        if (validFiles.length > 1) {
            this.showBatchProcessingOptions(type);
        }
    }

    validateFile(file, type) {
        const validationRules = {
            upscale: {
                maxSize: 500 * 1024 * 1024, // 500MB
                allowedTypes: ['video/mp4', 'video/avi', 'video/mov', 'video/mkv', 'video/quicktime'],
                allowedExtensions: ['.mp4', '.avi', '.mov', '.mkv']
            },
            speech: {
                maxSize: 100 * 1024 * 1024, // 100MB
                allowedTypes: ['video/mp4', 'audio/mpeg', 'audio/wav', 'audio/mp4', 'video/quicktime'],
                allowedExtensions: ['.mp4', '.mp3', '.wav', '.m4a', '.mov']
            },
            detection: {
                maxSize: 500 * 1024 * 1024, // 500MB
                allowedTypes: ['video/mp4', 'video/avi', 'video/mov', 'video/mkv', 'video/quicktime'],
                allowedExtensions: ['.mp4', '.avi', '.mov', '.mkv']
            }
        };

        const rules = validationRules[type];
        if (!rules) return { valid: false, message: 'Invalid file type specified.' };

        // Check file size
        if (file.size > rules.maxSize) {
            return {
                valid: false,
                message: `File size (${this.formatFileSize(file.size)}) exceeds maximum allowed size (${this.formatFileSize(rules.maxSize)}).`
            };
        }

        // Check file type
        const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
        const isValidType = rules.allowedTypes.includes(file.type) || rules.allowedExtensions.includes(fileExtension);

        if (!isValidType) {
            return {
                valid: false,
                message: `File type not supported. Allowed formats: ${rules.allowedExtensions.join(', ')}`
            };
        }

        // Additional validation for video duration (estimate based on file size)
        if (type === 'upscale' && file.size > 100 * 1024 * 1024) {
            return {
                valid: true,
                message: '',
                warning: 'Large video file detected. Processing may take significant time.'
            };
        }

        return { valid: true, message: '' };
    }

    generateFilePreview(file, type) {
        const previewContainer = document.querySelector(`#${type}-tab .upload-area`);
        let existingPreview = previewContainer.querySelector('.file-preview');
        if (existingPreview) {
            existingPreview.remove();
        }

        const previewDiv = document.createElement('div');
        previewDiv.className = 'file-preview mt-3';

        if (file.type.startsWith('video/')) {
            const video = document.createElement('video');
            video.src = URL.createObjectURL(file);
            video.controls = false;
            video.muted = true;
            video.style.maxWidth = '200px';
            video.style.maxHeight = '150px';
            video.style.borderRadius = '8px';
            video.className = 'border';

            // Get video metadata
            video.addEventListener('loadedmetadata', () => {
                const duration = Math.round(video.duration);
                const durationStr = `${Math.floor(duration / 60)}:${(duration % 60).toString().padStart(2, '0')}`;
                const metadata = document.createElement('small');
                metadata.className = 'text-muted d-block mt-2';
                metadata.innerHTML = `<i class="fas fa-clock"></i> Duration: ${durationStr} • Resolution: ${video.videoWidth}x${video.videoHeight}`;
                previewDiv.appendChild(metadata);
            });

            previewDiv.appendChild(video);
        } else if (file.type.startsWith('audio/')) {
            const audio = document.createElement('audio');
            audio.src = URL.createObjectURL(file);
            audio.controls = true;
            audio.style.maxWidth = '300px';
            audio.className = 'mt-2';

            previewDiv.appendChild(audio);
        }

        previewContainer.appendChild(previewDiv);
    }

    showFileList(files, type) {
        const uploadArea = document.querySelector(`#${type}-tab .upload-area`);
        let existingList = uploadArea.querySelector('.file-list');
        if (existingList) {
            existingList.remove();
        }

        const fileList = document.createElement('div');
        fileList.className = 'file-list mt-3';
        fileList.style.maxHeight = '200px';
        fileList.style.overflowY = 'auto';

        files.forEach((file, index) => {
            const fileItem = document.createElement('div');
            fileItem.className = 'file-item d-flex justify-content-between align-items-center p-2 border rounded mb-1';
            fileItem.innerHTML = `
                <div class="flex-grow-1">
                    <small><strong>${file.name}</strong></small><br>
                    <small class="text-muted">${this.formatFileSize(file.size)}</small>
                </div>
                <button class="btn btn-sm btn-outline-danger" onclick="window.aiContentCreator.removeFile(${index}, '${type}')">
                    <i class="fas fa-times"></i>
                </button>
            `;
            fileList.appendChild(fileItem);
        });

        uploadArea.appendChild(fileList);
    }

    showBatchProcessingOptions(type) {
        const tabPane = document.querySelector(`#${type}-tab`);
        let existingOptions = tabPane.querySelector('.batch-options');
        if (existingOptions) {
            existingOptions.remove();
        }

        const batchOptions = document.createElement('div');
        batchOptions.className = 'batch-options mt-4 p-3 border rounded';
        batchOptions.innerHTML = `
            <h6><i class="fas fa-layer-group"></i> Batch Processing Options</h6>
            <div class="row">
                <div class="col-md-6">
                    <label class="form-label">Processing Mode</label>
                    <select class="form-control" id="${type}-batch-mode">
                        <option value="sequential">Sequential (One at a time)</option>
                        <option value="parallel">Parallel (Multiple at once)</option>
                    </select>
                </div>
                <div class="col-md-6">
                    <label class="form-label">Priority</label>
                    <select class="form-control" id="${type}-batch-priority">
                        <option value="normal">Normal</option>
                        <option value="high">High Priority</option>
                        <option value="low">Low Priority</option>
                    </select>
                </div>
            </div>
            <div class="mt-3">
                <div class="form-check">
                    <input class="form-check-input" type="checkbox" id="${type}-stop-on-error" checked>
                    <label class="form-check-label" for="${type}-stop-on-error">
                        Stop processing if any file fails
                    </label>
                </div>
            </div>
        `;

        const processButton = tabPane.querySelector(`button[onclick*="start${type.charAt(0).toUpperCase() + type.slice(1)}"]`);
        processButton.parentNode.insertBefore(batchOptions, processButton);
    }

    removeFile(index, type) {
        const files = this[`${type}Files`];
        if (files && index >= 0 && index < files.length) {
            files.splice(index, 1);

            if (files.length === 0) {
                // Reset upload area
                const uploadArea = document.querySelector(`#${type}-tab .upload-area`);
                const icon = uploadArea.querySelector('i');
                const title = uploadArea.querySelector('h5');
                const description = uploadArea.querySelector('p');

                icon.className = `fas fa-${type === 'speech' ? 'microphone' : 'video'} fa-3x mb-3`;
                title.textContent = `Upload ${type === 'speech' ? 'Audio/Video' : 'Video'} File`;
                description.textContent = 'Click here or drag and drop your file';

                // Remove file list and batch options
                const fileList = uploadArea.querySelector('.file-list');
                const batchOptions = document.querySelector(`#${type}-tab .batch-options`);
                if (fileList) fileList.remove();
                if (batchOptions) batchOptions.remove();

                delete this[`${type}Files`];
            } else {
                // Update file list
                this.showFileList(files, type);
            }
        }
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
        const files = this.upscaleFiles || [this.upscaleFile];
        if (!files || files.length === 0) {
            this.showError('Please select video file(s) first.');
            return;
        }

        const scale = document.getElementById('upscale-factor').value;
        const model = document.getElementById('upscale-model').value;

        if (files.length === 1) {
            // Single file processing
            await this.processSingleFile(files[0], 'upscale', { scale, model });
        } else {
            // Batch processing
            const batchMode = document.getElementById('upscale-batch-mode')?.value || 'sequential';
            const priority = document.getElementById('upscale-batch-priority')?.value || 'normal';
            const stopOnError = document.getElementById('upscale-stop-on-error')?.checked ?? true;

            await this.processBatchFiles(files, 'upscale', { scale, model }, {
                mode: batchMode,
                priority: priority,
                stopOnError: stopOnError
            });
        }
    }

    async processSingleFile(file, type, params) {
        const formData = new FormData();
        formData.append('file', file);

        // Add type-specific parameters
        Object.entries(params).forEach(([key, value]) => {
            formData.append(key, value);
        });

        try {
            const endpoint = this.getEndpointForType(type);
            const response = await fetch(`${this.apiBase}/api/v1/${endpoint}/upload`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();

            if (result.task_id) {
                this.addTaskToStatus(result.task_id, `${this.getTaskNameForType(type)} - ${file.name}`, 'processing');
                this.pollTaskStatus(result.task_id, endpoint);
            }
        } catch (error) {
            this.showError(`Failed to start ${type}: ${error.message}`);
        }
    }

    async processBatchFiles(files, type, params, options) {
        const endpoint = this.getEndpointForType(type);
        const taskName = this.getTaskNameForType(type);

        this.showSuccess(`Starting batch processing of ${files.length} files...`);

        if (options.mode === 'parallel') {
            // Process all files in parallel
            const promises = files.map(file => this.processSingleFile(file, type, params));

            try {
                await Promise.all(promises);
            } catch (error) {
                if (options.stopOnError) {
                    this.showError('Batch processing stopped due to error: ' + error.message);
                }
            }
        } else {
            // Process files sequentially
            for (let i = 0; i < files.length; i++) {
                const file = files[i];

                try {
                    await this.processSingleFile(file, type, params);
                    this.showSuccess(`Started processing file ${i + 1} of ${files.length}: ${file.name}`);

                    // Small delay between sequential uploads
                    if (i < files.length - 1) {
                        await new Promise(resolve => setTimeout(resolve, 1000));
                    }
                } catch (error) {
                    this.showError(`Failed to process ${file.name}: ${error.message}`);

                    if (options.stopOnError) {
                        this.showError('Batch processing stopped due to error');
                        break;
                    }
                }
            }
        }
    }

    getEndpointForType(type) {
        const endpoints = {
            'upscale': 'upscale',
            'speech': 'speech',
            'detection': 'products'
        };
        return endpoints[type] || type;
    }

    getTaskNameForType(type) {
        const names = {
            'upscale': 'Video Upscaling',
            'speech': 'Speech Transcription',
            'detection': 'Product Detection'
        };
        return names[type] || type;
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

        const startTime = new Date();

        taskCard.innerHTML = `
            <div class="d-flex justify-content-between align-items-center">
                <div class="flex-grow-1">
                    <div class="d-flex justify-content-between align-items-center">
                        <h6 class="mb-1">${taskName}</h6>
                        <span class="status-badge status-${status}" id="status-${taskId}">${status.toUpperCase()}</span>
                    </div>
                    <small class="text-muted">Task ID: ${taskId} • Started: ${startTime.toLocaleTimeString()}</small>
                    <div class="mt-2">
                        <div class="d-flex justify-content-between align-items-center mb-1">
                            <small class="text-muted" id="progress-text-${taskId}">Initializing...</small>
                            <small class="text-muted" id="progress-percent-${taskId}">0%</small>
                        </div>
                        <div class="progress" style="height: 8px;">
                            <div class="progress-bar progress-bar-striped progress-bar-animated"
                                 id="progress-${taskId}" role="progressbar" style="width: 0%"></div>
                        </div>
                        <div class="mt-2">
                            <small class="text-muted" id="eta-${taskId}"></small>
                            <small class="text-muted float-end" id="speed-${taskId}"></small>
                        </div>
                    </div>
                </div>
                <div class="ms-3">
                    <button class="btn btn-sm btn-outline-danger" onclick="cancelTask('${taskId}')"
                            id="cancel-btn-${taskId}" title="Cancel Task">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
            </div>
        `;

        statusCards.appendChild(taskCard);
        this.currentTasks.set(taskId, {
            name: taskName,
            status,
            element: taskCard,
            startTime: startTime,
            lastUpdate: startTime
        });
    }

    updateTaskStatus(taskId, status, progress = 0, details = {}) {
        const task = this.currentTasks.get(taskId);
        if (!task) return;

        const statusBadge = document.getElementById(`status-${taskId}`);
        const progressBar = document.getElementById(`progress-${taskId}`);
        const progressText = document.getElementById(`progress-text-${taskId}`);
        const progressPercent = document.getElementById(`progress-percent-${taskId}`);
        const etaElement = document.getElementById(`eta-${taskId}`);
        const speedElement = document.getElementById(`speed-${taskId}`);

        // Update status badge
        if (statusBadge) {
            statusBadge.className = `status-badge status-${status}`;
            statusBadge.textContent = status.toUpperCase();
        }

        // Update progress bar
        if (progressBar) {
            progressBar.style.width = `${progress}%`;

            // Remove animation when completed
            if (status === 'completed' || status === 'failed') {
                progressBar.classList.remove('progress-bar-animated');
            }
        }

        // Update progress text and percentage
        if (progressPercent) {
            progressPercent.textContent = `${Math.round(progress)}%`;
        }

        if (progressText) {
            const statusMessages = {
                'pending': 'Waiting in queue...',
                'processing': details.current_step || 'Processing...',
                'completed': 'Processing completed!',
                'failed': 'Processing failed'
            };
            progressText.textContent = statusMessages[status] || 'Processing...';
        }

        // Calculate and display ETA and speed
        const now = new Date();
        const elapsed = (now - task.startTime) / 1000; // seconds

        if (progress > 0 && progress < 100 && status === 'processing') {
            const eta = ((100 - progress) / progress) * elapsed;
            const etaMinutes = Math.floor(eta / 60);
            const etaSeconds = Math.floor(eta % 60);

            if (etaElement) {
                etaElement.innerHTML = `<i class="fas fa-clock"></i> ETA: ${etaMinutes}:${etaSeconds.toString().padStart(2, '0')}`;
            }

            // Show processing speed if available
            if (details.frames_processed && details.total_frames) {
                const fps = details.frames_processed / elapsed;
                if (speedElement) {
                    speedElement.innerHTML = `${fps.toFixed(1)} fps`;
                }
            } else if (details.processing_speed) {
                if (speedElement) {
                    speedElement.innerHTML = details.processing_speed;
                }
            }
        } else if (status === 'completed') {
            if (etaElement) {
                etaElement.innerHTML = `<i class="fas fa-check"></i> Completed in ${Math.floor(elapsed / 60)}:${(Math.floor(elapsed) % 60).toString().padStart(2, '0')}`;
            }
            if (speedElement) {
                speedElement.innerHTML = '';
            }
        } else if (status === 'failed') {
            if (etaElement) {
                etaElement.innerHTML = `<i class="fas fa-exclamation-triangle"></i> Failed after ${Math.floor(elapsed / 60)}:${(Math.floor(elapsed) % 60).toString().padStart(2, '0')}`;
            }
            if (speedElement) {
                speedElement.innerHTML = '';
            }
        }

        // Update task object
        task.status = status;
        task.lastUpdate = now;
    }

    async pollTaskStatus(taskId, endpoint) {
        const poll = async () => {
            try {
                const response = await fetch(`${this.apiBase}/api/v1/${endpoint}/status/${taskId}`);

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }

                const result = await response.json();

                // Extract detailed progress information
                const progressDetails = {
                    current_step: result.current_step,
                    frames_processed: result.frames_processed,
                    total_frames: result.total_frames,
                    processing_speed: result.processing_speed
                };

                this.updateTaskStatus(taskId, result.status, result.progress || 0, progressDetails);

                if (result.status === 'completed') {
                    this.handleTaskCompleted(taskId, endpoint, result);
                } else if (result.status === 'failed' || result.status === 'cancelled') {
                    this.handleTaskFailed(taskId, result.error || 'Task was cancelled');
                } else if (result.status === 'processing' || result.status === 'pending') {
                    // Continue polling with adaptive interval
                    const pollInterval = result.progress > 0 ? 1000 : 3000;
                    setTimeout(poll, pollInterval);
                }
            } catch (error) {
                console.error('Failed to poll task status:', error);

                // Check if task was cancelled by user
                const task = this.currentTasks.get(taskId);
                if (task && task.status !== 'cancelled') {
                    this.updateTaskStatus(taskId, 'failed', 0);
                    this.showError(`Connection error: ${error.message}`);

                    // Retry with exponential backoff
                    setTimeout(poll, 5000);
                }
            }
        };

        poll();
    }

    async cancelTask(taskId) {
        try {
            const task = this.currentTasks.get(taskId);
            if (!task || task.status === 'completed' || task.status === 'failed') {
                return;
            }

            // Update UI immediately
            this.updateTaskStatus(taskId, 'cancelled', 0);

            // Attempt to cancel on server (implement this endpoint in backend)
            const response = await fetch(`${this.apiBase}/api/v1/tasks/cancel/${taskId}`, {
                method: 'POST'
            });

            if (response.ok) {
                this.showSuccess('Task cancelled successfully');
            } else {
                this.showError('Task cancellation requested, but may still be processing');
            }

            // Hide cancel button
            const cancelBtn = document.getElementById(`cancel-btn-${taskId}`);
            if (cancelBtn) {
                cancelBtn.style.display = 'none';
            }

        } catch (error) {
            console.error('Failed to cancel task:', error);
            this.showError('Failed to cancel task: ' + error.message);
        }
    }

    handleTaskCompleted(taskId, endpoint, result) {
        this.updateTaskStatus(taskId, 'completed', 100);
        this.addResultToGrid(taskId, endpoint, result);

        // Add to analytics history
        const task = this.currentTasks.get(taskId);
        if (task) {
            this.addToHistory(
                taskId,
                result.fileName || `Task ${taskId}`,
                endpoint,
                'completed',
                task.startTime,
                new Date(),
                result.file_size || null
            );
        }

        this.showSuccess(`Task ${taskId} completed successfully!`);
        this.showNotification('Task Completed', `${task?.name || 'Task'} completed successfully!`, 'success');
    }

    handleTaskFailed(taskId, error) {
        this.updateTaskStatus(taskId, 'failed', 0);

        // Add to analytics history
        const task = this.currentTasks.get(taskId);
        if (task) {
            this.addToHistory(
                taskId,
                `Task ${taskId}`,
                task.type || 'unknown',
                'failed',
                task.startTime,
                new Date(),
                null
            );
        }

        this.showError(`Task ${taskId} failed: ${error}`);
        this.showNotification('Task Failed', `Task failed: ${error}`, 'error');
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

    // History Management
    addToHistory(taskId, fileName, type, status, startTime, endTime = null, fileSize = null) {
        const historyItem = {
            taskId: taskId,
            fileName: fileName,
            type: type,
            status: status,
            startTime: startTime,
            endTime: endTime,
            fileSize: fileSize,
            duration: endTime ? Math.round((endTime - startTime) / 1000) : null
        };

        // Add to beginning of history
        this.processingHistory.unshift(historyItem);

        // Limit history to 100 items
        if (this.processingHistory.length > 100) {
            this.processingHistory = this.processingHistory.slice(0, 100);
        }

        // Save to localStorage
        localStorage.setItem('processingHistory', JSON.stringify(this.processingHistory));

        // Refresh history display if visible
        this.loadProcessingHistory();
    }

    loadProcessingHistory() {
        const tbody = document.getElementById('history-table-body');
        const emptyState = document.getElementById('history-empty');

        if (this.processingHistory.length === 0) {
            tbody.innerHTML = '';
            emptyState.style.display = 'block';
            return;
        }

        emptyState.style.display = 'none';

        // Apply filters
        const statusFilter = document.getElementById('history-filter-status')?.value || '';
        const typeFilter = document.getElementById('history-filter-type')?.value || '';
        const dateFilter = document.getElementById('history-filter-date')?.value || '';
        const searchFilter = document.getElementById('history-search')?.value.toLowerCase() || '';

        const filteredHistory = this.processingHistory.filter(item => {
            const matchesStatus = !statusFilter || item.status === statusFilter;
            const matchesType = !typeFilter || item.type === typeFilter;
            const matchesDate = !dateFilter || new Date(item.startTime).toDateString() === new Date(dateFilter).toDateString();
            const matchesSearch = !searchFilter || item.fileName.toLowerCase().includes(searchFilter);

            return matchesStatus && matchesType && matchesDate && matchesSearch;
        });

        tbody.innerHTML = filteredHistory.map(item => {
            const statusBadge = `<span class="badge bg-${this.getStatusColor(item.status)}">${item.status}</span>`;
            const startTime = new Date(item.startTime).toLocaleString();
            const duration = item.duration ? `${Math.floor(item.duration / 60)}:${(item.duration % 60).toString().padStart(2, '0')}` : '--';
            const fileSize = item.fileSize ? this.formatFileSize(item.fileSize) : '--';

            return `
                <tr>
                    <td>
                        <div class="d-flex align-items-center">
                            <i class="fas fa-${this.getFileIcon(item.type)} me-2"></i>
                            ${item.fileName}
                        </div>
                    </td>
                    <td>${this.getTaskNameForType(item.type)}</td>
                    <td>${statusBadge}</td>
                    <td>${startTime}</td>
                    <td>${duration}</td>
                    <td>${fileSize}</td>
                    <td>
                        <button class="btn btn-sm btn-outline-primary me-1" onclick="window.aiContentCreator.retryTask('${item.taskId}')" title="Retry">
                            <i class="fas fa-redo"></i>
                        </button>
                        <button class="btn btn-sm btn-outline-danger" onclick="window.aiContentCreator.removeFromHistory('${item.taskId}')" title="Remove">
                            <i class="fas fa-trash"></i>
                        </button>
                    </td>
                </tr>
            `;
        }).join('');
    }

    getStatusColor(status) {
        const colors = {
            'completed': 'success',
            'failed': 'danger',
            'processing': 'primary',
            'cancelled': 'warning'
        };
        return colors[status] || 'secondary';
    }

    getFileIcon(type) {
        const icons = {
            'upscale': 'video',
            'speech': 'microphone',
            'tts': 'volume-up',
            'detection': 'search'
        };
        return icons[type] || 'file';
    }

    clearProcessingHistory() {
        if (confirm('Are you sure you want to clear all processing history? This action cannot be undone.')) {
            this.processingHistory = [];
            localStorage.removeItem('processingHistory');
            this.loadProcessingHistory();
            this.showSuccess('Processing history cleared successfully.');
        }
    }

    removeFromHistory(taskId) {
        this.processingHistory = this.processingHistory.filter(item => item.taskId !== taskId);
        localStorage.setItem('processingHistory', JSON.stringify(this.processingHistory));
        this.loadProcessingHistory();
    }

    // Settings Management
    loadSettings() {
        // Load settings into UI
        document.getElementById('setting-default-quality').value = this.settings.defaultQuality;
        document.getElementById('setting-concurrent-limit').value = this.settings.concurrentLimit;
        document.getElementById('setting-auto-download').checked = this.settings.autoDownload;
        document.getElementById('setting-notifications').checked = this.settings.notifications;
        document.getElementById('setting-output-format').value = this.settings.outputFormat;
        document.getElementById('setting-size-warning').value = this.settings.sizeWarning;
        document.getElementById('setting-cleanup-temp').checked = this.settings.cleanupTemp;
        document.getElementById('setting-processing-priority').value = this.settings.processingPriority;
        document.getElementById('setting-gpu-usage').value = this.settings.gpuUsage;
        document.getElementById('setting-memory-limit').value = this.settings.memoryLimit;
        document.getElementById('setting-api-timeout').value = this.settings.apiTimeout;
        document.getElementById('setting-retry-attempts').value = this.settings.retryAttempts;
    }

    saveSettings() {
        // Read settings from UI
        this.settings = {
            defaultQuality: document.getElementById('setting-default-quality').value,
            concurrentLimit: parseInt(document.getElementById('setting-concurrent-limit').value),
            autoDownload: document.getElementById('setting-auto-download').checked,
            notifications: document.getElementById('setting-notifications').checked,
            outputFormat: document.getElementById('setting-output-format').value,
            sizeWarning: parseInt(document.getElementById('setting-size-warning').value),
            cleanupTemp: document.getElementById('setting-cleanup-temp').checked,
            processingPriority: document.getElementById('setting-processing-priority').value,
            gpuUsage: document.getElementById('setting-gpu-usage').value,
            memoryLimit: parseInt(document.getElementById('setting-memory-limit').value),
            apiTimeout: parseInt(document.getElementById('setting-api-timeout').value),
            retryAttempts: parseInt(document.getElementById('setting-retry-attempts').value)
        };

        // Save to localStorage
        localStorage.setItem('appSettings', JSON.stringify(this.settings));

        // Close modal
        const modal = bootstrap.Modal.getInstance(document.getElementById('settingsModal'));
        modal.hide();

        this.showSuccess('Settings saved successfully!');
    }

    resetSettings() {
        if (confirm('Are you sure you want to reset all settings to defaults?')) {
            this.settings = this.getDefaultSettings();
            this.loadSettings();
            this.showSuccess('Settings reset to defaults.');
        }
    }

    // Notification support
    showNotification(title, message, type = 'info') {
        if (this.settings.notifications && 'Notification' in window) {
            if (Notification.permission === 'granted') {
                new Notification(title, {
                    body: message,
                    icon: '/static/images/favicon.ico'
                });
            } else if (Notification.permission !== 'denied') {
                Notification.requestPermission().then(permission => {
                    if (permission === 'granted') {
                        new Notification(title, {
                            body: message,
                            icon: '/static/images/favicon.ico'
                        });
                    }
                });
            }
        }
    }

    // Analytics and Performance Tracking
    initializeAnalytics() {
        // Initialize charts when analytics section becomes visible
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting && !this.analytics.chartsInitialized) {
                    this.setupCharts();
                    this.updateAnalytics();
                    this.analytics.chartsInitialized = true;
                }
            });
        });

        const analyticsSection = document.getElementById('analytics');
        if (analyticsSection) {
            observer.observe(analyticsSection);
        }

        // Setup time range change listener
        const timeRangeSelect = document.getElementById('analytics-time-range');
        if (timeRangeSelect) {
            timeRangeSelect.addEventListener('change', () => {
                this.updateAnalytics();
            });
        }

        // Setup chart toggle buttons
        document.querySelectorAll('[data-chart]').forEach(button => {
            button.addEventListener('click', (e) => {
                // Remove active class from siblings
                e.target.parentElement.querySelectorAll('.btn').forEach(btn => {
                    btn.classList.remove('active');
                });
                // Add active class to clicked button
                e.target.classList.add('active');

                // Update chart
                this.updateVolumeChart(e.target.dataset.chart);
            });
        });
    }

    setupCharts() {
        // Processing Volume Chart
        const volumeCtx = document.getElementById('processing-volume-chart');
        if (volumeCtx) {
            this.analytics.charts.volume = new Chart(volumeCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Tasks Processed',
                        data: [],
                        borderColor: 'rgb(102, 126, 234)',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            labels: {
                                color: '#e0e0e0'
                            }
                        }
                    },
                    scales: {
                        x: {
                            type: 'time',
                            time: {
                                unit: 'hour'
                            },
                            ticks: {
                                color: '#e0e0e0'
                            },
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            }
                        },
                        y: {
                            ticks: {
                                color: '#e0e0e0'
                            },
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            }
                        }
                    }
                }
            });
        }

        // Task Distribution Pie Chart
        const distributionCtx = document.getElementById('task-distribution-chart');
        if (distributionCtx) {
            this.analytics.charts.distribution = new Chart(distributionCtx, {
                type: 'doughnut',
                data: {
                    labels: ['Video Upscaling', 'Speech-to-Text', 'Text-to-Speech', 'Product Detection'],
                    datasets: [{
                        data: [0, 0, 0, 0],
                        backgroundColor: [
                            'rgb(102, 126, 234)',
                            'rgb(118, 75, 162)',
                            'rgb(240, 147, 251)',
                            'rgb(134, 239, 172)'
                        ],
                        borderWidth: 2,
                        borderColor: '#1a1a2e'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                color: '#e0e0e0',
                                padding: 20
                            }
                        }
                    }
                }
            });
        }

        // Error Distribution Chart
        const errorCtx = document.getElementById('error-distribution-chart');
        if (errorCtx) {
            this.analytics.charts.errors = new Chart(errorCtx, {
                type: 'bar',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Error Frequency',
                        data: [],
                        backgroundColor: 'rgba(220, 53, 69, 0.8)',
                        borderColor: 'rgb(220, 53, 69)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        x: {
                            ticks: {
                                color: '#e0e0e0',
                                maxRotation: 45
                            },
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            }
                        },
                        y: {
                            ticks: {
                                color: '#e0e0e0'
                            },
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            }
                        }
                    }
                }
            });
        }
    }

    updateAnalytics() {
        const timeRange = document.getElementById('analytics-time-range')?.value || '7d';
        const filteredHistory = this.getFilteredHistory(timeRange);

        this.updateKeyMetrics(filteredHistory);
        this.updatePerformanceTable(filteredHistory);
        this.updateCharts(filteredHistory);
        this.updateRecentActivity();
        this.updateErrorAnalysis(filteredHistory);
    }

    getFilteredHistory(timeRange) {
        const now = new Date();
        let cutoffTime;

        switch (timeRange) {
            case '24h':
                cutoffTime = new Date(now - 24 * 60 * 60 * 1000);
                break;
            case '7d':
                cutoffTime = new Date(now - 7 * 24 * 60 * 60 * 1000);
                break;
            case '30d':
                cutoffTime = new Date(now - 30 * 24 * 60 * 60 * 1000);
                break;
            case 'all':
            default:
                cutoffTime = new Date(0);
                break;
        }

        return this.processingHistory.filter(item =>
            new Date(item.startTime) >= cutoffTime
        );
    }

    updateKeyMetrics(history) {
        const totalTasks = history.length;
        const completedTasks = history.filter(item => item.status === 'completed').length;
        const failedTasks = history.filter(item => item.status === 'failed').length;
        const successRate = totalTasks > 0 ? (completedTasks / totalTasks * 100).toFixed(1) : 0;

        // Calculate total data processed
        const totalDataBytes = history.reduce((sum, item) => sum + (item.fileSize || 0), 0);
        const totalDataGB = (totalDataBytes / (1024 * 1024 * 1024)).toFixed(2);

        // Calculate average duration
        const completedWithDuration = history.filter(item => item.duration && item.status === 'completed');
        const avgDuration = completedWithDuration.length > 0
            ? completedWithDuration.reduce((sum, item) => sum + item.duration, 0) / completedWithDuration.length
            : 0;

        // Update UI
        document.getElementById('total-tasks-count').textContent = totalTasks;
        document.getElementById('success-rate-count').textContent = `${successRate}%`;
        document.getElementById('avg-duration-count').textContent = this.formatDuration(avgDuration);
        document.getElementById('data-processed-count').textContent = `${totalDataGB} GB`;

        // Update progress bars
        document.getElementById('total-tasks-progress').style.width = `${Math.min(totalTasks / 100 * 100, 100)}%`;
        document.getElementById('success-rate-progress').style.width = `${successRate}%`;
        document.getElementById('avg-duration-progress').style.width = `${Math.min(avgDuration / 300 * 100, 100)}%`;
        document.getElementById('data-processed-progress').style.width = `${Math.min(parseFloat(totalDataGB) / 10 * 100, 100)}%`;

        // Update badges with trends (simplified - would need historical comparison in real implementation)
        const trends = this.calculateTrends(history);
        document.getElementById('total-tasks-badge').textContent = trends.tasks;
        document.getElementById('success-rate-badge').textContent = `${successRate}%`;
        document.getElementById('avg-duration-badge').textContent = this.formatDuration(avgDuration);
        document.getElementById('data-processed-badge').textContent = trends.data;
    }

    calculateTrends(history) {
        // Simplified trend calculation - in real implementation, would compare with previous period
        const recentTasks = history.filter(item =>
            new Date(item.startTime) > new Date(Date.now() - 24 * 60 * 60 * 1000)
        );

        return {
            tasks: `+${recentTasks.length}`,
            data: `+${Math.round(recentTasks.reduce((sum, item) => sum + (item.fileSize || 0), 0) / (1024 * 1024))}MB`
        };
    }

    updatePerformanceTable(history) {
        const tbody = document.getElementById('performance-metrics-table');
        if (!tbody) return;

        const types = ['upscale', 'speech', 'tts', 'detection'];
        const typeNames = {
            'upscale': 'Video Upscaling',
            'speech': 'Speech-to-Text',
            'tts': 'Text-to-Speech',
            'detection': 'Product Detection'
        };

        const tableRows = types.map(type => {
            const typeHistory = history.filter(item => item.type === type);
            const totalTasks = typeHistory.length;
            const completedTasks = typeHistory.filter(item => item.status === 'completed').length;
            const successRate = totalTasks > 0 ? (completedTasks / totalTasks * 100).toFixed(1) : 0;

            const completedWithDuration = typeHistory.filter(item => item.duration && item.status === 'completed');
            const avgDuration = completedWithDuration.length > 0
                ? completedWithDuration.reduce((sum, item) => sum + item.duration, 0) / completedWithDuration.length
                : 0;

            const totalData = typeHistory.reduce((sum, item) => sum + (item.fileSize || 0), 0);
            const avgSpeed = avgDuration > 0 && totalData > 0 ? (totalData / (1024 * 1024) / avgDuration).toFixed(2) : 0;

            const trend = this.getTrendIcon(successRate);

            return `
                <tr>
                    <td>${typeNames[type] || type}</td>
                    <td>${totalTasks}</td>
                    <td><span class="badge bg-${successRate >= 80 ? 'success' : successRate >= 60 ? 'warning' : 'danger'}">${successRate}%</span></td>
                    <td>${this.formatDuration(avgDuration)}</td>
                    <td>${avgSpeed} MB/s</td>
                    <td>${this.formatFileSize(totalData)}</td>
                    <td>${trend}</td>
                </tr>
            `;
        }).join('');

        tbody.innerHTML = tableRows;
    }

    getTrendIcon(value) {
        if (value >= 80) {
            return '<i class="fas fa-arrow-up text-success"></i>';
        } else if (value >= 60) {
            return '<i class="fas fa-arrow-right text-warning"></i>';
        } else {
            return '<i class="fas fa-arrow-down text-danger"></i>';
        }
    }

    updateCharts(history) {
        if (!this.analytics.charts.volume || !this.analytics.charts.distribution) return;

        // Update volume chart
        this.updateVolumeChart('volume', history);

        // Update distribution chart
        const typeCount = {
            'upscale': history.filter(h => h.type === 'upscale').length,
            'speech': history.filter(h => h.type === 'speech').length,
            'tts': history.filter(h => h.type === 'tts').length,
            'detection': history.filter(h => h.type === 'detection').length
        };

        this.analytics.charts.distribution.data.datasets[0].data = [
            typeCount.upscale, typeCount.speech, typeCount.tts, typeCount.detection
        ];
        this.analytics.charts.distribution.update();
    }

    updateVolumeChart(chartType, history = null) {
        if (!this.analytics.charts.volume) return;

        history = history || this.getFilteredHistory(document.getElementById('analytics-time-range')?.value || '7d');

        // Group data by time period
        const timeGroups = this.groupByTimePeriod(history, chartType);

        this.analytics.charts.volume.data.labels = timeGroups.labels;
        this.analytics.charts.volume.data.datasets[0].data = timeGroups.data;

        // Update chart label based on type
        const chartLabels = {
            'volume': 'Tasks Processed',
            'duration': 'Processing Time (minutes)',
            'size': 'Data Processed (MB)'
        };

        this.analytics.charts.volume.data.datasets[0].label = chartLabels[chartType];
        this.analytics.charts.volume.update();
    }

    groupByTimePeriod(history, metric) {
        const groups = new Map();
        const timeRange = document.getElementById('analytics-time-range')?.value || '7d';

        // Determine grouping interval
        const interval = timeRange === '24h' ? 'hour' : 'day';

        history.forEach(item => {
            const date = new Date(item.startTime);
            let key;

            if (interval === 'hour') {
                key = new Date(date.getFullYear(), date.getMonth(), date.getDate(), date.getHours());
            } else {
                key = new Date(date.getFullYear(), date.getMonth(), date.getDate());
            }

            const keyStr = key.toISOString();
            if (!groups.has(keyStr)) {
                groups.set(keyStr, { count: 0, duration: 0, size: 0 });
            }

            const group = groups.get(keyStr);
            group.count++;
            group.duration += item.duration || 0;
            group.size += (item.fileSize || 0) / (1024 * 1024); // Convert to MB
        });

        const labels = Array.from(groups.keys()).sort();
        let data;

        switch (metric) {
            case 'duration':
                data = labels.map(label => groups.get(label).duration / 60); // Convert to minutes
                break;
            case 'size':
                data = labels.map(label => groups.get(label).size);
                break;
            case 'volume':
            default:
                data = labels.map(label => groups.get(label).count);
                break;
        }

        return {
            labels: labels.map(label => new Date(label)),
            data: data
        };
    }

    updateRecentActivity() {
        const container = document.getElementById('recent-activity-list');
        if (!container) return;

        const recentItems = this.processingHistory
            .slice(0, 10)
            .map(item => {
                const timeAgo = this.getTimeAgo(new Date(item.startTime));
                const statusColor = this.getStatusColor(item.status);
                const icon = this.getFileIcon(item.type);

                return `
                    <div class="d-flex align-items-center mb-3 p-2 border-start border-${statusColor}" style="border-width: 3px !important;">
                        <i class="fas fa-${icon} me-3 text-${statusColor}"></i>
                        <div class="flex-grow-1">
                            <div class="fw-semibold">${item.fileName}</div>
                            <small class="text-muted">${this.getTaskNameForType(item.type)} • ${timeAgo}</small>
                        </div>
                        <span class="badge bg-${statusColor}">${item.status}</span>
                    </div>
                `;
            }).join('');

        container.innerHTML = recentItems || '<div class="text-center text-muted p-4">No recent activity</div>';
    }

    updateErrorAnalysis(history) {
        const failedTasks = history.filter(item => item.status === 'failed');
        const errorSection = document.getElementById('error-analysis-section');

        if (failedTasks.length === 0) {
            errorSection.style.display = 'none';
            return;
        }

        errorSection.style.display = 'block';

        // Group errors by type (simplified - in real implementation would parse actual error messages)
        const errorTypes = {
            'Network Error': failedTasks.filter(t => Math.random() > 0.7).length,
            'File Format Error': failedTasks.filter(t => Math.random() > 0.8).length,
            'Processing Timeout': failedTasks.filter(t => Math.random() > 0.6).length,
            'Memory Limit': failedTasks.filter(t => Math.random() > 0.9).length,
            'Unknown Error': failedTasks.length - Object.values(errorTypes).reduce((a, b) => a + b, 0)
        };

        // Update error table
        const tbody = document.getElementById('error-analysis-table');
        if (tbody) {
            tbody.innerHTML = Object.entries(errorTypes)
                .filter(([, count]) => count > 0)
                .map(([type, count]) => {
                    const impact = count > 5 ? 'High' : count > 2 ? 'Medium' : 'Low';
                    const impactColor = impact === 'High' ? 'danger' : impact === 'Medium' ? 'warning' : 'success';
                    const lastOccurred = new Date(Math.max(...failedTasks.map(t => new Date(t.startTime)))).toLocaleDateString();

                    return `
                        <tr>
                            <td>${type}</td>
                            <td>${count}</td>
                            <td>${lastOccurred}</td>
                            <td><span class="badge bg-${impactColor}">${impact}</span></td>
                        </tr>
                    `;
                }).join('');
        }

        // Update error chart
        if (this.analytics.charts.errors) {
            const errorLabels = Object.keys(errorTypes).filter(key => errorTypes[key] > 0);
            const errorData = errorLabels.map(key => errorTypes[key]);

            this.analytics.charts.errors.data.labels = errorLabels;
            this.analytics.charts.errors.data.datasets[0].data = errorData;
            this.analytics.charts.errors.update();
        }
    }

    getTimeAgo(date) {
        const now = new Date();
        const diff = now - date;
        const minutes = Math.floor(diff / 60000);
        const hours = Math.floor(diff / 3600000);
        const days = Math.floor(diff / 86400000);

        if (days > 0) return `${days}d ago`;
        if (hours > 0) return `${hours}h ago`;
        if (minutes > 0) return `${minutes}m ago`;
        return 'Just now';
    }

    formatDuration(seconds) {
        if (!seconds || seconds === 0) return '0:00';
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = Math.floor(seconds % 60);
        return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
    }

    startSystemMonitoring() {
        // Simulate system metrics (in real implementation, would call actual system API)
        const updateSystemMetrics = () => {
            const metrics = {
                cpu: Math.random() * 100,
                memory: Math.random() * 100,
                gpu: Math.random() * 100,
                disk: Math.random() * 100
            };

            Object.entries(metrics).forEach(([type, value]) => {
                const element = document.getElementById(`${type}-usage`);
                const bar = document.getElementById(`${type}-usage-bar`);

                if (element) element.textContent = `${Math.round(value)}%`;
                if (bar) bar.style.width = `${value}%`;
            });
        };

        // Update metrics every 5 seconds
        this.analytics.updateInterval = setInterval(updateSystemMetrics, 5000);
        updateSystemMetrics(); // Initial update
    }

    // Export functionality
    exportReport() {
        const timeRange = document.getElementById('analytics-time-range')?.value || '7d';
        const history = this.getFilteredHistory(timeRange);

        const report = {
            generatedAt: new Date().toISOString(),
            timeRange: timeRange,
            summary: {
                totalTasks: history.length,
                completedTasks: history.filter(h => h.status === 'completed').length,
                failedTasks: history.filter(h => h.status === 'failed').length,
                successRate: history.length > 0 ? (history.filter(h => h.status === 'completed').length / history.length * 100).toFixed(2) : 0,
                totalDataProcessed: history.reduce((sum, h) => sum + (h.fileSize || 0), 0),
                averageDuration: history.filter(h => h.duration).reduce((sum, h, _, arr) => sum + h.duration / arr.length, 0)
            },
            taskBreakdown: {
                upscale: history.filter(h => h.type === 'upscale').length,
                speech: history.filter(h => h.type === 'speech').length,
                tts: history.filter(h => h.type === 'tts').length,
                detection: history.filter(h => h.type === 'detection').length
            },
            detailedHistory: history
        };

        // Create and download CSV
        this.downloadCSV(report);
    }

    downloadCSV(report) {
        const csvContent = [
            ['AI Content Creator - Analytics Report'],
            [`Generated: ${new Date(report.generatedAt).toLocaleString()}`],
            [`Time Range: ${report.timeRange}`],
            [''],
            ['Summary'],
            [`Total Tasks,${report.summary.totalTasks}`],
            [`Completed Tasks,${report.summary.completedTasks}`],
            [`Failed Tasks,${report.summary.failedTasks}`],
            [`Success Rate,${report.summary.successRate}%`],
            [`Total Data Processed,${this.formatFileSize(report.summary.totalDataProcessed)}`],
            [`Average Duration,${this.formatDuration(report.summary.averageDuration)}`],
            [''],
            ['Task Breakdown'],
            [`Video Upscaling,${report.taskBreakdown.upscale}`],
            [`Speech-to-Text,${report.taskBreakdown.speech}`],
            [`Text-to-Speech,${report.taskBreakdown.tts}`],
            [`Product Detection,${report.taskBreakdown.detection}`],
            [''],
            ['Detailed History'],
            ['File Name,Type,Status,Start Time,Duration,File Size']
        ];

        // Add detailed history
        report.detailedHistory.forEach(item => {
            csvContent.push([
                item.fileName,
                this.getTaskNameForType(item.type),
                item.status,
                new Date(item.startTime).toLocaleString(),
                this.formatDuration(item.duration || 0),
                this.formatFileSize(item.fileSize || 0)
            ]);
        });

        const csv = csvContent.map(row => row.join(',')).join('\n');
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `ai-content-analytics-${report.timeRange}-${new Date().toISOString().split('T')[0]}.csv`;
        a.click();
        window.URL.revokeObjectURL(url);

        this.showSuccess('Analytics report exported successfully!');
    }

    // Enhanced history tracking
    addToHistory(taskId, fileName, type, status, startTime, endTime = null, fileSize = null) {
        const historyItem = {
            taskId: taskId,
            fileName: fileName,
            type: type,
            status: status,
            startTime: startTime,
            endTime: endTime,
            fileSize: fileSize,
            duration: endTime ? Math.round((endTime - startTime) / 1000) : null
        };

        // Add to beginning of history
        this.processingHistory.unshift(historyItem);

        // Limit history to 100 items
        if (this.processingHistory.length > 100) {
            this.processingHistory = this.processingHistory.slice(0, 100);
        }

        // Save to localStorage
        localStorage.setItem('processingHistory', JSON.stringify(this.processingHistory));

        // Refresh displays
        this.loadProcessingHistory();
        if (this.analytics.chartsInitialized) {
            this.updateAnalytics();
        }
    }

    // Demo data generation for showcasing analytics
    generateDemoData() {
        // Only generate demo data if history is empty
        if (this.processingHistory.length > 0) {
            return;
        }

        const types = ['upscale', 'speech', 'tts', 'detection'];
        const statuses = ['completed', 'failed', 'completed', 'completed', 'completed']; // 80% success rate
        const fileNames = [
            'product_demo.mp4', 'presentation.mp4', 'interview.wav', 'tutorial.mp4',
            'marketing_video.mp4', 'podcast_episode.mp3', 'webinar.mp4', 'testimonial.mp4',
            'advertisement.mp4', 'training_video.mp4', 'conference_call.wav', 'explainer.mp4',
            'brand_video.mp4', 'customer_review.mp4', 'sales_pitch.mp3', 'demo_reel.mp4'
        ];

        const now = new Date();

        // Generate 30 days worth of sample data
        for (let i = 0; i < 50; i++) {
            const daysAgo = Math.floor(Math.random() * 30);
            const hoursAgo = Math.floor(Math.random() * 24);
            const minutesAgo = Math.floor(Math.random() * 60);

            const startTime = new Date(now);
            startTime.setDate(startTime.getDate() - daysAgo);
            startTime.setHours(startTime.getHours() - hoursAgo);
            startTime.setMinutes(startTime.getMinutes() - minutesAgo);

            const endTime = new Date(startTime);
            const processingTime = Math.floor(Math.random() * 600) + 30; // 30 seconds to 10 minutes
            endTime.setSeconds(endTime.getSeconds() + processingTime);

            const type = types[Math.floor(Math.random() * types.length)];
            const status = statuses[Math.floor(Math.random() * statuses.length)];
            const fileName = fileNames[Math.floor(Math.random() * fileNames.length)];
            const fileSize = Math.floor(Math.random() * 500 * 1024 * 1024) + 10 * 1024 * 1024; // 10MB to 500MB

            const historyItem = {
                taskId: `demo-${i}-${Date.now()}`,
                fileName: fileName,
                type: type,
                status: status,
                startTime: startTime.getTime(),
                endTime: status === 'completed' ? endTime.getTime() : null,
                fileSize: fileSize,
                duration: status === 'completed' ? processingTime : null
            };

            this.processingHistory.push(historyItem);
        }

        // Sort by start time (newest first)
        this.processingHistory.sort((a, b) => b.startTime - a.startTime);

        // Save to localStorage
        localStorage.setItem('processingHistory', JSON.stringify(this.processingHistory));

        // Add a visual indicator that this is demo data
        const demoIndicator = document.createElement('div');
        demoIndicator.className = 'alert alert-info alert-dismissible fade show position-fixed';
        demoIndicator.style.cssText = 'top: 80px; right: 20px; z-index: 9999; max-width: 350px;';
        demoIndicator.innerHTML = `
            <i class="fas fa-info-circle me-2"></i>
            <strong>Demo Mode:</strong> Sample analytics data has been generated for demonstration.
            Real data will replace this as you use the platform.
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        document.body.appendChild(demoIndicator);

        // Auto-dismiss after 10 seconds
        setTimeout(() => {
            if (demoIndicator.parentElement) {
                demoIndicator.remove();
            }
        }, 10000);
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

function cancelTask(taskId) {
    window.aiContentCreator.cancelTask(taskId);
}

function refreshHistory() {
    window.aiContentCreator.loadProcessingHistory();
}

function clearHistory() {
    window.aiContentCreator.clearProcessingHistory();
}

function saveSettings() {
    window.aiContentCreator.saveSettings();
}

function resetSettings() {
    window.aiContentCreator.resetSettings();
}

function refreshAnalytics() {
    window.aiContentCreator.updateAnalytics();
}

function exportReport() {
    window.aiContentCreator.exportReport();
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