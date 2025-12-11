/**
 * Main application controller.
 */

class App {
    constructor() {
        this.api = new API();
        this.boxManager = new BoxManager();
        this.canvas = new CanvasRenderer(
            document.getElementById('annotationCanvas'),
            this.boxManager
        );

        this.currentImageIndex = 0;
        this.images = [];
        this.allAnnotations = {};

        this.setupEventListeners();
        this.updateUI();
    }

    /**
     * Set up event listeners for UI controls.
     */
    setupEventListeners() {
        // Load button
        document.getElementById('loadBtn').addEventListener('click', () => {
            const dir = document.getElementById('imageDir').value.trim();
            if (dir) {
                this.loadDirectory(dir);
            } else {
                this.showStatus('Please enter a valid IMAGE_DIR path', 'error');
            }
        });

        // Navigation buttons
        document.getElementById('prevBtn').addEventListener('click', () => this.prevImage());
        document.getElementById('nextBtn').addEventListener('click', () => this.nextImage());
        document.getElementById('saveBtn').addEventListener('click', () => this.saveCurrentImage());

        // Zoom controls
        document.getElementById('zoomIn').addEventListener('click', () => {
            this.canvas.zoomIn();
        });
        document.getElementById('zoomOut').addEventListener('click', () => {
            this.canvas.zoomOut();
        });
        document.getElementById('fitToScreen').addEventListener('click', () => {
            this.canvas.fitToScreen();
        });

        // Listen to canvas changes to update Save button
        this.canvas.onBoxChange = () => {
            this.updateSaveButton();
            this.updateBoxList();
        };

        // Arrow key navigation
        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowLeft') {
                this.prevImage();
                e.preventDefault();
            } else if (e.key === 'ArrowRight') {
                this.nextImage();
                e.preventDefault();
            }
        });

        // Allow Enter key in input to load
        document.getElementById('imageDir').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                document.getElementById('loadBtn').click();
            }
        });
    }

    /**
     * Load image directory and annotations.
     * @param {string} dirPath - Path to IMAGE_DIR
     */
    async loadDirectory(dirPath) {
        try {
            this.showStatus('Loading images...', 'info');

            const data = await this.api.loadImageDirectory(dirPath);

            this.images = data.images;
            this.allAnnotations = data.annotations;
            this.currentImageIndex = 0;

            if (this.images.length === 0) {
                this.showStatus('No images found in directory', 'error');
                return;
            }

            await this.loadImage(0);
            this.showStatus(`Loaded ${this.images.length} images`, 'success');

        } catch (error) {
            this.showStatus(`Error: ${error.message}`, 'error');
            console.error('Error loading directory:', error);
        }
    }

    /**
     * Load a specific image.
     * @param {number} index - Image index
     */
    async loadImage(index) {
        if (index < 0 || index >= this.images.length) {
            return;
        }

        try {
            this.currentImageIndex = index;
            const filename = this.images[index];
            const imageUrl = this.api.getImageUrl(filename);
            const annotations = this.allAnnotations[filename] || [];

            // Load annotations
            this.boxManager.loadBoxes(annotations);

            // Load image
            await this.canvas.loadImage(imageUrl);

            // Update UI
            this.updateUI();
            this.updateBoxList();

            this.showStatus(`Loaded image ${index + 1} of ${this.images.length}`, 'success');

        } catch (error) {
            this.showStatus(`Error loading image: ${error.message}`, 'error');
            console.error('Error loading image:', error);
        }
    }

    /**
     * Save current image annotations.
     */
    async saveCurrentImage() {
        try {
            this.showStatus('Saving...', 'info');

            const filename = this.images[this.currentImageIndex];
            const annotations = this.boxManager.exportToJSON();

            await this.api.saveAnnotations(filename, annotations);

            // Update local annotations
            this.allAnnotations[filename] = annotations;

            // Mark as saved
            this.boxManager.markAsSaved();

            this.updateSaveButton();
            this.updateBoxList();
            this.showStatus('Saved successfully', 'success');

        } catch (error) {
            this.showStatus(`Error saving: ${error.message}`, 'error');
            console.error('Error saving annotations:', error);
        }
    }

    /**
     * Navigate to previous image.
     */
    async prevImage() {
        if (this.currentImageIndex > 0) {
            if (this.boxManager.hasChanges()) {
                const confirmed = confirm('You have unsaved changes. Save before navigating?');
                if (confirmed) {
                    await this.saveCurrentImage();
                }
            }
            await this.loadImage(this.currentImageIndex - 1);
        }
    }

    /**
     * Navigate to next image.
     */
    async nextImage() {
        if (this.currentImageIndex < this.images.length - 1) {
            if (this.boxManager.hasChanges()) {
                const confirmed = confirm('You have unsaved changes. Save before navigating?');
                if (confirmed) {
                    await this.saveCurrentImage();
                }
            }
            await this.loadImage(this.currentImageIndex + 1);
        }
    }

    /**
     * Update the Save button state.
     */
    updateSaveButton() {
        const saveBtn = document.getElementById('saveBtn');
        const hasChanges = this.boxManager.hasChanges();

        saveBtn.disabled = !hasChanges;
        saveBtn.style.opacity = hasChanges ? '1' : '0.5';
    }

    /**
     * Update the box list panel.
     */
    updateBoxList() {
        const listContainer = document.getElementById('boxListContainer');
        const boxes = this.boxManager.getAllBoxes();

        listContainer.innerHTML = '';

        boxes.forEach((box, idx) => {
            const li = document.createElement('li');
            li.textContent = `Box ${idx + 1}: ${box.transcription || '(empty)'}`;

            if (box.isNew) {
                li.classList.add('new-box');
            } else if (box.isModified) {
                li.classList.add('modified-box');
            }

            if (box.id === this.boxManager.selectedBoxId) {
                li.classList.add('selected');
            }

            li.addEventListener('click', () => {
                this.boxManager.selectBox(box.id);
                this.canvas.render();
                this.updateBoxList();
            });

            listContainer.appendChild(li);
        });
    }

    /**
     * Update the UI state.
     */
    updateUI() {
        // Update image info
        if (this.images.length > 0) {
            document.getElementById('imageName').textContent = this.images[this.currentImageIndex];
            document.getElementById('imageIndex').textContent =
                `${this.currentImageIndex + 1} / ${this.images.length}`;
        } else {
            document.getElementById('imageName').textContent = '-';
            document.getElementById('imageIndex').textContent = '0 / 0';
        }

        // Update box count
        document.getElementById('boxCount').textContent = this.boxManager.boxes.length;

        // Update navigation buttons
        document.getElementById('prevBtn').disabled = this.currentImageIndex === 0;
        document.getElementById('nextBtn').disabled =
            this.currentImageIndex === this.images.length - 1 || this.images.length === 0;

        // Update save button
        this.updateSaveButton();
    }

    /**
     * Show status message.
     * @param {string} message - Status message
     * @param {string} type - Message type ('info', 'success', 'error')
     */
    showStatus(message, type = 'info') {
        const statusElement = document.getElementById('statusMessage');
        statusElement.textContent = message;

        // Change color based on type
        switch (type) {
            case 'success':
                statusElement.style.color = '#28a745';
                break;
            case 'error':
                statusElement.style.color = '#dc3545';
                break;
            case 'info':
            default:
                statusElement.style.color = 'white';
                break;
        }
    }
}

// Initialize application when DOM is ready
document.addEventListener('DOMContentLoaded', async () => {
    globalThis.app = new App();

    // Auto-load from default IMAGE_DIR
    try {
        await globalThis.app.loadDirectory('');
    } catch (error) {
        console.log('Could not auto-load IMAGE_DIR:', error);
    }
});
