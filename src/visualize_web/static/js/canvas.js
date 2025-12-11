/**
 * CanvasRenderer handles canvas rendering and mouse interactions.
 */

class CanvasRenderer {
    constructor(canvasElement, boxManager) {
        this.canvas = canvasElement;
        this.ctx = this.canvas.getContext('2d');
        this.boxManager = boxManager;

        // State
        this.image = null;
        this.scale = 1.0;
        this.offset = {x: 0, y: 0};

        // Interaction state
        this.dragMode = null;  // null, 'draw', 'move', 'resize'
        this.dragStart = null;
        this.dragTarget = null;
        this.hoverBoxId = null;
        this.hoverHandle = null;

        // Callbacks
        this.onBoxChange = null;

        this.setupEventListeners();
    }

    /**
     * Set up event listeners for mouse interactions.
     */
    setupEventListeners() {
        this.canvas.addEventListener('mousedown', this.handleMouseDown.bind(this));
        this.canvas.addEventListener('mousemove', this.handleMouseMove.bind(this));
        this.canvas.addEventListener('mouseup', this.handleMouseUp.bind(this));
        this.canvas.addEventListener('mouseleave', this.handleMouseLeave.bind(this));

        // Keyboard events
        document.addEventListener('keydown', this.handleKeyDown.bind(this));
    }

    /**
     * Load and display an image.
     * @param {string} imageUrl - URL of the image
     * @returns {Promise<void>}
     */
    async loadImage(imageUrl) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => {
                this.image = img;
                this.fitToScreen();
                this.render();
                resolve();
            };
            img.onerror = reject;
            img.src = imageUrl;
        });
    }

    /**
     * Fit image to screen.
     */
    fitToScreen() {
        if (!this.image) return;

        const containerWidth = this.canvas.parentElement.clientWidth - 40;
        const containerHeight = this.canvas.parentElement.clientHeight - 40;

        const scaleX = containerWidth / this.image.width;
        const scaleY = containerHeight / this.image.height;

        this.scale = Math.min(scaleX, scaleY, 1);
        this.canvas.width = this.image.width * this.scale;
        this.canvas.height = this.image.height * this.scale;
        this.offset = {x: 0, y: 0};

        this.render();
        this.updateZoomDisplay();
    }

    /**
     * Zoom in.
     */
    zoomIn() {
        this.scale *= 1.2;
        this.canvas.width = this.image.width * this.scale;
        this.canvas.height = this.image.height * this.scale;
        this.render();
        this.updateZoomDisplay();
    }

    /**
     * Zoom out.
     */
    zoomOut() {
        this.scale *= 0.8;
        this.canvas.width = this.image.width * this.scale;
        this.canvas.height = this.image.height * this.scale;
        this.render();
        this.updateZoomDisplay();
    }

    /**
     * Update zoom level display.
     */
    updateZoomDisplay() {
        const zoomLevel = document.getElementById('zoomLevel');
        if (zoomLevel) {
            zoomLevel.textContent = `${Math.round(this.scale * 100)}%`;
        }
    }

    /**
     * Render the canvas.
     */
    render() {
        if (!this.image) return;

        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw image
        this.ctx.drawImage(
            this.image,
            0, 0,
            this.image.width, this.image.height,
            0, 0,
            this.canvas.width, this.canvas.height
        );

        // Draw all boxes
        const boxes = this.boxManager.getAllBoxes();
        boxes.forEach(box => {
            this.drawBox(box);
        });

        // Draw current drag box (if drawing)
        if (this.dragMode === 'draw' && this.dragStart && this.dragTarget) {
            this.drawDragBox(this.dragStart, this.dragTarget);
        }
    }

    /**
     * Draw a bounding box.
     * @param {Object} box - Box object
     */
    drawBox(box) {
        const selected = box.id === this.boxManager.selectedBoxId;
        const hover = box.id === this.hoverBoxId;

        // Convert image coordinates to canvas coordinates
        const x = box.bbox.x * this.scale;
        const y = box.bbox.y * this.scale;
        const width = box.bbox.width * this.scale;
        const height = box.bbox.height * this.scale;

        // Set style based on box state
        this.ctx.strokeStyle = selected ? '#007bff' : (hover ? '#28a745' : (box.isNew ? '#28a745' : (box.isModified ? '#ffc107' : '#dc3545')));
        this.ctx.lineWidth = selected ? 3 : 2;

        // Draw rectangle
        this.ctx.strokeRect(x, y, width, height);

        // Draw label background
        if (box.transcription) {
            const labelText = box.transcription;
            const fontSize = 14;
            this.ctx.font = `${fontSize}px Arial`;
            const textWidth = this.ctx.measureText(labelText).width;

            this.ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
            this.ctx.fillRect(x, y - fontSize - 4, textWidth + 8, fontSize + 4);

            this.ctx.fillStyle = 'white';
            this.ctx.fillText(labelText, x + 4, y - 4);
        }

        // Draw resize handles if selected
        if (selected) {
            this.drawResizeHandles(x, y, width, height);
        }
    }

    /**
     * Draw resize handles for selected box.
     * @param {number} x - Box x
     * @param {number} y - Box y
     * @param {number} width - Box width
     * @param {number} height - Box height
     */
    drawResizeHandles(x, y, width, height) {
        const handleSize = 8;
        this.ctx.fillStyle = '#007bff';

        // Corner handles
        const corners = [
            {x: x, y: y},                      // nw
            {x: x + width, y: y},              // ne
            {x: x + width, y: y + height},     // se
            {x: x, y: y + height}              // sw
        ];

        corners.forEach(corner => {
            this.ctx.fillRect(
                corner.x - handleSize / 2,
                corner.y - handleSize / 2,
                handleSize,
                handleSize
            );
        });

        // Edge handles
        const edges = [
            {x: x + width / 2, y: y},              // n
            {x: x + width, y: y + height / 2},     // e
            {x: x + width / 2, y: y + height},     // s
            {x: x, y: y + height / 2}              // w
        ];

        edges.forEach(edge => {
            this.ctx.fillRect(
                edge.x - handleSize / 2,
                edge.y - handleSize / 2,
                handleSize,
                handleSize
            );
        });
    }

    /**
     * Draw the box being dragged during creation.
     * @param {Object} start - Start point {x, y}
     * @param {Object} end - End point {x, y}
     */
    drawDragBox(start, end) {
        const x = Math.min(start.x, end.x);
        const y = Math.min(start.y, end.y);
        const width = Math.abs(end.x - start.x);
        const height = Math.abs(end.y - start.y);

        this.ctx.strokeStyle = '#28a745';
        this.ctx.lineWidth = 2;
        this.ctx.setLineDash([5, 5]);
        this.ctx.strokeRect(x, y, width, height);
        this.ctx.setLineDash([]);
    }

    /**
     * Handle mouse down event.
     * @param {MouseEvent} e - Mouse event
     */
    handleMouseDown(e) {
        const canvasPos = this.getCanvasCoords(e);
        const imagePos = this.canvasToImage(canvasPos.x, canvasPos.y);

        const selectedBox = this.boxManager.getSelectedBox();

        // Check if clicking on resize handle of selected box
        if (selectedBox) {
            const handle = this.getResizeHandle(selectedBox.id, imagePos.x, imagePos.y);
            if (handle) {
                this.dragMode = 'resize';
                this.dragStart = {...imagePos};
                this.dragTarget = {
                    boxId: selectedBox.id,
                    handle: handle,
                    originalBbox: {...selectedBox.bbox}
                };
                return;
            }
        }

        // Check if clicking inside a box
        const clickedBoxId = this.getBoxAtPoint(imagePos.x, imagePos.y);

        if (clickedBoxId !== null) {
            this.boxManager.selectBox(clickedBoxId);
            this.dragMode = 'move';
            this.dragStart = {...imagePos};
            this.dragTarget = {
                boxId: clickedBoxId,
                originalBbox: {...this.boxManager.getBox(clickedBoxId).bbox}
            };
            this.render();
            return;
        }

        // Start drawing new box
        this.boxManager.deselectBox();
        this.dragMode = 'draw';
        this.dragStart = {...imagePos};
        this.dragTarget = {...imagePos};
        this.render();
    }

    /**
     * Handle mouse move event.
     * @param {MouseEvent} e - Mouse event
     */
    handleMouseMove(e) {
        const canvasPos = this.getCanvasCoords(e);
        const imagePos = this.canvasToImage(canvasPos.x, canvasPos.y);

        // Update hover state
        const hoverBoxId = this.getBoxAtPoint(imagePos.x, imagePos.y);
        if (hoverBoxId !== this.hoverBoxId) {
            this.hoverBoxId = hoverBoxId;
            if (!this.dragMode) {
                this.render();
            }
        }

        // Handle dragging
        if (this.dragMode === 'draw') {
            this.dragTarget = {...imagePos};
            this.render();
        } else if (this.dragMode === 'move') {
            const dx = imagePos.x - this.dragStart.x;
            const dy = imagePos.y - this.dragStart.y;

            const newBbox = {
                x: this.dragTarget.originalBbox.x + dx,
                y: this.dragTarget.originalBbox.y + dy,
                width: this.dragTarget.originalBbox.width,
                height: this.dragTarget.originalBbox.height
            };

            this.boxManager.updateBox(this.dragTarget.boxId, newBbox);
            this.render();
        } else if (this.dragMode === 'resize') {
            this.handleResize(imagePos);
            this.render();
        } else {
            // Update cursor based on hover state
            const selectedBox = this.boxManager.getSelectedBox();
            if (selectedBox) {
                const handle = this.getResizeHandle(selectedBox.id, imagePos.x, imagePos.y);
                this.canvas.style.cursor = this.getCursorForHandle(handle);
            } else if (hoverBoxId !== null) {
                this.canvas.style.cursor = 'pointer';
            } else {
                this.canvas.style.cursor = 'crosshair';
            }
        }
    }

    /**
     * Handle mouse up event.
     * @param {MouseEvent} e - Mouse event
     */
    handleMouseUp(e) {
        if (this.dragMode === 'draw') {
            const canvasPos = this.getCanvasCoords(e);
            const imagePos = this.canvasToImage(canvasPos.x, canvasPos.y);

            const x = Math.min(this.dragStart.x, imagePos.x);
            const y = Math.min(this.dragStart.y, imagePos.y);
            const width = Math.abs(imagePos.x - this.dragStart.x);
            const height = Math.abs(imagePos.y - this.dragStart.y);

            // Only add box if it has meaningful size
            if (width > 5 && height > 5) {
                const boxId = this.boxManager.addBox({x, y, width, height});
                this.boxManager.selectBox(boxId);

                if (this.onBoxChange) {
                    this.onBoxChange();
                }
            }
        } else if (this.dragMode === 'move' || this.dragMode === 'resize') {
            if (this.onBoxChange) {
                this.onBoxChange();
            }
        }

        this.dragMode = null;
        this.dragStart = null;
        this.dragTarget = null;
        this.render();
    }

    /**
     * Handle mouse leave event.
     */
    handleMouseLeave() {
        this.hoverBoxId = null;
        this.render();
    }

    /**
     * Handle key down event.
     * @param {KeyboardEvent} e - Keyboard event
     */
    handleKeyDown(e) {
        if (e.key === 'Delete' || e.key === 'Backspace') {
            const selectedBox = this.boxManager.getSelectedBox();
            if (selectedBox) {
                this.boxManager.deleteBox(selectedBox.id);
                this.render();
                if (this.onBoxChange) {
                    this.onBoxChange();
                }
                e.preventDefault();
            }
        } else if (e.key === 'Escape') {
            this.boxManager.deselectBox();
            this.dragMode = null;
            this.dragStart = null;
            this.dragTarget = null;
            this.render();
        }
    }

    /**
     * Handle resize operation.
     * @param {Object} imagePos - Current mouse position in image coordinates
     */
    handleResize(imagePos) {
        const box = this.boxManager.getBox(this.dragTarget.boxId);
        const handle = this.dragTarget.handle;
        const original = this.dragTarget.originalBbox;

        const dx = imagePos.x - this.dragStart.x;
        const dy = imagePos.y - this.dragStart.y;

        let newBbox = {...original};

        switch (handle) {
            case 'nw':
                newBbox.x = original.x + dx;
                newBbox.y = original.y + dy;
                newBbox.width = original.width - dx;
                newBbox.height = original.height - dy;
                break;
            case 'ne':
                newBbox.y = original.y + dy;
                newBbox.width = original.width + dx;
                newBbox.height = original.height - dy;
                break;
            case 'se':
                newBbox.width = original.width + dx;
                newBbox.height = original.height + dy;
                break;
            case 'sw':
                newBbox.x = original.x + dx;
                newBbox.width = original.width - dx;
                newBbox.height = original.height + dy;
                break;
            case 'n':
                newBbox.y = original.y + dy;
                newBbox.height = original.height - dy;
                break;
            case 's':
                newBbox.height = original.height + dy;
                break;
            case 'e':
                newBbox.width = original.width + dx;
                break;
            case 'w':
                newBbox.x = original.x + dx;
                newBbox.width = original.width - dx;
                break;
        }

        // Ensure minimum size
        if (newBbox.width < 10) newBbox.width = 10;
        if (newBbox.height < 10) newBbox.height = 10;

        this.boxManager.updateBox(this.dragTarget.boxId, newBbox);
    }

    /**
     * Get resize handle at a given point.
     * @param {number} boxId - Box ID
     * @param {number} x - X coordinate in image space
     * @param {number} y - Y coordinate in image space
     * @returns {string|null} Handle type or null
     */
    getResizeHandle(boxId, x, y) {
        const box = this.boxManager.getBox(boxId);
        if (!box) return null;

        const threshold = 10 / this.scale;  // Threshold in image space
        const bbox = box.bbox;

        // Check corners first
        if (Math.abs(x - bbox.x) < threshold && Math.abs(y - bbox.y) < threshold) return 'nw';
        if (Math.abs(x - (bbox.x + bbox.width)) < threshold && Math.abs(y - bbox.y) < threshold) return 'ne';
        if (Math.abs(x - (bbox.x + bbox.width)) < threshold && Math.abs(y - (bbox.y + bbox.height)) < threshold) return 'se';
        if (Math.abs(x - bbox.x) < threshold && Math.abs(y - (bbox.y + bbox.height)) < threshold) return 'sw';

        // Check edges
        if (Math.abs(x - (bbox.x + bbox.width / 2)) < threshold && Math.abs(y - bbox.y) < threshold) return 'n';
        if (Math.abs(x - (bbox.x + bbox.width)) < threshold && Math.abs(y - (bbox.y + bbox.height / 2)) < threshold) return 'e';
        if (Math.abs(x - (bbox.x + bbox.width / 2)) < threshold && Math.abs(y - (bbox.y + bbox.height)) < threshold) return 's';
        if (Math.abs(x - bbox.x) < threshold && Math.abs(y - (bbox.y + bbox.height / 2)) < threshold) return 'w';

        return null;
    }

    /**
     * Get cursor style for a resize handle.
     * @param {string|null} handle - Handle type
     * @returns {string} CSS cursor value
     */
    getCursorForHandle(handle) {
        const cursors = {
            'nw': 'nw-resize',
            'ne': 'ne-resize',
            'se': 'se-resize',
            'sw': 'sw-resize',
            'n': 'n-resize',
            's': 's-resize',
            'e': 'e-resize',
            'w': 'w-resize'
        };
        return cursors[handle] || 'crosshair';
    }

    /**
     * Get box at a given point.
     * @param {number} x - X coordinate in image space
     * @param {number} y - Y coordinate in image space
     * @returns {number|null} Box ID or null
     */
    getBoxAtPoint(x, y) {
        const boxes = this.boxManager.getAllBoxes();

        // Iterate in reverse to prioritize topmost boxes
        for (let i = boxes.length - 1; i >= 0; i--) {
            const box = boxes[i];
            const bbox = box.bbox;

            if (x >= bbox.x && x <= bbox.x + bbox.width &&
                y >= bbox.y && y <= bbox.y + bbox.height) {
                return box.id;
            }
        }

        return null;
    }

    /**
     * Get canvas coordinates from mouse event.
     * @param {MouseEvent} e - Mouse event
     * @returns {Object} Canvas coordinates {x, y}
     */
    getCanvasCoords(e) {
        const rect = this.canvas.getBoundingClientRect();
        return {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        };
    }

    /**
     * Convert canvas coordinates to image coordinates.
     * @param {number} canvasX - Canvas X coordinate
     * @param {number} canvasY - Canvas Y coordinate
     * @returns {Object} Image coordinates {x, y}
     */
    canvasToImage(canvasX, canvasY) {
        return {
            x: canvasX / this.scale,
            y: canvasY / this.scale
        };
    }

    /**
     * Convert image coordinates to canvas coordinates.
     * @param {number} imageX - Image X coordinate
     * @param {number} imageY - Image Y coordinate
     * @returns {Object} Canvas coordinates {x, y}
     */
    imageToCanvas(imageX, imageY) {
        return {
            x: imageX * this.scale,
            y: imageY * this.scale
        };
    }
}
