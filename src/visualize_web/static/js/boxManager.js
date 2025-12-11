/**
 * BoxManager handles state management for bounding boxes.
 */

class BoxManager {
    constructor() {
        this.boxes = [];              // Current boxes (flat array)
        this.originalBoxes = [];      // Loaded boxes (for change detection)
        this.selectedBoxId = null;
        this.nextId = 0;
    }

    /**
     * Load boxes from annotations data.
     * @param {Array} annotations - Array of annotations from backend
     */
    loadBoxes(annotations) {
        // annotations is a flat array (backend already flattened comment + content)
        // Convert from JSON two-point format to internal {x, y, width, height} format
        // JSON format: bbox = [[x_min, y_min], [x_max, y_max]]
        // Internal format: bbox = {x, y, width, height}

        this.boxes = annotations.map((ann, idx) => ({
            id: idx,
            bbox: {
                x: ann.bbox[0][0],
                y: ann.bbox[0][1],
                width: ann.bbox[1][0] - ann.bbox[0][0],
                height: ann.bbox[1][1] - ann.bbox[0][1]
            },
            transcription: ann.transcription,
            difficult: ann.difficult,
            confidence: ann.confidence,
            isNew: false,
            isModified: false,
            originalBbox: null
        }));

        this.originalBoxes = JSON.parse(JSON.stringify(this.boxes));
        this.nextId = this.boxes.length;
        this.selectedBoxId = null;
    }

    /**
     * Add a new bounding box.
     * @param {Object} bbox - Bounding box {x, y, width, height}
     * @returns {number} ID of the new box
     */
    addBox(bbox) {
        const box = {
            id: this.nextId++,
            bbox: bbox,
            transcription: "",
            difficult: false,
            confidence: 0,
            isNew: true,
            isModified: false,
            originalBbox: null
        };
        this.boxes.push(box);
        return box.id;
    }

    /**
     * Update a bounding box.
     * @param {number} id - Box ID
     * @param {Object} bbox - New bounding box {x, y, width, height}
     */
    updateBox(id, bbox) {
        const box = this.boxes.find(b => b.id === id);
        if (box) {
            box.bbox = bbox;
            if (!box.isNew) {
                box.isModified = true;
            }
        }
    }

    /**
     * Delete a bounding box.
     * @param {number} id - Box ID
     */
    deleteBox(id) {
        this.boxes = this.boxes.filter(b => b.id !== id);
        if (this.selectedBoxId === id) {
            this.selectedBoxId = null;
        }
    }

    /**
     * Select a bounding box.
     * @param {number} id - Box ID
     */
    selectBox(id) {
        this.selectedBoxId = id;
    }

    /**
     * Deselect the current box.
     */
    deselectBox() {
        this.selectedBoxId = null;
    }

    /**
     * Get the selected box.
     * @returns {Object|null} The selected box or null
     */
    getSelectedBox() {
        if (this.selectedBoxId === null) return null;
        return this.boxes.find(b => b.id === this.selectedBoxId);
    }

    /**
     * Get a box by ID.
     * @param {number} id - Box ID
     * @returns {Object|null} The box or null
     */
    getBox(id) {
        return this.boxes.find(b => b.id === id);
    }

    /**
     * Check if there are unsaved changes.
     * @returns {boolean} True if there are changes
     */
    hasChanges() {
        if (this.boxes.length !== this.originalBoxes.length) {
            return true;
        }

        return this.boxes.some(box => box.isNew || box.isModified);
    }

    /**
     * Export boxes to JSON format.
     * @returns {Array} Array of annotations in two-point format
     */
    exportToJSON() {
        // Convert internal {x, y, width, height} format back to two-point format
        return this.boxes.map(box => ({
            transcription: box.transcription,
            bbox: [
                [box.bbox.x, box.bbox.y],  // Top-left
                [box.bbox.x + box.bbox.width, box.bbox.y + box.bbox.height]  // Bottom-right
            ],
            difficult: box.difficult,
            confidence: box.confidence
        }));
    }

    /**
     * Mark all boxes as saved (reset change flags).
     */
    markAsSaved() {
        this.boxes.forEach(box => {
            box.isNew = false;
            box.isModified = false;
            box.originalBbox = {...box.bbox};
        });
        this.originalBoxes = JSON.parse(JSON.stringify(this.boxes));
    }

    /**
     * Get all boxes.
     * @returns {Array} Array of all boxes
     */
    getAllBoxes() {
        return this.boxes;
    }
}
