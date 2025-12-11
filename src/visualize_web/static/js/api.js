/**
 * API layer for communicating with the Flask backend.
 */

class API {
    constructor() {
        this.baseUrl = '';
    }

    /**
     * Load image directory and annotations.
     * @param {string} dirPath - Path to IMAGE_DIR
     * @returns {Promise<{images: string[], annotations: Object}>}
     */
    async loadImageDirectory(dirPath) {
        const response = await fetch(`/api/images?dir=${encodeURIComponent(dirPath)}`);

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to load images');
        }

        const data = await response.json();
        // Backend returns: {images: [...], annotations: {filename: [...]}}
        // Note: Backend flattens comment + content arrays into single array per image
        return data;
    }

    /**
     * Get URL for an image.
     * @param {string} filename - Image filename (not full path)
     * @returns {string} Image URL
     */
    getImageUrl(filename) {
        return `/api/image/${encodeURIComponent(filename)}`;
    }

    /**
     * Save annotations for an image.
     * @param {string} filename - Image filename (not full path)
     * @param {Array} annotations - List of annotations
     * @returns {Promise<void>}
     */
    async saveAnnotations(filename, annotations) {
        const response = await fetch('/api/annotations', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                filename: filename,
                annotations: annotations
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to save annotations');
        }

        return await response.json();
    }
}
