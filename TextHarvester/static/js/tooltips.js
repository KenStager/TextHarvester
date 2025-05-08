/**
 * Enhanced tooltips with Lottie animations
 * This file handles the initialization and management of animated tooltips
 * across the web scraper application.
 */

// Store animation instances for reuse
const animations = {};

/**
 * Initialize Bootstrap tooltips with custom options
 */
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl, {
            html: true,
            boundary: 'viewport',
            delay: { show: 100, hide: 100 }
        });
    });
}

/**
 * Initialize Lottie animations in tooltips
 */
function initializeAnimations() {
    // Look for any containers with animation data
    const animationContainers = document.querySelectorAll('.tooltip-animation');
    
    animationContainers.forEach(container => {
        const animationName = container.dataset.animation;
        if (!animationName) return;
        
        // Load the animation if not already loaded
        if (!animations[animationName]) {
            loadAnimation(container, animationName);
        }
    });
}

/**
 * Load a Lottie animation
 * 
 * @param {HTMLElement} container - The container element
 * @param {string} animationName - The name of the animation
 */
function loadAnimation(container, animationName) {
    const animationPath = `/static/animations/${animationName}.json`;
    
    // Create and configure the animation
    animations[animationName] = lottie.loadAnimation({
        container: container,
        renderer: 'svg',
        loop: true,
        autoplay: true,
        path: animationPath,
        rendererSettings: {
            progressiveLoad: true,
            preserveAspectRatio: 'xMidYMid slice'
        }
    });
}

/**
 * Create a tooltip content with animation
 * 
 * @param {string} title - The tooltip title
 * @param {string} text - The tooltip description text
 * @param {string} animationName - The animation JSON file name
 * @returns {string} HTML content for the tooltip
 */
function createAnimatedTooltipContent(title, text, animationName) {
    return `
        <div class="animated-tooltip">
            <div class="tooltip-animation-container">
                <div class="tooltip-animation" data-animation="${animationName}"></div>
            </div>
            <div class="tooltip-content">
                <h5 class="tooltip-title">${title}</h5>
                <p class="tooltip-text">${text}</p>
            </div>
        </div>
    `;
}

/**
 * Add a new animated tooltip to an element
 * 
 * @param {string} selector - CSS selector for the target element
 * @param {string} title - Tooltip title
 * @param {string} text - Tooltip description
 * @param {string} animationName - Animation JSON file name
 * @param {string} placement - Tooltip placement (top, bottom, left, right)
 */
function addAnimatedTooltip(selector, title, text, animationName, placement = 'auto') {
    const element = document.querySelector(selector);
    if (!element) return;
    
    // Add tooltip attributes
    element.setAttribute('data-bs-toggle', 'tooltip');
    element.setAttribute('data-bs-html', 'true');
    element.setAttribute('data-bs-placement', placement);
    element.setAttribute('data-bs-animation', 'true');
    element.setAttribute('data-bs-container', 'body');
    element.setAttribute('title', createAnimatedTooltipContent(title, text, animationName));
    
    // Initialize the tooltip
    const tooltip = new bootstrap.Tooltip(element, {
        html: true,
        boundary: 'viewport',
        delay: { show: 100, hide: 100 }
    });
    
    // Set up animation initialization on tooltip show
    element.addEventListener('shown.bs.tooltip', function () {
        // Find the animation container in the tooltip
        const tooltipId = element.getAttribute('aria-describedby');
        if (!tooltipId) return;
        
        const tooltipElement = document.getElementById(tooltipId);
        if (!tooltipElement) return;
        
        const animationContainer = tooltipElement.querySelector('.tooltip-animation');
        if (!animationContainer) return;
        
        // Load the animation
        loadAnimation(animationContainer, animationName);
    });
}

// Initialize tooltips when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeTooltips();
});