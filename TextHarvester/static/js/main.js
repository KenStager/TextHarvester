// Main JavaScript functionality for the web scraper interface

// Initialize on document ready
document.addEventListener('DOMContentLoaded', function() {
    // Initialize intelligence components in the background
    setTimeout(function() {
        fetch('/init-intelligence')
            .then(response => response.json())
            .then(data => {
                console.log('Intelligence initialization:', data);
            })
            .catch(error => {
                console.error('Error initializing intelligence components:', error);
            });
    }, 2000);  // Wait 2 seconds after page load to avoid slowing down initial rendering

    // Initialize any tooltips
    const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    [...tooltipTriggerList].map(tooltipTriggerEl => new bootstrap.Tooltip(tooltipTriggerEl));
    
    // Set up event listeners
    setupJobStatusRefresh();
    setupFormValidation();
    setupSourceListSelection();
});

/**
 * Set up automatic refresh for job status pages
 */
function setupJobStatusRefresh() {
    const statusContainer = document.getElementById('job-status-container');
    if (!statusContainer) return;
    
    const jobId = statusContainer.dataset.jobId;
    const refreshInterval = 5000; // 5 seconds
    
    // Function to update job status
    const updateJobStatus = () => {
        console.log(`Fetching job status for job ${jobId}`);
        fetch(`/jobs/${jobId}`)
            .then(response => {
                console.log(`Response status: ${response.status}`);
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                // Update status badge
                const statusBadge = document.getElementById('status-badge');
                if (statusBadge) {
                    statusBadge.textContent = data.status;
                    
                    // Update badge class based on status
                    statusBadge.className = 'badge';
                    switch(data.status) {
                        case 'pending':
                            statusBadge.classList.add('bg-secondary');
                            break;
                        case 'running':
                            statusBadge.classList.add('bg-primary');
                            break;
                        case 'completed':
                            statusBadge.classList.add('bg-success');
                            break;
                        case 'failed':
                            statusBadge.classList.add('bg-danger');
                            break;
                    }
                }
                
                // Update statistics
                document.getElementById('urls-processed').textContent = data.urls_processed;
                document.getElementById('urls-successful').textContent = data.urls_successful;
                document.getElementById('urls-failed').textContent = data.urls_failed;
                
                // Update times
                if (data.start_time) {
                    document.getElementById('start-time').textContent = new Date(data.start_time).toLocaleString();
                }
                if (data.end_time) {
                    document.getElementById('end-time').textContent = new Date(data.end_time).toLocaleString();
                }
                
                // If job is still running, schedule another update
                if (data.status === 'running' || data.status === 'pending') {
                    setTimeout(updateJobStatus, refreshInterval);
                } else {
                    // Job is complete or failed, reload page to get final data
                    window.location.reload();
                }
            })
            .catch(error => {
                console.error('Error updating job status:', error);
                // Even on error, try again
                setTimeout(updateJobStatus, refreshInterval);
            });
    };
    
    // Initial status check after a short delay
    setTimeout(updateJobStatus, 1000);
}

/**
 * Set up form validation for configuration forms
 */
function setupFormValidation() {
    const configForm = document.getElementById('config-form');
    if (!configForm) return;
    
    configForm.addEventListener('submit', function(event) {
        // Check if predefined source list is selected
        const predefinedSource = document.getElementById('predefined_source');
        const isPredefinedSourceSelected = predefinedSource && predefinedSource.value !== 'custom';
        
        // Only validate URLs if using custom source
        if (!isPredefinedSourceSelected) {
            const baseUrls = document.getElementById('base_urls').value.trim();
            if (!baseUrls) {
                event.preventDefault();
                
                // Show error alert
                const alertContainer = document.getElementById('alert-container');
                alertContainer.innerHTML = `
                    <div class="alert alert-danger alert-dismissible fade show" role="alert">
                        Please provide at least one base URL to scrape or select a predefined source list.
                        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                    </div>
                `;
                
                // Scroll to top to show alert
                window.scrollTo(0, 0);
            }
        }
        
        // Validate other numeric fields
        const numericFields = ['max_depth', 'rate_limit', 'max_retries'];
        for (const fieldId of numericFields) {
            const field = document.getElementById(fieldId);
            const value = parseInt(field.value);
            
            if (isNaN(value) || value < 1) {
                event.preventDefault();
                
                // Show error alert
                const alertContainer = document.getElementById('alert-container');
                alertContainer.innerHTML = `
                    <div class="alert alert-danger alert-dismissible fade show" role="alert">
                        ${fieldId.replace('_', ' ')} must be a positive number.
                        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                    </div>
                `;
                
                // Scroll to top to show alert
                window.scrollTo(0, 0);
                break;
            }
        }
    });
}

/**
 * Format relative time for job display
 */
function formatRelativeTime(dateString) {
    if (!dateString) return 'N/A';
    
    const date = new Date(dateString);
    const now = new Date();
    const diffInSeconds = Math.floor((now - date) / 1000);
    
    if (diffInSeconds < 60) {
        return `${diffInSeconds} seconds ago`;
    } else if (diffInSeconds < 3600) {
        return `${Math.floor(diffInSeconds / 60)} minutes ago`;
    } else if (diffInSeconds < 86400) {
        return `${Math.floor(diffInSeconds / 3600)} hours ago`;
    } else {
        return `${Math.floor(diffInSeconds / 86400)} days ago`;
    }
}

/**
 * Set up source list selection functionality
 */
function setupSourceListSelection() {
    const sourceSelect = document.getElementById('predefined_source');
    if (!sourceSelect) return;
    
    const customUrlsContainer = document.getElementById('custom_urls_container');
    if (!customUrlsContainer) return;
    
    // Handle change event
    sourceSelect.addEventListener('change', function() {
        const selectedValue = this.value;
        
        if (selectedValue === 'custom') {
            // Show custom URL input
            customUrlsContainer.style.display = 'block';
        } else {
            // Hide custom URL input when using predefined source
            customUrlsContainer.style.display = 'none';
        }
    });
    
    // Initial setup
    if (sourceSelect.value !== 'custom') {
        customUrlsContainer.style.display = 'none';
    }
    
    // Override form validation for predefined source lists
    const configForm = document.getElementById('config-form');
    if (configForm) {
        const originalValidation = configForm.onsubmit;
        
        configForm.onsubmit = function(event) {
            const selectedValue = sourceSelect.value;
            
            // If using predefined source, bypass URL validation
            if (selectedValue !== 'custom') {
                return true;
            }
            
            // Otherwise use the original validation
            if (originalValidation) {
                return originalValidation(event);
            }
        };
    }
}
