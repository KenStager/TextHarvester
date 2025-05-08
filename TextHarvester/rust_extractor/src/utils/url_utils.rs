use url::Url;

/// Check if a URL is valid
pub fn is_valid_url(url_str: &str) -> bool {
    match Url::parse(url_str) {
        Ok(url) => url.scheme() == "http" || url.scheme() == "https",
        Err(_) => false,
    }
}

/// Extract domain from URL
pub fn get_domain(url_str: &str) -> Option<String> {
    Url::parse(url_str).ok().and_then(|url| url.host_str().map(|s| s.to_string()))
}

/// Normalize URL (remove fragments, default ports, etc.)
pub fn normalize_url(url_str: &str) -> Option<String> {
    match Url::parse(url_str) {
        Ok(mut url) => {
            // Remove fragments
            url.set_fragment(None);
            
            // Remove default ports
            if (url.scheme() == "http" && url.port() == Some(80)) ||
               (url.scheme() == "https" && url.port() == Some(443)) {
                url.set_port(None).unwrap_or(());
            }
            
            Some(url.to_string())
        },
        Err(_) => None,
    }
}

/// Join a base URL with a relative URL
pub fn join_url(base_url: &str, relative_url: &str) -> Option<String> {
    match Url::parse(base_url) {
        Ok(base) => base.join(relative_url).ok().map(|u| u.to_string()),
        Err(_) => None,
    }
}

/// Check if a URL is from the same domain as the base URL
pub fn is_same_domain(base_url: &str, url: &str) -> bool {
    let base_domain = get_domain(base_url);
    let url_domain = get_domain(url);
    
    match (base_domain, url_domain) {
        (Some(base), Some(target)) => base == target,
        _ => false,
    }
}