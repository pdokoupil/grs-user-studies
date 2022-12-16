// This script registers some of the handlers that are commonly used for interaction reporting and makes sure they are delivered to the server

// Returns bounding box of given element w.r.t. viewport
function getElementBoundingBox(element) {
    let x = element.getBoundingClientRect();
    return {
        "left": x.left,
        "top": x.top,
        "width": x.width,
        "height": x.height
    };
}

function getViewportBoundingBox() {
    return getElementBoundingBox(document.documentElement);
}

function reportViewportChange(endpoint, data, csrfToken) {
    return fetch(endpoint,
        {
            method: "POST",
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrfToken
            },
            body: JSON.stringify(data),
            redirect: "follow"
        }
    )
}

// Starts listening for viewport changes and posts them to the given endpoint
// initialReport parameter allow us to report about current values
// (this is especially useful if we want to report initial viewport dimensions, before any user action)
function startViewportChangeReporting(endpoint, csrfToken, initialReport=false) {
    
    if (initialReport === true) {
        reportViewportChange(endpoint, getViewportBoundingBox(), csrfToken); 
    }

    window.addEventListener("scroll", function(e) {
        reportViewportChange(endpoint, getViewportBoundingBox(), csrfToken);
    });
}


// Starts listening for viewport changes and posts them to the given endpoint
// initialReport parameter allow us to report about current values
// (this is especially useful if we want to report initial viewport dimensions, before any user action)
function startViewportChangeReportingWithLimit(endpoint, csrfToken, timeLimitSeconds, initialReport=false) {
    if (initialReport === true) {
        reportViewportChange(endpoint, getViewportBoundingBox(), csrfToken); 
    }

    var lastReported = new Date();

    window.addEventListener("scroll", function(e) {
        let now = new Date();
        console.log(endpoint);
        if ((now - lastReported) / 1000 > timeLimitSeconds) {
            reportViewportChange(endpoint, getViewportBoundingBox(), csrfToken);
            lastReported = now;
        }
    });
}