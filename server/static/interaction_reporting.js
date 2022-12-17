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

// Returns context, something that is relevant for most of the reported status and that makes
// sense to accompany reported information
function getContext(extra="") {
    return {
        "url": window.location.href,
        "time": String(new Date()),
        "viewport": getViewportBoundingBox(),
        "extra": extra
    };
}

function reportViewportChange(endpoint, data, csrfToken, extraCtxLambda=()=>"") {
    data = {
        "viewport": getViewportBoundingBox(),
        "context": getContext(extraCtxLambda())
    }
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
function startViewportChangeReporting(endpoint, csrfToken, initialReport=true, extraCtxLambda=()=>"") {
    
    if (initialReport === true) {
        reportViewportChange(endpoint, csrfToken, extraCtxLambda); 
    }

    window.addEventListener("scroll", function(e) {
        reportViewportChange(endpoint, csrfToken, extraCtxLambda);
    });
}


// Starts listening for viewport changes and posts them to the given endpoint
// initialReport parameter allow us to report about current values
// (this is especially useful if we want to report initial viewport dimensions, before any user action)
function startViewportChangeReportingWithLimit(endpoint, csrfToken, timeLimitSeconds, initialReport=true, extraCtxLambda=()=>"") {
    if (initialReport === true) {
        reportViewportChange(endpoint, getViewportBoundingBox(), csrfToken, extraCtxLambda); 
    }

    var lastReported = new Date();

    window.addEventListener("scroll", function(e) {
        let now = new Date();
        console.log(endpoint);
        if ((now - lastReported) / 1000 > timeLimitSeconds) {
            reportViewportChange(endpoint, getViewportBoundingBox(), csrfToken, extraCtxLambda);
            lastReported = now;
        }
    });
}

function registerClickedButtonReporting(endpoint, csrfToken, btns, extraCtxLambda=()=>"") { 
    btns.forEach(btn => {

        btn.addEventListener('click', event => {
            data = {
                "id": event.target.id,
                "text": event.target.textContent,
                "name": event.target.name,
                "context": getContext(extraCtxLambda=()=>"")
            };
            console.log(data);
            fetch(endpoint,
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
        });

    });
}

function reportLoadedPage(endpoint, csrfToken, pageName, extraCtxLambda=()=>"") {
    data = {
        "page": pageName,
        "context": getContext(extraCtxLambda())
    };
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

function reportSelectedItem(endpoint, csrfToken, selectedItem, selectedItems, extraCtxLambda=()=>"") {
    data = {
        "selected_item": selectedItem,
        "selected_items": selectedItems,
        "context": getContext(extraCtxLambda())
    };
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

function reportDeselectedItem(endpoint, csrfToken, deselectedItem, selectedItems, extraCtxLambda=()=>"") {
    data = {
        "deselected_item": deselectedItem,
        "selected_items": selectedItems,
        "context": getContext(extraCtxLambda())
    };
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