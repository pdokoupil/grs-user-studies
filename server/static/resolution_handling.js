
function validateScreenResolution(minW, minH) {
    let success = true;
    
    if (minW) {
        success = success && window.innerWidth >= minW;
    }

    if (minH) {
        success = success && window.innerHeight >= minH;
    }

    return success;
}