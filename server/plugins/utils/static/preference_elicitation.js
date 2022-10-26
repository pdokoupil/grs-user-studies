
function preference_elicitation() {

}


window.app = new Vue({
    el: '#app',
    delimiters: ['[[',']]'], // Used to replace double { escaping with double [ escaping (to prevent jinja vs vue inference)
    data: function() {
        
    return {
        userEmail: "{{email}}"
    }
    }
})