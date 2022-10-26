
function preference_elicitation() {

}


window.app = new Vue({
    el: '#app',
    delimiters: ['[[',']]'], // Used to replace double { escaping with double [ escaping (to prevent jinja vs vue inference)
    data: function() {
        
    return {
        userEmail: "{{email}}",
        items: []
    }
    },
    async mounted() {
        console.log("Mounted was called");
        // Get the number of items user is supposed to select
        console.log(this.items);
        let clusterData = await fetch("/utils/cluster-data").then((resp) => resp.json()).then((resp) => resp);
        
        for (var clusterIdx = 0; clusterIdx < clusterData.length; ++clusterIdx) {
            for (var k in clusterData[clusterIdx]["tags"]) {
                let tag = clusterData[clusterIdx]["tags"][k]["tag"];
                let movies = clusterData[clusterIdx]["tags"][k]["movies"];
                this.items.push({"cluster": clusterIdx + 1,"tag": tag, "movies": movies.join(";")});
            }
            // Cluster delimiter
            this.items.push({"cluster": "","tag": "", "movies": "", _rowVariant: 'danger'});
        }
      }
})