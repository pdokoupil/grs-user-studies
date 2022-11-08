
function preference_elicitation() {

}


window.app = new Vue({
    el: '#app',
    delimiters: ['[[',']]'], // Used to replace double { escaping with double [ escaping (to prevent jinja vs vue inference)
    data: function() {
    let impl = document.getElementById("impl").className;
    return {
        userEmail: "{{email}}",
        items: [],
        selected: [],
        impl: impl,
        selectMode: "multi",
        lastSelectedCluster: null,
        handle: false,
        rows: [],
        itemsPerRow: 8
    }
    },
    async mounted() {
        console.log("Mounted was called {{impl}}");
        // Get the number of items user is supposed to select
        console.log(this.items);
        console.log("Querying cluster data");
        let clusterData = await fetch("/utils/cluster-data-" + this.impl).then((resp) => resp.json()).then((resp) => resp);
        console.log("DONE")
        console.log(clusterData);
        
        if (this.impl !== "1") {
            let row = [];
            for (var k in clusterData) {
                this.items.push({
                    "movieName": clusterData[k]["movie"],
                    "movie": {
                        "idx": clusterData[k]["movie_idx"],
                        "url": clusterData[k]["url"]
                    }
                });
                row.push({
                    "movieName": clusterData[k]["movie"],
                    "movie": {
                        "idx": clusterData[k]["movie_idx"],
                        "url": clusterData[k]["url"]
                    }
                });
                if (row.length >= this.itemsPerRow) {
                    this.rows.push(row);
                    row = [];
                }
            }
            if (row.length > 0) {
                this.rows.push(row);
            }
        } else {
            for (var clusterIdx = 0; clusterIdx < clusterData.length; ++clusterIdx) {
                for (var k in clusterData[clusterIdx]["tags"]) {
                    let tag = clusterData[clusterIdx]["tags"][k]["tag"];
                    let movies = clusterData[clusterIdx]["tags"][k]["movies"];
                    let movie_names = movies.map((element) => element.movie);
                    console.log(movie_names);
                    let movie_urls = movies.map((element) => element.url);
                    console.log(movie_urls);
                    for (var i in movies) {
                        let movie = movies[i];
                        this.items.push({"cluster": clusterIdx + 1, "tag": tag, "movie": {
                            "idx": movie.movie_idx,
                            "url": movie.url
                        }});
                    }
                    //this.items.push({"cluster": clusterIdx + 1,"tag": tag, "movies": movie_names.join(";")});
                }
                // Cluster delimiter
                //this.items.push({"cluster": "","tag": "", "movies": "", _rowVariant: 'danger'});
                this.items.push({"cluster": "","tag": "", "movie": "", _rowVariant: 'danger'});
            }
        }
        

        
    },
    methods: {
        onSelectMovie(event, item) {
            console.log("Hello from ID=" + event.srcElement.id);
            console.log(event.srcElement.cl)
            console.log(event);
            console.log(item.movieName);

            // for (var k in this.items) {
            //     let item = this.items[k];
            //     if (item.movie.idx === event.srcElement.id) {
            //         console.log("Found match");
            //         console.log(this.items[k]);
            //         this.items[k]["selected"] = true;
            //         console.log(this.items[k]);
            //     }
            // }
            let index = this.selected.indexOf(item);
            if (index > -1) {
                // Already there, remove it
                this.selected.splice(index, 1);
                event.srcElement.parentElement.classList.remove("bg-info");
            } else {
                // Not there, insert
                this.selected.push(item);
                event.srcElement.parentElement.classList.add("bg-info");
            }
        },
        onRowClicked(item) {
            console.log("@@@ onRowClicked");
            console.log(item);
            if (this.impl === "1") {
                if (this.lastSelectedCluster !== null) {
                    // Deselect all the items from that cluster
                    // for (var i in this.items) {
                    //     if (this.items[i].cluster === this.lastSelectedCluster) {
                    //         console.log("Unselecting: " + i);
                    //         let idx = parseInt(i, 10);
                    //         this.$refs.selectableTable.unselectRow(idx);
                    //     }
                    // }
                    this.$refs.selectableTable.clearSelected();
                    // 
                }
                
                
                this.lastSelectedCluster = item.cluster;
                console.log(this.lastSelectedCluster);
                let selectedItems = [];
                // Select all items from this cluster
                for (var i in this.items) {
                    if (this.items[i].cluster === this.lastSelectedCluster && this.items[i] != item) {
                        console.log("Selecting: " + i);
                        let idx = parseInt(i, 10);
                        this.$refs.selectableTable.selectRow(idx);
                        selectedItems.push(this.items[i]);
                    }

                }
                this.selected = selectedItems;

                return;
                // console.log("Selected items:");
                // console.log(items);
                // console.log(this.$refs.selectableTable);
                //console.log("Select all items from the same cluster");
                //this.$refs.selectableTable.selectRow(2);
                //this.$refs.selectableTable.selectRow(3);
            } else {
                //this.selected = item;
                
                let index = this.selected.indexOf(item);
                if (index > -1) {
                    this.selected.splice(index, 1);
                    //this.$refs.selectableTable.unselectRow(this.items.indexOf(item));
                } else {
                    this.selected.push(item);
                    //this.$refs.selectableTable.selectRow(this.items.indexOf(item));
                }
                
            }
        },
        async onElicitationFinish() {
            if (this.impl === "1") {
                let cid = parseInt(this.selected[0].cluster, 10) - 1;
                let params = "selectedCluster=" + cid.toString();
                console.log("Selected cluster is: " + params);
                let got = await fetch("/utils/send-feedback?impl=" + this.impl + "&" + params).then((resp) => resp.json()).then((resp) => resp);
                console.log("Got values: " + got);
                console.log(got);
            } else {
                let params = "selectedMovies=" + this.selected.map((x) => x.movie.idx).join(",");
                console.log("Selected movies are: " + params);
                let got = await fetch("/utils/send-feedback?impl=" + this.impl + "&" + params).then((resp) => resp.json()).then((resp) => resp);
                console.log("Got values: " + got);
                console.log(got);

                let redirected = false;

                // Continue with step 1
                let res = await fetch(nextStepUrl,
                {
                    method: "POST",
                    headers: {
                        'Content-Type': 'application/json',
                        'X-CSRFToken': csrfToken
                    },
                    body: JSON.stringify(got),
                    redirect: "follow"
                }
                ).then(response => {
                    if (response.redirected) {
                        console.log(response);
                        window.location.href = response.url;
                        redirected = true;
                    } else {
                        return response.text()
                    }
                });

                // Follow link and ensure that URL bar is reloaded as well
                console.log(res);
                if (redirected === false) {
                    document.body.innerHTML = res;
                    window.history.pushState("", "", nextStepUrl);
                }
            }
            
        }
    }
})