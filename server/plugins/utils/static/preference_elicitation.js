
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
        rows2: [],
        itemsPerRow: 80,
        jumboHeader: "Preference Elicitation",
        disableNextStep: false,
        searchMovieName: null,
        itemsBackup: null,
        rowsBackup: null
    }
    },
    async mounted() {
        console.log("Mounted was called {{impl}}");

        const btns = document.querySelectorAll(".btn");
        
        // This was used for reporting as previously reporting endpoints were defined inside plugin
        console.log(`Consuming plugin is: ${consumingPlugin}`);

        // Register the handlers for event reporting
        startViewportChangeReportingWithLimit(`/utils/changed-viewport`, csrfToken, 5.0);
        registerClickedButtonReporting(`/utils/on-input`, csrfToken, btns);
        reportLoadedPage(`/utils/loaded-page`, csrfToken, "preference_elicitation", ()=>{
            return {"impl":"{{impl}}"};
        });
        
        // Get the number of items user is supposed to select
        console.log(this.items);
        console.log("Querying cluster data");
        let data = await fetch("/utils/cluster-data-" + this.impl + "?i=0").then((resp) => resp.json()).then((resp) => resp);
        console.log("DONE")
        console.log(data);
        
        res = this.prepareTable(data);
        this.rows = res["rows"];
        this.items = res["items"];
        
    },
    methods: {
        async handlePrefixSearch(movieName) {
            let foundMovies = await fetch("/utils/movie-search?attrib=movie&pattern="+movieName).then(resp => resp.json());
            console.log(foundMovies);
            return foundMovies;
        },
        prepareTable(data) {
            let row = [];
            let rows = [];
            let items = [];
            for (var k in data) {
                items.push({
                    "movieName": data[k]["movie"],
                    "movie": {
                        "idx": data[k]["movie_idx"],
                        "url": data[k]["url"]
                    }
                });
                row.push({
                    "movieName": data[k]["movie"],
                    "movie": {
                        "idx": data[k]["movie_idx"],
                        "url": data[k]["url"]
                    }
                });
                if (row.length >= this.itemsPerRow) {
                    rows.push(row);
                    row = [];
                }
            }
            if (row.length > 0) {
                rows.push(row);
            }

            return {"rows": rows, "items": items };
        },
        async onClickSearch(event) {
            let data = await this.handlePrefixSearch(this.searchMovieName);
            let res = this.prepareTable(data);
            
            // Do not overwrite backups when doing repeated search
            if (this.itemsBackup === null) {
                this.itemsBackup = this.items;
                this.rowsBackup = this.rows;
            }

            this.rows = res["rows"];
            this.items = res["items"];
        },
        onKeyDownSearchMovieName(e) {
            if (e.key === "Enter") {
                this.onClickSearch(null);
            }
        },
        onClickCancelSearch() {
            this.items = this.itemsBackup;
            this.rows = this.rowsBackup;
            this.itemsBackup = null;
            this.rowsBackup = null;
        },
        async onClickLoadMore() {
            let data = await fetch("/utils/cluster-data-" + this.impl).then((resp) => resp.json()).then((resp) => resp);
            res = this.prepareTable(data);
            this.rows = res["rows"];
            this.items = res["items"];
        },
        onUpdateSearchMovieName(newValue) {
            console.log(this.searchMovieName);
        },
        onSelectMovie(event, item) {
            console.log("Hello from ID=" + event.srcElement.id);
            console.log(event.srcElement)
            console.log(event);
            console.log(item.movieName);

            let index = this.selected.indexOf(item);
            if (index > -1) {
                // Already there, remove it
                this.selected.splice(index, 1);
                event.srcElement.classList.remove("selected");
                reportDeselectedItem(`/utils/deselected-item`, csrfToken, item, this.selected);
            } else {
                // Not there, insert
                this.selected.push(item);
                event.srcElement.classList.add("selected");
                reportSelectedItem(`/utils/selected-item`, csrfToken, item, this.selected);
            }
        },
        onRowClicked(item) {
            console.log("@@@ onRowClicked");
            console.log(item);

                
            let index = this.selected.indexOf(item);
            if (index > -1) {
                this.selected.splice(index, 1);
                //this.$refs.selectableTable.unselectRow(this.items.indexOf(item));
            } else {
                this.selected.push(item);
                //this.$refs.selectableTable.selectRow(this.items.indexOf(item));
            }
        },
        onElicitationFinish(form) {
            let implTag = document.createElement("input");
            let selectedMoviesTag = document.createElement("input");
            implTag.setAttribute("type","hidden");
            implTag.setAttribute("name","impl");
            implTag.setAttribute("value",this.impl);
            selectedMoviesTag.setAttribute("type","hidden");
            selectedMoviesTag.setAttribute("name","selectedMovies");
            selectedMoviesTag.setAttribute("value", this.selected.map((x) => x.movie.idx).join(","));

            form.appendChild(implTag);
            form.appendChild(selectedMoviesTag);
            console.log(form);
            form.submit();
        }
    }
})