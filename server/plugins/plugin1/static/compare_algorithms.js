console.log(movies);

window.app = new Vue({
    el: '#app',
    delimiters: ['[[',']]'], // Used to replace double { escaping with double [ escaping (to prevent jinja vs vue inference)
    data: function() {
        console.log("Called");
        let moviesColumnified = [];

        const colsPerRow = 2;

        for (variantIdx in movies) {
            let variantResults = movies[variantIdx];
            let variantResultsColumnified = [];
            let row = [];
            for (idx in variantResults) {
                let movie = variantResults[idx];
                row.push(movie);
                if (row.length >= colsPerRow) {
                    variantResultsColumnified.push(row);
                    row = [];
                }
            }
            if (row.length > 0) {
                variantResultsColumnified.push(row);
            }
            moviesColumnified.push(variantResultsColumnified);
        }
        console.log(moviesColumnified);
        return {
            variantsResults: moviesColumnified,
            selected: [],
            algorithmComparisonValue: null
        }
    },
    computed: {
        algorithmComparisonState() {
            console.log(this.algorithmComparisonValue);
            return this.algorithmComparisonValue != null;
        }
    },
    methods: {

        // Custom (movie specific) implementation of indexOf operator
        // Considers only movie's properties
        movieIndexOf(arr, item) {
            for (let idx in arr) {
                let arrItem = arr[idx];
                if (arrItem.movie_idx === item.movie_idx
                    && arrItem.movie === item.movie
                    && arrItem.url === item.url) {
                        return idx;
                    }
            }
            return -1;
        },
        onSelectMovie(event, item, variant) {
            let index = this.movieIndexOf(this.selected, item);
            if (index > -1) {
                // Already there, remove it
                var copies = document.getElementsByName(event.srcElement.name);
                for (let j = 0; j < copies.length; ++j) {
                    copies[j].parentElement.classList.remove("bg-info");
                }
                this.selected.splice(index, 1);
            } else {
                // Not there, insert
                var copies = document.getElementsByName(event.srcElement.name);
                for (let j = 0; j < copies.length; ++j) {
                    copies[j].parentElement.classList.add("bg-info");
                }
                this.selected.push(item);
            }
        }
    },
    async mounted() {
        console.log("Mounted was called");
    }
})