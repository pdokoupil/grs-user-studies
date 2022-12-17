console.log(movies);

window.app = new Vue({
    el: '#app',
    delimiters: ['[[',']]'], // Used to replace double { escaping with double [ escaping (to prevent jinja vs vue inference)
    data: function() {
        console.log("Called");
        let moviesColumnified = [];

        const colsPerRow = itemsPerRow;

        var numAlgorithms = 0;

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
            ++numAlgorithms;
        }
        console.log(moviesColumnified);
        return {
            variantsResults: moviesColumnified,
            selected: [],
            selectedMovieIndices: "",
            algorithmComparisonValue: null,
            algorithmComparisonValidated: false,
            numAlgorithms: numAlgorithms,
            algorithm1Q1Validated: false,
            algorithm2Q1Validated: false,
            dontLikeAnythingValue: false
        }
    },
    computed: {
        algorithmComparisonState() {
            console.log(this.algorithmComparisonValue);
            this.algorithmComparisonValidated = this.algorithmComparisonValue != null;
            return this.algorithmComparisonValue != null;
        },
        dontLikeAnythingState() {
            return this.dontLikeAnythingValue;
        },
        allValidated() {
            let dontLikeAnythingValidated = this.selected.length > 0 || this.dontLikeAnythingValue;
            return this.algorithmComparisonValidated && this.algorithm1Q1Validated && this.algorithm2Q1Validated && dontLikeAnythingValidated;
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
            item['variant'] = variant;
            let index = this.movieIndexOf(this.selected, item);
            if (index > -1) {
                // Already there, remove it
                var copies = document.getElementsByName(event.srcElement.name);
                for (let j = 0; j < copies.length; ++j) {
                    copies[j].classList.remove("selected");
                }
                this.selected.splice(index, 1);
            } else {
                // Not there, insert
                var copies = document.getElementsByName(event.srcElement.name);
                for (let j = 0; j < copies.length; ++j) {
                    copies[j].classList.add("selected");
                }
                this.selected.push(item);
            }
            this.selectedMovieIndices = this.selected.map((x) => x.movie_idx).join(",");
        },
        onAlgorithmRatingChanged(newRating, algorithmIndex) {
            if (algorithmIndex == 0) {
                this.algorithm1Q1Validated = true;
                this.algorithm1Q1Variant = "success";
            } else if (algorithmIndex == 1) {
                this.algorithm2Q1Validated = true;
                this.algorithm2Q1Variant = "success";
            }
        },
        algorithmQ1Variant(algorithmIndex) {
            if (algorithmIndex == 0 && this.algorithm1Q1Validated) {
                return "success";
            }
            if (algorithmIndex == 1 && this.algorithm2Q1Validated) {
                return "success";
            }
            return "danger";
        }
    },
    async mounted() {
        console.log("Mounted was called");
        // Register the handlers for event reporting
        startViewportChangeReportingWithLimit(`/utils/changed-viewport`, csrfToken, 5.0);
        registerClickedButtonReporting(`/utils/clicked-button`, csrfToken, btns);
        reportLoadedPage(`/utils/loaded-page`, csrfToken, "compare_algorithms", ()=>
            {
                return {
                    "result_layout": resultLayout,
                    "movies": movies,
                    "iteration": iteration,
                    "min_iteration_to_cancel": minIterationToCancel
                };
            }
        );
    }
})