console.log(movies);

window.app = new Vue({
    el: '#app',
    delimiters: ['[[',']]'], // Used to replace double { escaping with double [ escaping (to prevent jinja vs vue inference)
    data: function() {
        console.log("Called");
        const colsPerRow = itemsPerRow;

        var numAlgorithms = Object.keys(movies).length;
        var variantNames = new Array(numAlgorithms);
        let moviesColumnified = new Array(numAlgorithms);

        for (variantIdx in movies) {
            let variantResults = movies[variantIdx]["movies"];
            let order = parseInt(movies[variantIdx]["order"]);
            variantNames[order] = variantIdx.toUpperCase();
            
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
            moviesColumnified[order] = variantResultsColumnified;
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
            dontLikeAnythingValue: false,
            algorithm1Q1Value: 0,
            algorithm2Q1Value: 0,
            variantNames: variantNames,
            imageHeight: 300,
            maxColumnsMaxWidth: 300
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
                reportDeselectedItem(`/utils/deselected-item`, csrfToken, item, this.selected);
            } else {
                // Not there, insert
                var copies = document.getElementsByName(event.srcElement.name);
                for (let j = 0; j < copies.length; ++j) {
                    copies[j].classList.add("selected");
                }
                this.selected.push(item);
                reportSelectedItem(`/utils/selected-item`, csrfToken, item, this.selected);
            }
            this.selectedMovieIndices = this.selected.map((x) => x.movie_idx).join(",");
        },
        onAlgorithmRatingChanged(newRating, algorithmIndex) {
            var oldRating = 0;
            if (algorithmIndex == 0) {
                this.algorithm1Q1Validated = true;
                this.algorithm1Q1Variant = "success";
                oldRating = this.algorithm1Q1Value;
                this.algorithm1Q1Value = newRating;
            } else if (algorithmIndex == 1) {
                this.algorithm2Q1Validated = true;
                this.algorithm2Q1Variant = "success";
                oldRating = this.algorithm2Q1Value;
                this.algorithm2Q1Value = newRating;
            }
            reportOnInput("/utils/on-input", csrfToken, "rating", {
                "variant": algorithmIndex,
                "variant_name": this.variantNames[algorithmIndex],
                "old_rating": oldRating,
                "new_rating": newRating
            });
        },
        algorithmQ1Variant(algorithmIndex) {
            if (algorithmIndex == 0 && this.algorithm1Q1Validated) {
                return "success";
            }
            if (algorithmIndex == 1 && this.algorithm2Q1Validated) {
                return "success";
            }
            return "danger";
        },
        updateImageHeight() {
            if (window.innerHeight <= 750) {
                this.imageHeight = 150;
            } else if (window.innerHeight <= 950) {
                this.imageHeight = 200;
            } else {
                this.imageHeight = 300;
            }
        },
        updateMaxColumnsMaxWidth() {
            console.log("Updating");
            if (window.innerWidth <= 1300) {
                this.maxColumnsMaxWidth = 140;
            } else {
                this.maxColumnsMaxWidth = 300;
            }
            console.log(this.maxColumnsMaxWidth);
        },
        updateResolutions() {
            this.updateImageHeight();
            this.updateMaxColumnsMaxWidth();
        }
    },
    async mounted() {
        console.log("Mounted was called");
        const btns = document.querySelectorAll(".btn");
        const chckbxs = document.querySelectorAll("input[type=checkbox]");
        const radios = document.querySelectorAll("input[type=radio]");
        // Register the handlers for event reporting
        startViewportChangeReportingWithLimit(`/utils/changed-viewport`, csrfToken, 1.0, true, compare_ctx_lambda);
        startScrollReportingWithLimit(`/utils/changed-viewport`, csrfToken, 1.0, document.getElementsByName("scrollableDiv"), compare_ctx_lambda);
        registerClickedButtonReporting(`/utils/on-input`, csrfToken, btns);
        registerClickedCheckboxReporting("/utils/on-input", csrfToken, chckbxs);
        registerClickedRadioReporting("/utils/on-input", csrfToken, radios);
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

        this.updateResolutions();
        window.addEventListener("resize", this.updateResolutions);
    }
})