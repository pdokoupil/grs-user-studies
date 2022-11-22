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
            variantsResults: moviesColumnified
        }
    },
    computed: {
    },
    methods: {
    },
    async mounted() {
        console.log("Mounted was called");
    }
})