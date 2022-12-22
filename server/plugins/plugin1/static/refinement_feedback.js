window.app = new Vue({
    el: '#app',
    delimiters: ['[[',']]'], // Used to replace double { escaping with double [ escaping (to prevent jinja vs vue inference)
    data: function() {
        
        return {
            relevance: defaultRelevance,
            diversity: defaultDiversity,
            novelty: defaultNovelty,
            relevanceValue: null,
            diversityValue: null,
            noveltyValue: null,
            relevanceDelta: 0,
            diversityDelta: 0,
            noveltyDelta: 0,
            deltaDescription: ["No attention", "Less attention", "Keep same", "More attention", "Full attention"]
        }
    },
    computed: {
    },
    methods: {
        onRelevanceChange(newRel) {
            console.log("Called relevance=%d, old_relevance=%d", relevance, this.relevance);

            var newRelevance = parseFloat(newRel);
            var relevance = parseFloat(this.relevance);
            var diversity = parseFloat(this.diversity);
            var novelty = parseFloat(this.novelty);
            var othersAccum = diversity + novelty;

            if (othersAccum == 0) {
                return newRel;
            }

            var diversityShare = diversity / othersAccum;
            var noveltyShare = novelty / othersAccum;

            if (newRelevance > this.relevance) {
                // Handle increase
                let diff = newRelevance - relevance;
                
                diversity -= diversityShare * diff;
                novelty -= noveltyShare * diff;

                let totalAccum = newRelevance + diversity + novelty;
                diversity = (diversity / totalAccum) * 100;
                novelty = (novelty / totalAccum) * 100;

                this.diversity = diversity.toFixed(1);
                this.novelty = novelty.toFixed(1);

                return ((newRelevance / totalAccum) * 100).toFixed(1); //this.relevance + diff;
            } else if (relevance < this.relevance) {
                // Handle decrease
                let diff = relevance - newRelevance;
                
                diversity += diversityShare * diff;
                novelty += noveltyShare * diff;

                let totalAccum = newRelevance + diversity + novelty;
                diversity = (diversity / totalAccum) * 100;
                novelty = (novelty / totalAccum) * 100;

                this.diversity = diversity.toFixed(1);
                this.novelty = novelty.toFixed(1);

                return ((newRelevance / totalAccum) * 100).toFixed(1); //this.relevance + diff;
            }

            return newRel;
        },
        onDiversityChange(newDiv) {
            console.log("Called diversity=%d", newDiv);

            var newDiversity = parseFloat(newDiv);
            var diversity = parseFloat(this.diversity);
            var relevance = parseFloat(this.relevance);
            var novelty = parseFloat(this.novelty);
            var othersAccum = relevance + novelty;

            if (othersAccum == 0) {
                return newDiv;
            }

            var relevanceShare = relevance / othersAccum;
            var noveltyShare = novelty / othersAccum;

            if (newDiversity > this.diversity) {
                // Handle increase
                let diff = newDiversity - diversity;
                
                relevance -= relevanceShare * diff;
                novelty -= noveltyShare * diff;

                let totalAccum = newDiversity + relevance + novelty;
                relevance = (relevance / totalAccum) * 100;
                novelty = (novelty / totalAccum) * 100;

                this.relevance = relevance.toFixed(1);
                this.novelty = novelty.toFixed(1);

                return ((newDiversity / totalAccum) * 100).toFixed(1);
            } else if (newDiversity < this.diversity) {
                // Handle decrease
                let diff = diversity - newDiversity;
                
                relevance += relevanceShare * diff;
                novelty += noveltyShare * diff;

                let totalAccum = newDiversity + relevance + novelty;
                relevance = (relevance / totalAccum) * 100;
                novelty = (novelty / totalAccum) * 100;

                this.relevance = relevance.toFixed(1);
                this.novelty = novelty.toFixed(1);

                return ((newDiversity / totalAccum) * 100).toFixed(1); 
            }

            return newDiv;
        },
        onNoveltyChange(newNov) {
            console.log("Called novelty=%d", newNov);

            var newNovelty = parseFloat(newNov);
            var novelty = parseFloat(this.novelty);
            var relevance = parseFloat(this.relevance);
            var diversity = parseFloat(this.diversity);
            var othersAccum = relevance + diversity;

            if (othersAccum == 0) {
                return newNov;
            }

            var relevanceShare = relevance / othersAccum;
            var diversityShare = diversity / othersAccum;

            if (newNovelty > this.novelty) {
                // Handle increase
                let diff = newNovelty - novelty;
                
                relevance -= relevanceShare * diff;
                diversity -= diversityShare * diff;

                let totalAccum = newNovelty + relevance + diversity;
                relevance = (relevance / totalAccum) * 100;
                diversity = (diversity / totalAccum) * 100;

                this.relevance = relevance.toFixed(1);
                this.diversity = diversity.toFixed(1);

                return ((newNovelty / totalAccum) * 100).toFixed(1);
            } else if (newNovelty < this.novelty) {
                // Handle decrease
                let diff = novelty - newNovelty;
                
                relevance += relevanceShare * diff;
                diversity += diversityShare * diff;

                let totalAccum = newNovelty + relevance + diversity;
                relevance = (relevance / totalAccum) * 100;
                diversity = (diversity / totalAccum) * 100;

                this.relevance = relevance.toFixed(1);
                this.diversity = diversity.toFixed(1);

                return ((newNovelty / totalAccum) * 100).toFixed(1); 
            }

            return newNov;
        },
        onRelevanceDeltaChange(newVal) {
            reportOnInput("/utils/on-input", csrfToken, "range", {
                "old_value": this.relevanceDelta,
                "new_value": newVal,
                "labels": this.deltaDescription,
                "metric": "relevance"
            });
            return newVal;
        },
        onDiversityDeltaChange(newVal) {
            reportOnInput("/utils/on-input", csrfToken, "range", {
                "old_value": this.relevanceDelta,
                "new_value": newVal,
                "labels": this.deltaDescription,
                "metric": "diversity"
            });
            return newVal;
        },
        onNoveltyDeltaChange(newVal) {
            reportOnInput("/utils/on-input", csrfToken, "range", {
                "old_value": this.relevanceDelta,
                "new_value": newVal,
                "labels": this.deltaDescription,
                "metric": "novelty"
            });
            return newVal;
        }
    },
    async mounted() {
        console.log("Mounted was called");
        const btns = document.querySelectorAll(".btn");
        
        // Register the handlers for event reporting
        startViewportChangeReportingWithLimit(`/utils/changed-viewport`, csrfToken, 5.0);
        registerClickedButtonReporting(`/utils/on-input`, csrfToken, btns);
        reportLoadedPage(`/utils/loaded-page`, csrfToken, "refinement_feedback", ()=>
            {
                data = {
                    "result_layout": resultLayout, // TODO REPLACE with layout_version 
                    "initial_weights": {
                        "relevance": defaultRelevance,
                        "diversity": defaultDiversity,
                        "novelty": defaultNovelty
                    }, // TODO add initial weights
                    "processed_inputs": {
                        "relevance": newRelevance,
                        "diversity": newDiversity,
                        "novelty": newNovelty
                    }
                };

                if (resultLayout == "1") {
                    data["raw_inputs"] = { // TODO add slider values and new weights
                        "relevance": relevance,
                        "diversity": diversity,
                        "novelty": novelty
                    }
                }
                else if (resultLayout == "2") {
                    data["raw_inputs"] = {
                        "relevance": relevanceValue,
                        "diversity": diversityValue,
                        "novelty": noveltyValue
                    }
                }
                else if (resultLayout == "3") {
                    data["raw_inputs"] = { // TODO add slider values and new weights
                        "relevance": relevanceDelta,
                        "diversity": diversityDelta,
                        "novelty": noveltyDelta
                    }
                }

                return data;
            }
        );
    }
})