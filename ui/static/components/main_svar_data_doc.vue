<template>
    <div>
        <markdown-element>

# Dynamic Brain Connectivity
# Dataset fMRI-UCI/UCL

The dataset consists of fMRI recordings from fourteen subjects watching a movie. Region-specific time series were extracted from the following:

Code     | Region of interest
-------: | -----------------
`CG`     | Central gyrus 
`PreCG`  | Pre-central gyrus 
`PoCG`   | Post-central gyrus 
`STG`    | Superior temporal gyrus 
`TTG`    | Transverse temporal gyrus 
`TTS`    | Transverse temporal sulcus 
`PP`     | Planum polare and 
`PT`     | Planum temporale 

> ![](../docs/models/svar/brain-description.png)

Previous studies have shown that these regions present a relationship patterns with motor or emotional stimuli [3].

# References

  [1] Ting, C. M., Ombao, H., Balqis Samdin, S., and Salleh, S. H. (2017). Estimating Time-Varying Effective Connectivity in High-Dimensional fMRI Data Using Regime-Switching Factor Models. arXiv preprint arXiv:1701.06754.

  [2] Wang, Y., Ting, C. M., and Ombao, H. (2016). Modeling effective connectivity in high-dimensional cortical source signals. IEEE Journal of Selected Topics in Signal Processing, 10(7), 1315-1325.

  [3] Eavani, H., Satterthwaite, T. D., Filipovych, R., Gur, R. E., Gur, R. C., and Davatzikos, C. (2015). Identifying sparse connectivity patterns in the brain using resting-state fMRI. Neuroimage, 105, 286-299.

        </markdown-element>
    </div>
</template>

<script>

var ApplicationSVARDataDocumentation = {
    template: iview.getTagNamesFromElement(document.currentScript.previousElementSibling),
    data(){
        return {
        }
    },
    methods: {
        /*compileDocumentMarkdown: _.debounce( (e) => {
            this.helpDocument = marked(e.target.value, { sanitize: true })
        }, 300),*/
    },
    computed: {
    }
}
Vue.component('app-svar-data-documentation', ApplicationSVARDataDocumentation)

</script>
