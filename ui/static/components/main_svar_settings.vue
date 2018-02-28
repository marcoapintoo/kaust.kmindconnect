<template>
    <div class="app-title-tab">
        <!--<iview-collapse v-model="value1">-->
        <iview-collapse v-model="collapsedTVVAR">
            <iview-panel name="1">
                TV-VAR model settings
                <iview-form-item label="Model order" slot="content">
                    <iview-input-number v-model="model.tvvar_model_order" size="small" :min="1" :max="20" :precision="0" placeholder="Enter a value from 1 to 10"></iview-input-number>
                </iview-form-item>
                <iview-form-item label="Window width" slot="content">
                    <iview-input-number v-model="model.tvvar_window_width" size="small" :min="2" :max="1000" :precision="0" placeholder="Enter a value from 2 to 1000"></iview-input-number>
                </iview-form-item>
                <iview-form-item label="Window shift" slot="content">
                    <iview-input-number v-model="model.tvvar_window_shift" size="small" :min="0" :max="100" :precision="0" placeholder="Enter a value from 0 to 100"></iview-input-number>
                </iview-form-item>
            </iview-panel>
        </iview-collapse>

        <iview-collapse v-model="collapsedClustering">
            <iview-panel name="1">
                Clustering settings
                <iview-form-item label="Number of states" slot="content">
                    <iview-input-number v-model="model.cluster_number" size="small" :min="1" :max="100" :precision="0" placeholder="Enter a value from 1 to 100"></iview-input-number>
                </iview-form-item>
            </iview-panel>
        </iview-collapse>

        <iview-collapse v-model="collapsedEM">
            <iview-panel name="1">
                Expectation-Maximization settings
                <iview-form-item label="Tolerance error" slot="content">
                    <iview-input-number v-model="model.em_tolerance_error" size="small" :min="0.000000001" :max="1" :step="0.01" :value="0.00000001" :precision="10" placeholder="Enter a value from 1 to 100"></iview-input-number>
                </iview-form-item>
                <iview-form-item label="Maximum number of iterations" slot="content">
                    <iview-input-number v-model="model.em_iterations"  size="small" :min="1" :max="100" :precision="0" placeholder="Enter a value from 1 to 10"></iview-input-number>
                </iview-form-item>
            </iview-panel>
        </iview-collapse>

        <iview-collapse v-model="collapsedSurface">
            <iview-panel name="1">
                Specific graphic
                <iview-form-item label="Graphic surface model" slot="content">
                    <file-input v-model="model.surface_pattern"></file-input>
                </iview-form-item>
                <iview-form-item label="ROI column information" slot="content">
                    <file-input v-model="model.surface_roi_names"></file-input>
                </iview-form-item>
            </iview-panel>
        </iview-collapse>

    </div>
</template>

<script>

class ApplicationSVARModel{
    constructor(){
        /*
        this.tvvar_model_order = 1
        this.tvvar_window_shift = 1
        this.tvvar_window_width = 100
        this.cluster_number = 3
        this.em_iterations = 15
        this.em_tolerance_error = 1e-10
        this.surface_pattern = ''
        this.surface_roi_names = ''
        */
        this.defaultValues = {
            tvvar_model_order: 1,
            tvvar_window_shift: 1,
            tvvar_window_width: 100,
            cluster_number: 3,
            em_iterations: 15,
            em_tolerance_error: 1e-10,
            surface_pattern: '',
            surface_roi_names: '',
        }
        this.resetValues()
    }
    resetValues(){
        var additionalArguments = Array.prototype.slice.call(arguments);
        for(let key in this.defaultValues){
            this[key] = this.defaultValues[key]
        }
    }
    toCommandLineArgs(){
        var additionalArguments = Array.prototype.slice.call(arguments);
        var commandArgs  = []
        var addCommands = (data, isMember) => {
            for(let key in data){
                commandArgs.push(`--${key.replace('_','-')}="${isMember? this[key]: data[key]}"`)
            }
        }
        addCommands(this.defaultValues)
        for (let i = 0; i < additionalArguments.length; i++) {
            const element = additionalArguments[i];
            addCommands(element)
        }
        return commandArgs//.join(" ")
    }
}

var ApplicationSVARSettings = {
    template: iview.getTagNamesFromElement(document.currentScript.previousElementSibling),
    props: {
        'type':{
            type: String,
            default: 'single',
        }
    },
    data(){
        return {
            collapsedTVVAR: '1',
            collapsedClustering: '1',
            collapsedEM: '1',
            collapsedSurface: '1',
            //
            //model: ApplicationSVARBaseModel(),
            model: svarstate[this._props.type.toLowerCase() + "Settings"],
        }
    },
    methods: {
        collapseEverything(){
            this.collapsedTVVAR = true
            this.collapsedClustering = true
            this.collapsedEM = true
            this.collapsedSurface = true
        },
        resetFields(){
            //this.model = ApplicationSVARBaseModel()
            this.model.resetValues()
            this.datasetPath = ''
            this.datasetMatlabField = ''
        },
        getFieldValues(){
            return [
                ["TV-VAR model order", this.model.tvvar_model_order],
                ["TV-VAR window shift", this.model.tvvar_window_shift],
                ["TV-VAR window width", this.model.tvvar_window_width],
                ["Number of clusters in K-means", this.model.cluster_number],
                ["Number of iterations in EM", this.model.em_iterations],
                ["Error tolerance in EM", this.model.em_tolerance_error],
                ["File path of the surface pattern", this.model.surface_pattern],
                ["File path of the surface ROI names", this.model.surface_roi_names],
            ]
        },
    },
    computed: {
    },
    watch: {
    }
}
Vue.component('app-svar-settings', ApplicationSVARSettings)


</script>
