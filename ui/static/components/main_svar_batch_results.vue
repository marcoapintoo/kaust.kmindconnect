<template>
    <div class="app-title-tab app-expand-screen">
        <iview-tabs type="card" :animated="false">
            <iview-tab-pane label="Visual results" class="no-scroll">
                <div class="app-toolbar">
                    <iview-select v-model="filePathResults" style="width:98%; text-align: left;">
                        <iview-option value="New York" label="New York">
                            <span>New York</span>
                            <span style="float:right;color:#ccc">America</span>
                        </iview-option>
                        <iview-option value="London" label="London">
                            <span>London</span>
                            <span style="float:right;color:#ccc">U.K.</span>
                        </iview-option>
                        <iview-option value="Sydney" label="Sydney">
                            <span>Sydney</span>
                            <span style="float:right;color:#ccc">Australian</span>
                        </iview-option>
                    </iview-select>
                </div>
                <div class="app-toolbar">
                    <iview-button-group shape="circle" size="small">
                        <iview-button @click="activeResult='clusters'">
                            <iview-icon type="ios-color-filter"></iview-icon>
                            Clusters (PCA)
                        </iview-button>
                        <iview-button @click="activeResult='states-signal'">
                            <iview-icon type="podium"></iview-icon>
                            Connectivity states
                        </iview-button>
                        <iview-button @click="activeResult='states-matrices'">
                            <iview-icon type="ios-grid-view"></iview-icon>
                            Coefficient matrices
                        </iview-button>
                        <iview-button @click="activeResult='brain-graphics'">
                            <iview-icon type="aperture"></iview-icon>
                            Brain coherence
                        </iview-button>
                        <iview-button @click="activeResult='brain-animation'">
                            <iview-icon type="videocamera"></iview-icon>
                            Connectivity evolution
                        </iview-button>
                    </iview-button-group>
                </div>
                <!--
                <iview-table class="single-settings-header" style="border:none;" stripe :columns="headerColumns" :data="headerData">
                </iview-table>
                -->
                <iview-tabs type="card" :value="activeResult" class="graphical-results" :animated="false">
                    <iview-tab-pane name="clusters">
                        <iframe
                        src="file:///Users/pinto.marco/KAUSTProjects/TonitruumUI0/kEEGConnect/python/graphical-outputs/results-6251_mean_fs/centroids.html"
                        ></iframe>
                    </iview-tab-pane>
                    <iview-tab-pane name="states-signal">
                        <iframe
                        src="file:///Users/pinto.marco/KAUSTProjects/TonitruumUI0/kEEGConnect/python/graphical-outputs/results-6251_mean_fs/kalman_states_smoothed.html"
                        ></iframe>
                    </iview-tab-pane>
                    <iview-tab-pane name="states-matrices">
                        <div class="content"><div class="state"><div class="title toolbar">State 1</div><iframe src="file:///Users/pinto.marco/KAUSTProjects/TonitruumUI0/kEEGConnect/python/graphical-outputs/results-6251_mean_fs/kalman_estimated_coherence-state-1.html"></iframe></div><div class="state"><div class="title toolbar">State 2</div><iframe src="file:///Users/pinto.marco/KAUSTProjects/TonitruumUI0/kEEGConnect/python/graphical-outputs/results-6251_mean_fs/kalman_estimated_coherence-state-2.html"></iframe></div><div class="state"><div class="title toolbar">State 3</div><iframe src="file:///Users/pinto.marco/KAUSTProjects/TonitruumUI0/kEEGConnect/python/graphical-outputs/results-6251_mean_fs/kalman_estimated_coherence-state-3.html"></iframe></div></div>
                    </iview-tab-pane>
                    <iview-tab-pane name="brain-graphics">
                        <div class="content"><div class="state"><div class="title toolbar">State 1</div><img src="file:///Users/pinto.marco/KAUSTProjects/TonitruumUI0/kEEGConnect/python/graphical-outputs/results-6251_mean_fs/coherence_state_1.svg"></div><div class="state"><div class="title toolbar">State 2</div><img src="file:///Users/pinto.marco/KAUSTProjects/TonitruumUI0/kEEGConnect/python/graphical-outputs/results-6251_mean_fs/coherence_state_2.svg"></div><div class="state"><div class="title toolbar">State 3</div><img src="file:///Users/pinto.marco/KAUSTProjects/TonitruumUI0/kEEGConnect/python/graphical-outputs/results-6251_mean_fs/coherence_state_3.svg"></div></div>
                    </iview-tab-pane>
                    <iview-tab-pane name="brain-animation">
                        <iframe
                        src="file:///Users/pinto.marco/KAUSTProjects/TonitruumUI0/kEEGConnect/ui/animation.html?path=/Users/pinto.marco/KAUSTProjects/TonitruumUI0/kEEGConnect/python/graphical-outputs/results-6251_mean_fs&codename="
                        ></iframe>
                    </iview-tab-pane>
                </iview-tabs>

                <!--
                -->
            </iview-tab-pane>
        </iview-tabs>
    </div>
</template>

<script>

var ApplicationSVARBatchResults = {
    template: iview.getTagNamesFromElement(document.currentScript.previousElementSibling),
    data(){
        return {
            filePathResults: '',
            //
            activeResult: 'clusters',
            //
            collapsedDataset: '1',
            //
            datasetPath: '',
            datasetMatlabField: '',
            //
            headerColumns: [
                {
                    title: 'Property',
                    key: 'name',
                    width: 260,
                },
                {
                    title: 'Value',
                    key: 'value'
                }
            ],
            headerData: [
            ],
            panelDisabled: false,
        }
    },
    methods: {
        handleTabRemove(){
            //
        },
        resetFields(){
            this.$refs.settings.resetFields()
            this.datasetPath = ''
            this.datasetMatlabField = ''
        },
        runModel(){
            const commandLine = this.$refs.settings.model.toCommandLineArgs({
                input_path: this.datasetPath,
                matlab_field_name: this.datasetMatlabField,
            })
            this.$root.$emit('execution-start', this, "svar", commandLine);
            this.panelDisabled = true;
        },
        modelFinished(status, errorData){
            try{
                status = (status || "ok").toLowerCase();
                errorData = errorData || {}
                this.panelDisabled = false;
                if(status != "ok"){
                    this.$Modal.error({
                        title: `Execution error`,
                        content: `
                        <strong>Status</strong>: ${status}
                        <br/>
                        <strong>Error details</strong>: ${JSON.stringify(errorData)}
                        `
                    })
                }
            } finally{
                this.panelDisabled = false;
                setTimeout(() => {
                    this.panelDisabled = false;
                }, 100);
            }
        }
    },
    created(){
        this.$on('execution-finished', this.modelFinished);
    },
    computed: {
        disableScreenClass(){
            return [
                "block-screen",
                (this.panelDisabled? "enabled": "disabled")
            ]
        }
    },
    watch: {
        datasetPath(ne){
            console.log(this)
            console.log(ne)
        }
    }
}
Vue.component('app-svar-batch-results', ApplicationSVARBatchResults)


</script>
