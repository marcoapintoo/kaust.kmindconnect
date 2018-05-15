<template>
    <div class="app-title-tab app-expand-screen">
        <iview-tabs type="card" :animated="false">
            <iview-tab-pane label="Visual results" class="no-scroll">
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
                        <iview-button @click="activeResult='brain-graphics'" v-if="modelGraph.surface_pattern!=''">
                            <iview-icon type="aperture"></iview-icon>
                            Brain coherence
                        </iview-button>
                        <iview-button @click="activeResult='brain-animation'" v-if="modelGraph.surface_pattern!=''">
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
                        :src="outputPath + '/centroids.html'"
                        ></iframe>
                    </iview-tab-pane>
                    <iview-tab-pane name="states-signal">
                        <iframe
                        :src="outputPath + '/kalman_states_smoothed.html'"
                        ></iframe>
                    </iview-tab-pane>
                    <iview-tab-pane name="states-matrices">
                        <div class="content" v-for="n in rangeStates" :key="n.id">
                            <div class="state">
                                <div class="title toolbar app-toolbar">State {{n.id}}</div>
                                <iframe :src="outputPath + '/kalman_estimated_coherence-state-' + n.id + '.html'"></iframe>
                            </div>
                        </div>
                    </iview-tab-pane>
                    <iview-tab-pane name="brain-graphics">
                        <div class="content" v-for="n in rangeStates" :key="n.id">
                            <div class="state">
                                <div class="title toolbar app-toolbar">State {{n.id}}</div>
                                <img :src="outputPath + '/coherence_state_' + n.id + '.svg'" />
                            </div>
                        </div>
                    </iview-tab-pane>
                    <iview-tab-pane name="brain-animation">
                        <iframe
                        src=""
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

var ApplicationSVARSingleResults = {
    template: iview.getTagNamesFromElement(document.currentScript.previousElementSibling),
    props: {
        type:{
            type: String,
            default: 'single',
        }
    },
    data(){
        return {
            activeResult: 'clusters',
            //
            modelGlobal: svarstate.globalSettings,
            modelGraph: svarstate[this._props.type.toLowerCase() + "Settings"],
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
            this.$root.$emit('execution-start', this, "single-svar", commandLine);
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
        rangeStates(){
            console.log(";;;;;", this.listFiles)
            x = []
            for (let i = 0; i < this.modelGraph.cluster_number; i++) {
                x.push({id: i + 1})
            }
            return x
        },
        outputPath(){
            let preffix = ""
            if(this.modelGlobal.path_output.startsWith("./")){
                preffix = "../"
            }
            return preffix + this.modelGlobal.path_output + "/" + this.modelGraph.hashCode + "/"
        },
        disableScreenClass(){
            return [
                "block-screen",
                (this.panelDisabled? "enabled": "disabled")
            ]
        },
        listFiles(){
            try {
                let listFiles = []
                const electron = require("electron").remote
                try {
                    const fs = electron.require('fs');
                    const files = fs.readdirSync(this.modelGlobal.path_output)
                    files.forEach((file) => {
                        let path = this.modelGlobal.path_output + "/" + file + "/experiment.json"
                        if (fs.existsSync(path)) {
                            var values = JSON.parse(fs.readFileSync(path, 'utf8'))
                            var filename = values["<datasets>"][0].replace(/.*?\//g, "")
                            console.log(filename, path)
                        }
                    });
                } catch (err) {
                    console.log(err);
                }
            } catch (e) {
                console.log("Function disabled in browser-mode")
            }
        }
    },
    watch: {
        datasetPath(ne){
            console.log(this)
            console.log(ne)
        }
    }
}
Vue.component('app-svar-single-results', ApplicationSVARSingleResults)


</script>
