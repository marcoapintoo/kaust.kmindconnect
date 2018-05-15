<template>
    <div class="app-title-tab app-expand-screen">
        <iview-tabs type="card" :animated="false">
            <iview-tab-pane label="Visual results" class="no-scroll">
                <app-svar-results
                    ref="results"
                    :outputPath="modelGlobal.path_output"
                    :hashCode="modelGraph.hashCode"
                    :states="modelGraph.cluster_number"
                    :showBrains="modelGraph.surface_pattern!=''"
                ></app-svar-results>
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
