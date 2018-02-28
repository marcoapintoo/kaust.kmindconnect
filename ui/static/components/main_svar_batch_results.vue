<template>
    <div class="app-title-tab app-expand-screen">
        <iview-tabs type="card" :animated="false">
            <iview-tab-pane label="Model results explorer" class="no-scroll">
                <div class="app-toolbar">
                    
                    <iview-select v-model="filePathResults" style="width:98%; text-align: left;">
                        <iview-option :value="datafile.hashcode" :label="datafile.basename" v-for="datafile in listFiles" :key="datafile.id">
                            <span>{{datafile.basename}}</span>
                            <span style="float:right;color:#ccc">Experiment ID: {{datafile.hashcode}}</span>
                        </iview-option>
                    </iview-select>
                    
                </div>
                <app-svar-results
                    :outputpath="modelGlobal.path_output"
                    :hashcode="!filePathResults?'':dictFiles[filePathResults].hashcode"
                    :states="!filePathResults?'':dictFiles[filePathResults].states"
                    :showbrains="!filePathResults?'':dictFiles[filePathResults].show_brains!=''"
                ></app-svar-results>
            </iview-tab-pane>
        </iview-tabs>
    </div>
</template>

<script>

var ApplicationSVARBatchResults = {
    template: iview.getTagNamesFromElement(document.currentScript.previousElementSibling),
    props: {
        type:{
            type: String,
            default: 'single',
        }
    },
    data(){
        return {
            filePathResults: '',
            //
            activeResult: 'clusters',
            //
            availableFiles: [],
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
            //
            listFiles:[],
            dictFiles:{},
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
        },
        fetchList(){
            let tempListFile = []
            setTimeout(() => {
                tempListFile = this._fetchListFiles();
                this.listFiles = tempListFile;
                for (let i = 0; i < tempListFile.length; i++) {
                    const element = tempListFile[i];
                    this.dictFiles[element.hashcode] = element
                }
            }, 3000);
        },
        _fetchListFiles(){
            try {
                let availableFiles = []
                const electron = require("electron").remote
                try {
                    const fs = electron.require('fs');
                    const files = fs.readdirSync(this.modelGlobal.path_output)
                    files.forEach((file) => {
                        let path = this.modelGlobal.path_output + "/" + file + "/experiment.json"
                        if (fs.existsSync(path)) {
                            var values = JSON.parse(fs.readFileSync(path, 'utf8'))
                            var filename = values["<datasets>"][0].replace(/.*?\//g, "")
                            availableFiles.push({
                                id: availableFiles.length,
                                basename: filename, 
                                hashcode: file, 
                                show_brains: values['--brain-surface'], 
                                states: values['--number-states'], 
                                //experiment: values
                            })
                            console.log(filename, path)
                        }
                    });
                    console.log(availableFiles)
                    this.availableFiles = availableFiles
                    return availableFiles
                } catch (err) {
                    console.log(err);
                }
            return []
            } catch (e) {
                console.log("Function disabled in browser-mode")
            }
            return []
        }
    },
    created(){
        this.$on('execution-finished', this.modelFinished);
        this.fetchList()
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
    },
    watch: {
    }
}
Vue.component('app-svar-batch-results', ApplicationSVARBatchResults)


</script>
