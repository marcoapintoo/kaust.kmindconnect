<template>
    <div class="app-title-tab batch-model">
        <iview-tabs type="card" :animated="false">
            <iview-tab-pane label="Analysis configuration">
                <!--
                <iview-table class="single-settings-header" style="border:none;" stripe :columns="headerColumns" :data="headerData">
                </iview-table>
                -->

                <iview-form :label-width="250" class="single-settings">
                    <app-svar-settings ref="settings" type="batch">
                    </app-svar-settings>
                </iview-form>

            </iview-tab-pane>

            <iview-tab-pane label="Batch analysis">
                <!--
                <iview-table class="single-settings-header" style="border:none;" stripe :columns="headerColumns" :data="headerData">
                </iview-table>
                -->

                <iview-form :label-width="250" class="single-settings">

                    <iview-collapse v-model="collapsedDataset">
                        <iview-panel name="1">
                            Data sources
                            
                            <iview-form-item label="Path file" slot="content">
                                <file-input v-model="model.input_path"></file-input>
                            </iview-form-item>

                            <iview-form-item label=".MAT field name" slot="content" v-if="model.input_path.toLowerCase().endsWith('.mat')">
                                <iview-input v-model="model.matlab_field_name"
                                placeholder="Select a field name" ></iview-input>
                            </iview-form-item>
                                        
                            <iview-form-item label="" slot="content" class="submit-buttons">
                                <iview-button size="small" type="primary"
                                :disabled="model.input_path.trim()==''"
                                 @click.prevent.stop="insert"> Include dataset </iview-button>
                            </iview-form-item>

                        </iview-panel>
                    </iview-collapse>

                    <i-table class="batch-settings-header" style="border:none;" border :columns="columns7" :data="model.input_paths"></i-table>

                    <iview-form-item class="submit-buttons">
                        <iview-button size="small" type="ghost" @click.prevent.stop="resetFields"> Reset fields</iview-button>
                        <iview-button size="small" type="primary" @click.prevent.stop="runModel" :disabled="model.input_paths.length==0" > Run model </iview-button>
                    </iview-form-item>
                </iview-form>
            </iview-tab-pane>
        </iview-tabs>
    </div>
</template>

<script>

function basename(path) {
     return path.replace( /.*\//, "" );
}
function datasetType(path) {
    let ext = path.toLowerCase().replace( /.*\./, "" );
    if(ext == 'mat'){
        return 'MATLAB data format'
    }else if(ext == 'csv'){
        return 'CSV file'
    }else if(ext == 'json'){
        return 'JSON file'
    }else{
        return "Unknown data format!"
    }
}

var ApplicationSVARBatchSettings = {
    template: iview.getTagNamesFromElement(document.currentScript.previousElementSibling),
    data(){
        return {
            collapsedDataset: '1',
            //
            model: svarstate['batchSettings'],
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

            //

            columns7: [
                {
                    title: ' ',
                    key: 'operations',
                    width: 70,
                    align: 'center',
                    render: (h, params) => {
                        return h('div', {
                            style: {
                                display: 'flex',
                            },
                            class: "entry-operations",
                        },[
                            h('Button', {
                                props: {
                                    type: 'primary',
                                    size: 'small',
                                    shape: 'circle',
                                    icon: 'search'
                                },
                                style: {
                                    marginRight: '5px'
                                },
                                on: {
                                    click: () => {
                                        this.show(params.index)
                                    }
                                }
                            }),
                            h('Button', {
                                props: {
                                    type: 'error',
                                    size: 'small',
                                    shape: 'circle',
                                    icon: 'close-round'
                                },
                                on: {
                                    click: () => {
                                        this.remove(params.index)
                                    }
                                }
                            })
                        ]);
                    }
                },
                {
                    title: 'Filename',
                    key: 'pathname',
                    render: (h, params) => {
                        return h('div', [
                            h('Icon', {
                                props: {
                                    type: 'document'
                                }
                            }),
                            h('span', {
                                props: {
                                },
                                class: 'reduced-path-entry',
                            }, basename(params.row.input_path))
                        ]);
                    }
                },
                {
                    title: 'Dataset type',
                    key: 'datatype',
                    render: (h, params) => {
                        return h('div', [
                            h('span', {
                                props: {
                                },
                                class: 'type-path-entry',
                            }, datasetType(params.row.input_path))
                        ]);
                    }
                },
                {
                    title: 'File path',
                    key: 'path',
                    render: (h, params) => {
                        return h('div', [
                            h('Input', {
                                props: {
                                value: params.row.input_path,
                                readonly: "readonly"
                                },
                                class: 'complete-path-entry',
                            }, )
                        ]);
                    }
                },
                {
                    title: 'Field name',
                    key: 'matlab_field_name',
                    render: (h, params) => {
                        return params.row.matlab_field_name? h('div', [
                            h('span', {
                                props: {
                                },
                                class: 'field-entry',
                            }, params.row.matlab_field_name)
                        ]): "";
                    }
                },
            ],
        }
    },
    methods: {
        handleTabRemove(){
            //
        },
        resetFields(){
            svarstate.resetBatchValues()
        },
        runModel(){
            svarstate.execute(this, "batch")
        },

        insert () {
            this.model.input_paths.push(
            {
                input_path: this.model.input_path,
                matlab_field_name: this.model.matlab_field_name,
            })
        },

        show (index) {
            var message = "";
            var fields = this.$refs.settings.getFieldValues()
            fields.splice(0, ["Type", datasetType(this.model.input_paths[index].input_path)])
            fields.push(["Path", this.model.input_paths[index].input_path])
            fields.push(["MATLAB field name", this.model.input_paths[index].matlab_field_name])
            for (let i = 0; i < fields.length; i++) {
                const element = fields[i];
                if(element[1].toString().trim() == '') continue
                message += `<strong>${element[0]}</strong>: ${element[1]} <br/>`
            }
            this.$Modal.info({
                title: `Dataset ${basename(this.model.input_paths[index].input_path)}`,
                content: message
            })
            /*
            this.$Modal.info({
                title: `Dataset ${basename(this.input_paths[index].path)}`,
                content: `
                <strong>Type</strong>: ${datasetType(this.input_paths[index].path)}
                <br/>
                <strong>Path</strong>: ${this.input_paths[index].path}
                <br/>
                <strong>MATLAB field name</strong> (if applicable): ${this.input_paths[index].matlab_field_name}
                `
            })
            */
        },
        remove (index) {
            this.model.input_paths.splice(index, 1);
        }
    },
    computed: {
    },
    watch: {
        input_path(ne){
            console.log(this)
            console.log(ne)
        }
    }
}
Vue.component('app-svar-batch-settings', ApplicationSVARBatchSettings)


</script>
