<template>
    <div class="app-title-tab batch-model">
        <iview-tabs type="card" :animated="false">
            <iview-tab-pane label="Analysis configuration">
                <!--
                <iview-table class="single-settings-header" style="border:none;" stripe :columns="headerColumns" :data="headerData">
                </iview-table>
                -->

                <iview-form :label-width="250" class="single-settings">
                    <app-svar-settings ref="settings">
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
                                <file-input v-model="datasetPath"></file-input>
                            </iview-form-item>

                            <iview-form-item label=".MAT field name" slot="content" v-if="datasetPath.toLowerCase().endsWith('.mat')">
                                <iview-input v-model="datasetMatlabField"
                                placeholder="Select a field name" ></iview-input>
                            </iview-form-item>
                                        
                            <iview-form-item label="" slot="content" class="submit-buttons">
                                <iview-button size="small" type="primary"
                                :disabled="datasetPath.trim()==''"
                                 @click.prevent.stop="insert"> Include dataset </iview-button>
                            </iview-form-item>

                        </iview-panel>
                    </iview-collapse>

                    <i-table class="batch-settings-header" style="border:none;" border :columns="columns7" :data="data6"></i-table>

                    <iview-form-item class="submit-buttons">
                        <iview-button size="small" type="ghost" @click.prevent.stop="resetFields"> Reset fields</iview-button>
                        <iview-button size="small" type="primary" @click.prevent.stop="runModel"> Run model </iview-button>
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
                            }, basename(params.row.path))
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
                            }, datasetType(params.row.path))
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
                                value: params.row.path,
                                readonly: "readonly"
                                },
                                class: 'complete-path-entry',
                            }, )
                        ]);
                    }
                },
                {
                    title: 'Field name',
                    key: 'fieldname',
                    render: (h, params) => {
                        return params.row.fieldname? h('div', [
                            h('span', {
                                props: {
                                },
                                class: 'field-entry',
                            }, params.row.fieldname)
                        ]): "";
                    }
                },
            ],
            data6: [
                {
                    fieldname: 'John Brown',
                    path: '/a/b/f/t/g.mat'
                },
                {
                    fieldname: '',
                    path: '/a/b/f/t//a/b/f/t/.mat/a/b/f/t/.mat/a/b/f/t/.mat/a/b/f/t/r.mat'
                },
                {
                    fieldname: 'Joe Black',
                    path: '/a/b/f/t/f.mat'
                },
                {
                    fieldname: 'Jon Snow',
                    path: 'Ottawa No. 2 Lake Park/a/b/fOttawa No. 2 Lake Park/a/b/fOttawa No. 2 Lake Park/a/b/fOttawa No. 2 Lake Park/a/b/fOttawa No. 2 Lake Park/a/b/f/t/knhbjkl.mat'
                }
            ],
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
            //
        },

        insert () {
            this.data6.push(
                {
                    path: this.datasetPath,
                    fieldname: this.datasetMatlabField,
                })
        },

        show (index) {
            var message = "";
            var fields = this.$refs.settings.getFieldValues()
            fields.splice(0, ["Type", datasetType(this.data6[index].path)])
            fields.push(["Path", this.data6[index].path])
            fields.push(["MATLAB field name", this.data6[index].fieldname])
            for (let i = 0; i < fields.length; i++) {
                const element = fields[i];
                if(element[1].toString().trim() == '') continue
                message += `<strong>${element[0]}</strong>: ${element[1]} <br/>`
            }
            this.$Modal.info({
                title: `Dataset ${basename(this.data6[index].path)}`,
                content: message
            })
            /*
            this.$Modal.info({
                title: `Dataset ${basename(this.data6[index].path)}`,
                content: `
                <strong>Type</strong>: ${datasetType(this.data6[index].path)}
                <br/>
                <strong>Path</strong>: ${this.data6[index].path}
                <br/>
                <strong>MATLAB field name</strong> (if applicable): ${this.data6[index].fieldname}
                `
            })
            */
        },
        remove (index) {
            this.data6.splice(index, 1);
        }
    },
    computed: {
    },
    watch: {
        datasetPath(ne){
            console.log(this)
            console.log(ne)
        }
    }
}
Vue.component('app-svar-batch-settings', ApplicationSVARBatchSettings)


</script>
