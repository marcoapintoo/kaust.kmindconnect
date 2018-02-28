<template>
    <div class="app-title-tab">
        <iview-tabs type="card">
            <iview-tab-pane label="Single-subject analysis configuration">

                <!--
                <iview-table class="single-settings-header" style="border:none;" stripe :columns="headerColumns" :data="headerData">
                </iview-table>
                -->

                <iview-form :label-width="250" class="single-settings">
                    
                    <div :class="disableScreenClass">
                    </div>
                    <app-svar-settings ref="settings">
                    </app-svar-settings>

                    <iview-collapse v-model="collapsedDataset">
                        <iview-panel name="1">
                            Data sources
                            <iview-form-item label="Path file" slot="content">
                                <file-input v-model="model.input_path"></file-input>
                            </iview-form-item>
                            <iview-form-item label=".MAT field name" slot="content" v-if="model.input_path.toLowerCase().endsWith('.mat')">
                                <iview-input size="small" v-model="model.matlab_field_name" placeholder="Select a file"></iview-input>
                            </iview-form-item>
                        </iview-panel>
                    </iview-collapse>

                    <iview-form-item class="submit-buttons">
                        <iview-button size="small" type="ghost" @click.prevent.stop="resetFields"> Reset fields</iview-button>
                        <iview-button size="small" type="primary" @click.prevent.stop="runModel" :disabled="model.input_path==''"> Run model </iview-button>
                    </iview-form-item>

                </iview-form>
            </iview-tab-pane>
        </iview-tabs>
    </div>
</template>

<script>

var ApplicationSVARSingleSettings = {
    template: iview.getTagNamesFromElement(document.currentScript.previousElementSibling),
    data(){
        return {
            collapsedDataset: '1',
            //
            model: svarstate['singleSettings'],
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
            svarstate.resetSingleValues()
        },
        runModel(){
            svarstate.execute(this, 'single')
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
}
Vue.component('app-svar-single-settings', ApplicationSVARSingleSettings)


</script>
