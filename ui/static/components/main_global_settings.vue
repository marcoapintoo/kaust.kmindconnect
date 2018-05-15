<template>
    <div class="app-title-tab">
        <iview-tabs type="card">
            <iview-tab-pane label="Application configuration">

                <!--
                <iview-table class="single-settings-header" style="border:none;" stripe :columns="headerColumns" :data="headerData">
                </iview-table>
                -->

                <iview-form :label-width="250" class="single-settings">
                    
                    <iview-collapse v-model="collapsedModelEngine">
                        <iview-panel name="1">
                            Executable settings
                            <iview-form-item label="Python runtime path" slot="content">
                                <file-input v-model="model.path_python" :readonly="false"></file-input>
                            </iview-form-item>
                            <iview-form-item label="Octave runtime path" slot="content">
                                <file-input v-model="model.path_octave" :readonly="false"></file-input>
                            </iview-form-item>
                            <iview-form-item label="Python script path" slot="content">
                                <file-input v-model="model.path_script_python" :readonly="false"></file-input>
                            </iview-form-item>
                            <iview-form-item label="MATLAB script path" slot="content">
                                <file-input v-model="model.path_script_octave" :readonly="false"></file-input>
                            </iview-form-item>
                        </iview-panel>
                    </iview-collapse>

                    <iview-collapse v-model="collapsedModelEngine">
                        <iview-panel name="1">
                            Simulation preferences
                            <iview-form-item label="Output path" slot="content">
                                <file-input v-model="model.path_output" :readonly="false"></file-input>
                            </iview-form-item>
                            <iview-form-item label="Simulation engine" slot="content">
                                <i-switch class="large-switch" :value="model.matlab_engine">
                                    <span type="android-done" slot="open">OCTAVE</span>
                                    <span type="android-close" slot="close">PYTHON</span>
                                </i-switch>
                            </iview-form-item>
                        </iview-panel>
                    </iview-collapse>

                </iview-form>
            </iview-tab-pane>
        </iview-tabs>



    </div>
</template>

<script>


var ApplicationGlobalSettings = {
    template: iview.getTagNamesFromElement(document.currentScript.previousElementSibling),
    data(){
        return {
            collapsedModelEngine: '1',
            //
            model: svarstate.globalSettings,
        }
    },
    methods: {
        resetFields(){
            //this.model = ApplicationSVARBaseModel()
            this.model.resetGlobalValues()
            //this.datasetPath = ''
            //this.datasetMatlabField = ''
        },
        getFieldValues(){
            return [
                //["File path of the surface ROI names", this.model.surface_roi_names],
            ]
        },
    },
    created(){
    },
    updated(){
        console.log()
        //this.$parent.$emit('input-changed', this, this.$el.querySelector("input").value)
    },
    computed: {
    },
    watch: {
    }
}
Vue.component('app-global-settings', ApplicationGlobalSettings)


</script>
