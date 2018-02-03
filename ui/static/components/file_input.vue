
<template>
    <span class="file-input">
        <!--
        <input\
        ref="input"\
        v-bind:value="value"\
        v-on:input="updateValue($event.target.value)">\
        -->
    <iview-button type="ghost" size="small" icon="android-folder" @click="loadFile"></iview-button>
    <iview-input ref="input" :value="value" size="small" placeholder="Select a file" :readonly="true"></iview-input>
    </span>
</template>

<script>

var FileInputElement = {
    template: iview.getTagNamesFromElement(document.currentScript.previousElementSibling),
    props: ['value'],
    data(){
        return {
        }
    },
    methods: {
        loadFile(){
            try {
                const { dialog } = require('electron').remote;
                try{
                    dialog.showOpenDialog((filepath) => {
                        console.log(filepath);
                        if (filepath === undefined) {
                            this.$emit('input', "")
                            return;
                        }
                        this.$emit('input', filepath[0])
                    });
                } catch (error) {
                    var err = new Error();
                    this.$Modal.warning({
                        title: "Execution error",
                        content: err + "\n" + err.stack
                    });
                }
            } catch (e) {
                this.$Modal.warning({
                    title: "Disabled function",
                    content: "This function has been disabled for browser-navigation."
                });
                this.$emit('input', "")
            }
            
        }
    },
    computed: {
    }
}

Vue.component('file-input', FileInputElement)

</script>
