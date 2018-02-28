
<template>
    <span class="file-input">
        <!--
        <input\
        ref="input"\
        v-bind:value="value"\
        v-on:input="updateValue($event.target.value)">\
        -->
    <iview-button type="ghost" size="small" icon="android-folder" @click="loadFile"></iview-button>
    
    <iview-input @on-keyup="changes" ref="input" :value="value" size="small" placeholder="Select a file" :readonly="readonly"></iview-input>
   
</template>

<script>

var FileInputElement = {
    template: iview.getTagNamesFromElement(document.currentScript.previousElementSibling),
    //props: ['value'],
    props: {
        'value': String,
        'readonly': {
            type: Boolean,
            default: true
        }
    },
    data(){
        return {
            //readonly: true
        }
    },
    methods: {
        changes(e){
            this.$emit('input', this.$el.querySelector("input").value)
        },
        loadFile(){
            var oldValue = this.$el.querySelector("input").value;
            try {
                const { dialog } = require('electron').remote;
                try{
                    dialog.showOpenDialog((filepath) => {
                        //console.log(filepath);
                        if (filepath === undefined) {
                            this.$emit('input', oldValue)
                            return;
                        }
                        if(filepath[0]) this.$emit('input', filepath[0])
                        //else this.$emit('input', oldValue)
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
                this.$emit('input', oldValue)
            }
        }
    },
    computed: {
        
    }
}

Vue.component('file-input', FileInputElement)

</script>
