<template id="s">
<div>
    <div class="document-markdown markdown">
        <div id="primarySlot" style="display:none"><slot></slot></div>
        <div v-html="helpDocumentMarkdown"></div>
        <!--
        <textarea id="helpDocume3nt" @input="compileDocumentMarkdown" :value="hiddenText"></textarea>
        <div v-html="helpDocumentMarkdown"></div>
        -->
    </div>
    </div>
</template>

<script>

var MarkdownElement = {
    template: iview.getTagNamesFromElement(document.currentScript.previousElementSibling),
    data(){
        return {
           //helpDocument: ''
        }
    },
    methods: {
        /*compileDocumentMarkdown: _.debounce( (e) => {
            this.helpDocument = marked(e.target.value, { sanitize: true })
        }, 300),*/
    },
    computed: {
        hiddenText(){
            console.log(this.primarySlot.innerHTML)
            return "";
            return this.$slots.default[0].text;
        },
        helpDocumentMarkdown() {
            // The slot text must be at least one non-empty line 
            let markdown = this.$slots.default[0].text;
            let lines = markdown.split("\n")
            let startWS = 1e1000;
            for(let i = 0; i < lines.length; i++){
                //if(lines[i] == "") continue;
                if(lines[i].trim() == "") continue;
                startWS = Math.min(startWS, lines[i].search(/\S/))
            }
            if(startWS > 0){
                for(let i = 0; i < lines.length; i++){
                    if(lines[i] == "") continue;
                    lines[i] = lines[i].substring(startWS)
                }
            }
            markdown = lines.join("\n");
            let md = markdown_engine()
            return md.render(markdown);
        },
    }
}
Vue.component('markdown-element', MarkdownElement)

</script>
