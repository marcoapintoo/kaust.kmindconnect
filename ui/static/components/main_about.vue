<template>
    <div class="about">
        <markdown-element>

## kMindConnect 0.2.0a
![kMindConnect](./images/icon.svg)
## Dynamic Brain Connectivity Platform

Project source code: [bitbucket.org/tonitruum/kaust.kmindconnect](https://bitbucket.org/tonitruum/kaust.kmindconnect)

[Currently private repository. Coming soon in github]

Group research homepage: [biostats.kaust.edu.sa/](https://biostats.kaust.edu.sa/)

        </markdown-element>
    </div>
</template>

<script>

var ApplicationAbout = {
    template: iview.getTagNamesFromElement(document.currentScript.previousElementSibling),
    data(){
        return {
        }
    },
    methods: {
        /*compileDocumentMarkdown: _.debounce( (e) => {
            this.helpDocument = marked(e.target.value, { sanitize: true })
        }, 300),*/
    },
    computed: {
    }
}
Vue.component('app-about', ApplicationAbout)

</script>
