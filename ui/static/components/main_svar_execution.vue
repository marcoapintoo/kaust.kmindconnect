<template>
    <div class="app-title-tab app-expand-screen app-log">
        <iview-tabs type="card" :animated="false">
            <iview-tab-pane label="Execution log" class="no-scroll">
                

                <iview-row>
                    <iview-col span="12">
                        <iview-card :bordered="false">
                            <p slot="title">Workflow</p>
                            <iview-scroll shadow class="console-output2">
                                <iview-steps :current="currentIndex" direction="vertical" >
                                    <iview-step :class='step.shortlabel' :title="step.type.toUpperCase()" :content="step.content" v-for="step in model.events" :key="step.id"></iview-step>
                                </iview-steps>
                            </iview-scroll>
                        </iview-card>
                    </iview-col>
                    <iview-col span="12" offset="0">
                        <iview-card shadow class="console-output">
                            <p slot="title">Console output</p>
                        <iview-scroll shadow class="console-output2">
                            <div class="container">
                                <pre :class="[step.type, 'console-output']" v-for="step in model.output" :key="step.id">{{step.content}}</pre>
                            </div>
                        </iview-scroll>
                        </iview-card>
                    </iview-col>
                </iview-row>

            </iview-tab-pane>
        </iview-tabs>
    </div>
</template>

<script>

var ApplicationSVARExecution = {
    template: iview.getTagNamesFromElement(document.currentScript.previousElementSibling),
    props: {
        'type':{
            type: String,
            default: 'single',
        }
    },
    data(){
        return {
            model: svarstate.commandUpdates[this._props.type.toLowerCase()],
            isCollapsed: false
        }
    },
    methods: {
        clearSteps(){
            this.steps = []
        },
        addStep(shortlabel, type, title, content){
            let id = this.steps.length == 0? 1: (this.steps[this.steps.length - 1].id + 1)
            let newStep = {
                shortlabel,
                id,
                title,
                type,
                content,
            }
            this.steps.push(newStep)
        },
    },
    created(){
        this.$on('add-step', this.addStep);
        this.$on('clear-steps', this.clearSteps);
    },
    computed: {
        currentIndex(){
            return this.model.events.length -1
        },
    },
    watch: {
        datasetPath(ne){
            console.log(this)
            console.log(ne)
        }
    }
}
Vue.component('app-svar-execution', ApplicationSVARExecution)


</script>
