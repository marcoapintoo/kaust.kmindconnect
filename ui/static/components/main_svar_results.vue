<template>
    <div class="" style="width:100%; height:90%;">
            <div class="app-toolbar">
                <iview-button-group shape="circle" size="small">
                    <iview-button @click="activeResult='clusters'">
                        <iview-icon type="ios-color-filter"></iview-icon>
                        Clusters (PCA)
                    </iview-button>
                    <iview-button @click="activeResult='states-signal'">
                        <iview-icon type="podium"></iview-icon>
                        Connectivity states
                    </iview-button>
                    <iview-button @click="activeResult='states-matrices'">
                        <iview-icon type="ios-grid-view"></iview-icon>
                        PDC matrices
                    </iview-button>
                    <iview-button @click="activeResult='brain-graphics'" v-if="showbrains">
                        <iview-icon type="aperture"></iview-icon>
                        Brain coherence
                    </iview-button>
                    <iview-button @click="activeResult='brain-animation'" v-if="showbrains">
                        <iview-icon type="videocamera"></iview-icon>
                        Connectivity evolution
                    </iview-button>
                </iview-button-group>
            </div>
            <!--
            <iview-table class="single-settings-header" style="border:none;" stripe :columns="headerColumns" :data="headerData">
            </iview-table>
            -->
            <iview-tabs type="card" :value="activeResult" class="graphical-results" :animated="false">
                <iview-tab-pane name="clusters">
                    <scatter-chart
                        :data="clustersData"
                        xtitle="PCA-1 Component"
                        ytitle="PCA-2 Component"
                        width="100%" height="90%"
                    />
                </iview-tab-pane>
                <!--
                <iview-tab-pane name="clusters">
                    <vue-iframes
                    :src="fixedOutputPath + '/' + hashcode + '/centroids.html'"
                    ></vue-iframes>
                </iview-tab-pane>
                -->
                <iview-tab-pane name="states-signal">
                    <column-chart
                        :data="statesSignalData"
                        xtitle="Time"
                        ytitle="States"
                        width="100%" height="90%"
                    />
                    <!--
                    <vue-iframes
                    :src="fixedOutputPath + '/' + hashcode + '/kalman_states_smoothed.html'"
                    ></vue-iframes>
                    -->
                </iview-tab-pane>
                <iview-tab-pane name="states-matrices">
                    <div class="content" v-for="n in rangeStates" :key="n.id">
                        <div class="state">
                            <div class="title toolbar app-toolbar">State {{n.id}}</div>
                            
                                <vue-iframes :src="fixedOutputPath + '/' + hashcode + '/kalman_estimated_coherence-state-' + n.id + '.html'"></vue-iframes>
                            
                        </div>
                    </div>
                </iview-tab-pane>
                <iview-tab-pane name="brain-graphics">
                    <div class="content" v-for="n in rangeStates" :key="n.id">
                        <div class="state">
                            <div class="title toolbar app-toolbar">State {{n.id}}</div>
                            <img :src="fixedOutputPath + '/' + hashcode + '/coherence_state_' + n.id + '.svg'" />
                        </div>
                    </div>
                </iview-tab-pane>
                <iview-tab-pane name="brain-animation">
                    
                    <vue-iframes
                    :src="preffix + 'static/components/main_svar_animation.html?path=' + fixedOutputPath + '/' + hashCodeURI + '/'"
                    ></vue-iframes>
                    
                </iview-tab-pane>
            </iview-tabs>

    </div>
</template>

<script>

var ApplicationSVARResults = {
    template: iview.getTagNamesFromElement(document.currentScript.previousElementSibling),
    props: ['outputpath', 'hashcode', 'states', 'showbrains'],
    data(){
        return {
            //
            activeResult: 'states-signal',
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
            lastHashCode: '',

            clustersData: [],
            statesSignalData: [],

        }
    },
    mounted(){
        console.log(this)
        if(this.hashcode != null && this.hashcode != ""){
            this.reloadResults(this.hashcode)
        }
    },
    methods: {
        reloadResults(hashcode){
            console.log("!!!!!!!!!!!!!!!")
            const self = this;
            var updateChart = (refname, url) => {
                var request = new XMLHttpRequest();
                request.open('GET', url, true);
                request.onload = function() {
                    if (request.status >= 200 && request.status < 400) {
                        var respJson = JSON.parse(request.responseText);
                        //self.$refs[refname].data = respJson
                        //self.$refs[refname].$props["data"] = respJson
                        self[refname] = respJson
                    } else {
                        console.error("Request status: " + request.status)
                    }
                };
                request.onerror = function(e) {console.error("Error: " + e)};
                request.send();
            }
            updateChart(
                "clustersData",
                this.fixedOutputPath + '/' + hashcode + '/centroids.json'
            )
            updateChart(
                "statesSignalData",
                this.fixedOutputPath + '/' + hashcode + '/brain_state_sequence.json'
            )
        }
    },
    created(){
        /////this.$on('execution-finished', this.modelFinished);
    },
    computed: {
        hashCodeURI(){
            return this.hashcode?this.hashcode.replace(/=/g, "%3d", /&/g, "%26"): ""
        },
        preffix(){
            return appArguments.dirname
            /*
            let preffix = ""
            if(this.outputpath.startsWith("./")){
                preffix = "../"
            }
            return preffix
            */
        },
        fixedOutputPath(){
            if(this.outputpath.startsWith("/")) return this.outputpath
            return appArguments.dirname + this.outputpath
        },
        rangeStates(){
            x = []
            for (let i = 0; i < this.states; i++) {
                x.push({id: i + 1})
            }
            return x
        },
    },
    watch: {
        hashcode(newval, oldVal){
            console.log("CHANGES!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", this.hashcode, newval, oldVal)
            this.reloadResults(newval)
            //this.reloadResults(hashcode)
        },
        outputpath(newval, oldVal){
            console.log("CHANGES2!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", this.hashcode, newval)
            //this.reloadResults(this.hashcode)
        },
        states(newval, oldVal){
            console.log("CHANGES3!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", this.hashcode, newval)
            //this.reloadResults(this.hashcode)
        }
    }
}
Vue.component('app-svar-results', ApplicationSVARResults)


</script>
