<template id="app-iview-layout">
    <div>
        <iview-menu name="left-bar-menu" :open-names="['simulation', 'exploratory', 'batch', 'configuration', 'documentation']">
        <iview-submenu name="simulation" class="submenu-title" @click="preventCollapse">
            <template slot="title">
                <i :class="subMenuTitleButtonClass" @click="collapsedSider"></i>
                <span style="width: 100%;display: inline-block;" @click.prevent.stop="preventCollapse">
                    SVAR/TVVAR Model
                </span>
                <span class="special-button button-hide" @click="toogleSubMenu"></span>
                <span class="special-button button-collapse" @click="collapsedSider"></span>
            </template>

            <iview-submenu name="documentation">
                <template slot="title">
                    <Icon type="help-bouy"></Icon>
                        <span> Documentation </span>
                </template>
                <iview-Menu-item name="documentation-1" class="link">
                    <router-link to="/svar/model/documentation">
                        <Icon type="erlenmeyer-flask"></Icon>
                        <span>Model description</span>
                    </router-link>
                </iview-Menu-item>
                <iview-Menu-item name="documentation-2" class="link">
                    <router-link to="/svar/data/documentation">
                        <Icon type="easel"></Icon>
                        <span>Sample dataset</span>
                    </router-link>
                </iview-Menu-item>
            </iview-submenu>
            
            <iview-submenu name="exploratory">
                <template slot="title">
                    <Icon type="eye"></Icon>
                        <span> Exploratory analysis </span>
                </template>
                <iview-Menu-item name="e-1" class="link">
                    <router-link to="/svar/single/settings">
                        <Icon type="clipboard"></Icon>
                        <span>Model settings</span>
                    </router-link>
                </iview-Menu-item>
                <iview-Menu-item name="exploratoryLog" ref="exploratoryLog" class="link">
                    <router-link to="/svar/single/log">
                        <Icon type="clipboard"></Icon>
                        <span>Execution progress</span>
                    </router-link>
                </iview-Menu-item>
                <iview-Menu-item name="e-2" class="link">
                    <router-link to="/svar/single/results">
                        <Icon type="stats-bars"></Icon>
                        <span>Results</span>
                    </router-link>
                </iview-Menu-item>
            </iview-submenu>

            <iview-submenu name="batch">
                <template slot="title">
                    <Icon type="social-buffer"></Icon>
                    <span> Batch processing</span>
                </template>
                <iview-Menu-item name="b-1" class="link">
                    <router-link to="/svar/batch/settings">
                        <Icon type="compose"></Icon>
                        <span>Model settings</span>
                    </router-link>
                </iview-Menu-item>
                <iview-Menu-item name="batchLog" ref="batchLog" class="link">
                    <router-link to="/svar/batch/log">
                        <Icon type="clipboard"></Icon>
                        <span>Execution progress</span>
                    </router-link>
                </iview-Menu-item>
                <iview-Menu-item name="b-2" class="link">
                    <router-link to="/svar/batch/results">
                        <Icon type="map"></Icon>
                        <span>Results explorer</span>
                    </router-link>
                </iview-Menu-item>
            </iview-submenu>
            
        </iview-submenu>

        <iview-submenu name="configuration" class="submenu-title" @click="preventCollapse">
            <template slot="title">
                <i :class="subMenuTitleButtonClass" @click="collapsedSider"></i>
                <span style="width: 100%;display: inline-block;" @click.prevent.stop="preventCollapse">
                    Configuration
                </span>
                <span class="special-button button-hide" @click="toogleSubMenu"></span>
                <span class="special-button button-collapse" @click="collapsedSider"></span>
            </template>

        <iview-Menu-item name="settings" class="link">
            <router-link to="/global/settings">
                <Icon type="settings"></Icon>
                <span> Environment Settings</span>
            </router-link>
        </iview-Menu-item>
        
        <iview-Menu-item name="document" class="link">
            <router-link to="/about">
                <Icon type="document"></Icon>
                <span>About...</span>
            </router-link>
        </iview-Menu-item>
        
        </iview-submenu>
        </iview-menu>



    </div>
</template>

<script>

var LeftSideBar = {
    //name: 'left-side-bar',
    template: iview.getTagNamesFromElement(document.currentScript.previousElementSibling),
    methods: {
        collapsedSider(){
            this.actualParent.collapsedSider()
        },
        toogleSubMenu(e){
            window.x=(e.target.parentElement)
            console.log(e.target.parentElement)
            //e.target.parentElement.parentElement.querySelector(".ivu-menu-submenu-title").click()
        },
        preventCollapse(e){
        },
        updated(e){
            console.log(e)
        }
    },
    data() {
        return {
        }
    },
    computed: {
        actualParent(){
            let parent = this.$parent
            while(parent["subMenuTitleButtonClass"] === undefined)
                parent = parent.$parent;
            return parent
        },
        subMenuTitleButtonClass(){
            return this.actualParent.subMenuTitleButtonClass
        },
        leftSideBarClass(){
            return this.actualParent.leftSideBarClass
        },
    }
}
Vue.component('app-left-sidebar', LeftSideBar)

</script>
