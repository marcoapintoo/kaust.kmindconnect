<template id="app-iview-layout">
    <div class="app-layout">
        <iview-layout>
            <!--
            <iview-header class="dark-background">
                <iview-menu mode="horizontal" theme="light" active-name="1">
                    <div class="app-layout-logo"></div>
                    <div class="app-layout-nav">
                        <iview-Menu-item name="1">
                        <Icon type="ios-navigate"></Icon>
                        Item 1
                        </iview-Menu-item>
                        <iview-Menu-item name="2">
                        <Icon type="ios-keypad"></Icon>
                        Item 2
                        </iview-Menu-item>
                        <iview-Menu-item name="3">
                        <Icon type="ios-analytics"></Icon>
                        Item 3
                        </iview-Menu-item>
                        <iview-Menu-item name="4">
                        <Icon type="ios-paper"></Icon>
                        Item 4
                        </iview-Menu-item>
                    </div>
                </iview-Menu>
            </iview-header>
            -->
            <iview-layout>
                <iview-sider
                :width="250"
                :class="leftSideBarClass"
                ref="leftSideBar" hide-trigger collapsible :collapsed-width="78" v-model="isCollapsed"
                show-trigger :style="{}">
                    <iview-Menu active-name="1-2" theme="light" width="auto" :open-names="['1']">

                        <slot name="left-side-bar"></slot>

                    </iview-Menu>
                </iview-sider>
                <div class="app-main-container">
                    <slot></slot>
                </div>
            </iview-layout>
        </iview-layout>
    </div>
</template>

<script>

var AppLayout = {
    template: iview.getTagNamesFromElement(document.currentScript.previousElementSibling),
    methods: {
        collapsedSider () {
            this.$refs.leftSideBar.toggleCollapse();
        }
    },
    data() {
        return {
            isCollapsed: false,
        }
    },
    computed: {
        subMenuTitleButtonClass(){
            return [
                //"ivu-menu-submenu-title-icon", 
                "ivu-icon",
                this.isCollapsed ? 'ivu-icon-plus' : 'ivu-icon-minus'
            ]
        },
        leftSideBarClass(){
            return [
                "app-sidebar",
                this.isCollapsed ? 'collapsed-menu' : ''
            ]
        },
    },

    events: {
        'get-submenu-classes': () => {
            console.log(1)
            return this.leftSideBarClass
        },
        'get-sidebar-classes': () => {
            console.log(2)
            return this.leftSideBarClass
        },
        'collapse-sidebar': () => {
            console.log(3)
            return this.collapsedSider
        },
    }
}
Vue.component('app-layout', AppLayout)

</script>
