<style scoped>
    .app-layout {
        border: 1px solid #d7dde4;
        background: #f5f7f9;
        position: relative;
        border-radius: 4px;
        overflow: hidden;

        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;

    }

    .app-layout-logo {
        width: 100px;
        height: 30px;
        background: #5b6270;
        border-radius: 3px;
        float: left;
        position: relative;
        top: 15px;
        left: 20px;
    }

    .app-layout-nav {
        width: 420px;
        margin: 0 auto;
        margin-right: 20px;
    }
</style>

<template id="app-iview-layout">
    <div class="app-layout">
        <iview-layout>
            <iview-header>
                <iview-menu mode="horizontal" theme="dark" active-name="1">
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
            <iview-layout>
                <iview-sider hide-trigger :style="{background: '#fff'}">
                    <iview-Menu active-name="1-2" theme="light" width="auto" :open-names="['1']">
                        <iview-submenu name="1">
                            <template slot="title">
                                <Icon type="ios-navigate"></Icon>
                                Item 1
                            </template>
                            <iview-Menu-item name="1-1">Option 1</iview-Menu-item>
                            <iview-Menu-item name="1-2">Option 2</iview-Menu-item>
                            <iview-Menu-item name="1-3">Option 3</iview-Menu-item>
                        </iview-submenu>
                        <iview-submenu name="2">
                            <template slot="title">
                                <Icon type="ios-keypad"></Icon>
                                Item 2
                            </template>
                            <iview-Menu-item name="2-1">Option 1</iview-Menu-item>
                            <iview-Menu-item name="2-2">Option 2</iview-Menu-item>
                        </iview-submenu>
                        <iview-submenu name="3">
                            <template slot="title">
                                <Icon type="ios-analytics"></Icon>
                                Item 3
                            </template>
                            <iview-Menu-item name="3-1">Option 1</iview-Menu-item>
                            <iview-Menu-item name="3-2">Option 2</iview-Menu-item>
                        </iview-submenu>
                    </iview-Menu>
                </iview-sider>
                <iview-layout :style="{padding: '0 24px 24px'}">
                    <iview-breadcrumb :style="{margin: '24px 0'}">
                        <iview-breadcrumb-item>Home</iview-breadcrumb-item>
                        <iview-breadcrumb-item>Components</iview-breadcrumb-item>
                        <iview-breadcrumb-item>iview-layout</iview-breadcrumb-item>
                    </iview-breadcrumb>
                    <iview-content :style="{padding: '24px', minHeight: '280px', background: '#fff'}">
                        iview-content
                    </iview-content>
                </iview-layout>
            </iview-layout>
        </iview-layout>
    </div>
</template>

<script>

/*
Vue.customElement('app-layout', {
    template: iview.getTagNamesFromElement(document.currentScript.previousElementSibling),
    methods: {

    },
    data() {
        return {
            message: 'Oh hai from the component'
        }
    },
})
*/
var AppLayout = {
    template: iview.getTagNamesFromElement(document.currentScript.previousElementSibling),
    methods: {

    },
    data() {
        return {
            message: 'Oh hai from the component'
        }
    },
}
Vue.component('app-layout', AppLayout)

</script>
