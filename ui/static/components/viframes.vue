
<template>
    <div ref="container">
        <slot></slot>
    </div>
</template>

<script>
var VueFrameElementCollection = {}
var VueFrameElement = {
    template: iview.getTagNamesFromElement(document.currentScript.previousElementSibling),
    props: {
        'src': String,
    },
    data(){
        return {
            //readonly: true
        }
    },
    mounted(){
        let referenced = VueFrameElementCollection[this.$props.src];
        if(!(this.$props.src in VueFrameElementCollection)){
            console.log("CREATING", this.$props.src)
            var ifrm = document.createElement("iframe");
            ifrm.setAttribute("src", this.$props.src);
            //ifrm.style.width = "100%";
            //ifrm.style.height = "100%";
            VueFrameElementCollection[this.$props.src] = ifrm;
            referenced = ifrm
            if (ifrm.parentNode) {
                ifrm.parentNode.removeChild(ifrm);
            }
        }
        referenced.display = "block"
        //this.$refs.container.appendChild(referenced);
        try {
            ////WARNING: This raises warning errors in Vue
            this.$refs.container.parentNode.appendChild(referenced);
        } catch (error) {
            
        }
    },
    destroyed(){
        function detach(node) {
            var parent = node.parentNode;
            var next = node.nextSibling;
            if (!parent) { return; }
            parent.removeChild(node);
            parent.insertBefore(node, next);
        }

        console.log("HDP!")
        let referenced = VueFrameElementCollection[this.$props.src];
        referenced.display = "none"
        detach(referenced)
    },
    methods: {
    },
    computed: {
        
    }
}

Vue.component('vue-iframes', VueFrameElement)

</script>
