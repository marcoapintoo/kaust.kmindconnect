var electronEnabled = false;
try {
    const electron = require('electron');
    electronEnabled = true;
} catch (error) {
}


function nativeready(fn) {
    if (document.attachEvent ? document.readyState === "complete" : document.readyState !== "loading") {
        fn();
    } else {
        document.addEventListener('DOMContentLoaded', fn);
    }
}

nativeready(() => {
    var event = new CustomEvent('DynamicContentLoaded', {});
})

function ready(fn) {
    if (document.dynamicContentState === "complete") {
        fn();
    } else {
        document.addEventListener("DynamicContentLoaded", fn)
    }
}

Vue.use(iview);
iview.lang('en-US');

iview.getTagNamesFromElement = (element) => {
    var html = element.innerHTML;
  
    for (let k = 3; k >= 1; k--) {
        for (let i = 0; i <= 25; i++) {
            let orig = String.fromCharCode("a".charCodeAt(0) + i)
            let dest = String.fromCharCode("A".charCodeAt(0) + i)
            html = html.replace(new RegExp("(</?\\s*iview)((?:-\\w+){" + k + "})-(?:" + orig + "|" + dest + ")", "g"), "$1$2" + dest)
        }
    }
    for (let i = 0; i <= 25; i++) {
        let orig = String.fromCharCode("a".charCodeAt(0) + i)
        let dest = String.fromCharCode("A".charCodeAt(0) + i)
        html = html.replace(new RegExp("(</?\\s*)iview-(?:" + orig + "|" + dest + ")", "g"), "$1" + dest)
    }

    return html;
}
nativeready(function () {
    //!DONT REMOVE
    Array.from(document.querySelectorAll('a[href^="http"]')).forEach((e) => { e.target = '_blank' })
});

