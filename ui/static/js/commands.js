var fs = require('fs');
var deleteFolderRecursive = function (path) {
    if (fs.existsSync(path)) {
        fs.readdirSync(path).forEach(function (file, index) {
            var curPath = path + "/" + file;
            if (fs.lstatSync(curPath).isDirectory()) { // recurse
                deleteFolderRecursive(curPath);
            } else { // delete file
                fs.unlinkSync(curPath);
            }
        });
        fs.rmdirSync(path);
    }
};

class IndividualExperiment {
    constructor() {
        this.kmindconnect_path = "/Users/pinto.marco/KAUSTProjects/TonitruumUI0/kEEGConnect/python"
        //this.kmindconnect_path = "/Applications/keegconnect.app/python"
        this.parameters = {
            "var_order": ["individual-configuration-var-order", 1, true],
            "number_states": ["individual-configuration-number-states", 3, true],
            "window_length": ["individual-configuration-window-length", 50, true],
            "window_shift": ["individual-configuration-window-shift", 1, true],
            "em_tolerance": ["individual-configuration-em-tolerance", 0.0001, true],
            "em_iterations": ["individual-configuration-em-iterations", 15, true],
            "dataset_path": ["individual-configuration-dataset-path", "", true],
            "mat_field": ["individual-configuration-dataset-mat-field", "mean_roi", false],
            "dataset_names": ["individual-configuration-dataset-names", "", true],
        };
        this.timeoutPath = 500;
    }
    defaultValues() {
        for (let i in this.parameters) {
            const elementID = this.parameters[i][0];
            const defaultValue = this.parameters[i][1];
            document.getElementById(elementID).value = defaultValue;
        }
    }
    resetValues() {
        for (let i in this.parameters) {
            const elementID = this.parameters[i][0];
            document.getElementById(elementID).value = "";
        }
    }
    getValue(code) {
        const elementID = this.parameters[code][0];
        return document.getElementById(elementID).value;
    }
    verifyValues() {
        var message = ""
        for (let i in this.parameters) {
            const elementID = this.parameters[i][0];
            const isMandatory = this.parameters[i][2];
            if (!isMandatory) continue;
            const value = document.getElementById(elementID).value;
            if (value == null || value == "") {
                message += i + ", ";
            }
        }
        if (message != "") {
            message = "The following parameters: " + message;
            message += "cannot be empty\n";
            alert(message);
            return false;
        }
        return true;
    }
    verifyPath(path, action) {
        var lookup = () => {
            var request = new XMLHttpRequest();
            request.open('GET', path, true);
            request.onload = function () {
                if (request.status >= 200 && request.status < 400) {
                    action(path)
                } else {
                    setTimeout(lookup, this.timeoutPath);
                }
            };
            request.onerror = () => { setTimeout(lookup, this.timeoutPath); }
            request.send();
        }
        setTimeout(lookup, this.timeoutPath);
    }
    executeExperiment() {
        var script_name = this.kmindconnect_path + "/kmindconnect.py"
        var brain_surface = this.kmindconnect_path + "/brain_structure_16.svg"
        var output_dir = this.kmindconnect_path + "/graphical-outputs/results-[[dataset]]"
        var dataset = this.getValue("dataset_path");

        //kmindconnect.py --matlab-field="mean_roi" --em-max-iterations=1  --labels="/Users/pinto.marco/KAUSTProjects/TonitruumUI0/kEEGConnect/matlab/data/ROI.txt" --brain-surface="./brain_structure_16.svg" --output-dir="./outputs/results-[[dataset]]" "/Users/pinto.marco/KAUSTProjects/TonitruumUI0/kEEGConnect/matlab/data/FS_16ROI_mean/6791_mean_fs.mat"        
        //var executablePath = "python";
        var executablePath = "/Users/pinto.marco/anaconda/bin/python";
        var parameters = [script_name];
        if (this.getValue("mat_field"))
            parameters.push("--matlab-field", this.getValue("mat_field"))
        parameters.push("--brain-surface", brain_surface)
        parameters.push("--output-dir", output_dir)
        parameters.push("--var-order", this.getValue("var_order"))
        parameters.push("--number-states", this.getValue("number_states"))
        parameters.push("--window-length", this.getValue("window_length"))
        parameters.push("--window-shift", this.getValue("window_shift"))
        parameters.push("--em-tolerance", this.getValue("em_tolerance"))
        parameters.push("--em-max-iterations", this.getValue("em_iterations"))
        parameters.push("--labels", this.getValue("dataset_names"))
        parameters.push("--clean-output-dir")
        parameters.push(dataset)

        console.log(parameters)
        var child = require('child_process').execFile;
        child(executablePath, parameters, function (err, data) {
            console.log(err)
            console.log(data.toString());
            document.querySelector(".window").classList.remove("processing")
            Array.from(document.querySelectorAll(".individual-configuration input, .individual-configuration button")).forEach((e) => {
                e.removeAttribute("readonly")
                e.removeAttribute("disabled")
            });
        });
        document.querySelector(".window").classList.add("processing")
        Array.from(document.querySelectorAll(".individual-configuration input, .individual-configuration button")).forEach((e) => {
            e.setAttribute("readonly", "readonly")
            e.setAttribute("disabled", "disabled")
        });
        var basename = dataset.split("/")
        basename = basename[basename.length - 1]
        basename = basename.split(".")
        basename = basename.slice(0, basename.length - 1)
        var basepath = output_dir.replace("[[dataset]]", basename)
        var footer = document.getElementById("app-footer")
        footer.innerText = "Starting experiment...";
        Array.from(document.querySelectorAll(".fallback")).forEach(e => {
            e.setAttribute("style", "display: block");
        })
        deleteFolderRecursive(basepath);
        //setTimeout(() => {
        this.verifyPath(basepath + "/centroids.html", url => {
            footer.innerText = "Centroids calculated...";
            console.log("Updating centroids graphics...")
            document.querySelector(".individual-graphic-result-pane.clusters iframe").setAttribute("src", url);
            document.querySelector(".individual-graphic-result-pane.clusters .fallback").setAttribute("style", "display: none;");
        })
        this.verifyPath(basepath + "/kalman_states_smoothed.html", url => {
            footer.innerText = "Brain states calculated...";
            console.log("Updating states graphics...")
            document.querySelector(".individual-graphic-result-pane.states iframe").setAttribute("src", url);
            document.querySelector(".individual-graphic-result-pane.states .fallback").setAttribute("style", "display: none;");
        })
        var n_states = Number.parseInt(this.getValue("number_states"))
        var matrixTemplate = "<div class='state'><div class='title toolbar'>State [[state]]</div><img src='[[basepath]]/coherence_state_[[state]].svg'/></div>"
        var matrixTemplateSource = basepath + '/coherence_state_[[state]].svg'.replace(/\[\[state\]\]/g, n_states);
        this.verifyPath(matrixTemplateSource, url => {
            footer.innerText = "Coherence calculated...";
            console.log("Updating coherence graphics...")
            //document.querySelector(".individual-graphic-result-pane.coherence iframe").setAttribute("src", url);
            document.querySelector(".individual-graphic-result-pane.coherence-surface .fallback").setAttribute("style", "display: none;");
            var matrixContent = ""
            for (let state = 0; state < n_states; state++) {
                matrixContent += matrixTemplate.replace(/\[\[basepath\]\]/g, basepath).replace(/\[\[state\]\]/g, state + 1);
            }
            document.querySelector(".individual-graphic-result-pane.coherence-surface .content").innerHTML = matrixContent;

            //

            //document.querySelector(".individual-graphic-result-pane.coherence iframe").setAttribute("src", url);
            let codename = ""
            let path = `${basepath}`
            let url2 = `././animation.html?path=${path}&codename=${codename}`
            document.querySelector(".individual-graphic-result-pane.animation .fallback").setAttribute("style", "display: none;");
            document.querySelector(".individual-graphic-result-pane.animation iframe").setAttribute("src", url2);
            
        })
        var stateTemplate = "<div class='state'><div class='title toolbar'>State [[state]]</div><iframe src='[[basepath]]/kalman_estimated_coherence-state-[[state]].html'></iframe></div>"
        var stateTemplateSource = basepath + '/kalman_estimated_coherence-state-[[state]].html'.replace(/\[\[state\]\]/g, n_states);
        this.verifyPath(stateTemplateSource, url => {
            footer.innerText = "Coherence surface calculated...";
            console.log("Updating coherence surface graphics...")
            //document.querySelector(".individual-graphic-result-pane.coherence iframe").setAttribute("src", url);
            document.querySelector(".individual-graphic-result-pane.coherence .fallback").setAttribute("style", "display: none;");
            var statesContent = ""
            for (let state = 0; state < n_states; state++) {
                statesContent += stateTemplate.replace(/\[\[basepath\]\]/g, basepath).replace(/\[\[state\]\]/g, state + 1);
            }
            document.querySelector(".individual-graphic-result-pane.coherence .content").innerHTML = statesContent;

        })
        //document.querySelector(".individual-graphic-result-pane.clusters iframe").setAttribute("src", basepath + "/centroids.html")
        //document.querySelector(".individual-graphic-result-pane.states iframe").setAttribute("src", basepath + "/kalman_states_smoothed.html")
        //document.querySelector(".individual-graphic-result-pane.coherence iframe").setAttribute("src", basepath + "/kalman_estimated_coherence.html")
        //}, 10 * 1000);
    }
    prepareButtons() {
        document.getElementById("individual-configuration-run").addEventListener("click", e => {
            if (this.verifyValues())
                this.executeExperiment();
        })
        document.getElementById("individual-configuration-default").addEventListener("click", e => {
            this.defaultValues();
        })
        document.getElementById("individual-configuration-reset").addEventListener("click", e => {
            this.resetValues();
        })
    }
    loadIndividualParameters() {
        var parameters_id = {
            "var_order": "individual-configuration-var-order",
            "number_states": "individual-configuration-number-states",
            "window-length": "individual-configuration-window-length",
            "window-shift": "individual-configuration-window-shift",
            "em-tolerance": "individual-configuration-em-tolerance",
            "em-iterations": "individual-configuration-em-iterations",
            "dataset-path": "individual-configuration-dataset-path",
            "mat-field": "individual-configuration-dataset-mat-field",
            "dataset-names": "individual-configuration-dataset-names",
        };
    }
}


ready(function () {
    var experiment = new IndividualExperiment();
    experiment.prepareButtons();
});

