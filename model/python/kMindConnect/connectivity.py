import os
import sys
import shutil
import hashlib
import base64
import json
import numpy as np
from .loaders import AutofileReader
from .individual import IndividualProcess, IndividualProcessLogger
import kMindConnect.plotting as plotutils
try:
    import oct2py
    from .matlab_engine import TVVAR as MatlabTVVAR, Clustering as MatlabClustering, SVAR as MatlabSVAR
except:
    pass
from .python_engine import TVVAR as PythonTVVAR, Clustering as PythonClustering, SVAR as PythonSVAR

class SVARConnectivityPlotter(IndividualProcessLogger):
    def __init__(self, associated_process, labels, output_folder, brain_surface_reference=None):
        IndividualProcessLogger.__init__(self, associated_process)
        self.brain_surface_reference = brain_surface_reference
        self.labels = labels
        self.output_folder = output_folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        self.loading_html = """
        <!doctype html><html><head><meta http-equiv="refresh" content="10"><style type="text/css">body{font-size: 500%;background-color: #f5f5f4;}
        body div.rotate{position: absolute;top: 50%;left: 50%;width: 20px;height: 20px;
        animation:spin 1s ease-in-out infinite;}@keyframes spin { 100% { transform:rotate(360deg);} }
        </style></head><body><div class="rotate">.</div></body></html>  
        """
        self.do_default_output("dataset_signals.html")
        self.do_default_output("time_coefficients.html")
        self.do_default_output("centroids.html")
        self.do_default_output("kalman_estimated_coherence.html")
        self.do_default_output("kalman_states_filtered.html")
        self.do_default_output("kalman_states_smoothed.html")

    def before_load_data(self):
        print(":: STATUS: Before loading data")

    def after_load_data(self):
        print(':: ACTIVITY: Plotting intermediate results')
        path = os.path.join(self.output_folder, "dataset_signals.html")
        plotutils.matrix_series(
            self.associated_process.data, path, columns=1)
        print(":: STATUS: Finishing loading data")

    def before_get_time_coefficients(self):
        print(":: STATUS: Before calculating TV-VAR coefficients")

    def after_get_time_coefficients(self):
        print(':: ACTIVITY: Plotting intermediate results')
        path = os.path.join(self.output_folder, "time_coefficients.html")

        plotutils.matrix_series(
            self.associated_process.time_coefficients, path,
            columns=len(self.associated_process.data), skip=50, staticPlot=True)
        print(":: STATUS: TV-VAR coefficients calculated")

    def before_get_cluster_coefficients(self):
        print(":: STATUS: Starting K-means")

    def after_get_cluster_coefficients(self):
        print(':: ACTIVITY: Plotting intermediate results')
        path = os.path.join(self.output_folder, "centroids.html")
        max_plotted = 1000
        N = len(self.associated_process.state_sequence)
        if N > max_plotted:
            idx = np.random.randint(0, N, max_plotted)
        else:
            idx = np.arange(N)
        plotutils.clusters(
            self.associated_process.time_coefficients.T[idx, :],
            self.associated_process.state_sequence.ravel()[idx],
            self.associated_process.centroids,
            filename=path,
            title="PCA projection of the TVVAR coefficients")
        print(":: STATUS: Get cluster centroids")

    def before_get_state_space_coefficients(self):
        print(":: STATUS: Starting space-state analysis")

    def after_get_state_space_coefficients(self):
        print(':: ACTIVITY: Plotting intermediate results')
        path = os.path.join(self.output_folder, "kalman_estimated_coherence.html")
        plotutils.heatmap(
            self.associated_process.coherence_estimated,
            path,
            labels=self.labels)

        print(':: ACTIVITY: Plotting intermediate results')
        path = os.path.join(self.output_folder, "kalman_states_filtered.html")
        plotutils.matrix_series(
            self.associated_process.state_sequence_filtered,
            path,
            columns=1, transpose=True, height=400)

        print(':: ACTIVITY: Plotting intermediate results')
        path = os.path.join(self.output_folder, "kalman_states_series_smoothed.html")
        plotutils.matrix_series(
            self.associated_process.state_sequence_smoothed,
            path,
            columns=1, transpose=True, height=400)

        print(':: ACTIVITY: Plotting intermediate results')
        path = os.path.join(self.output_folder, "kalman_states_smoothed.html")
        plotutils.multiary_series(
            self.associated_process.state_sequence_smoothed,
            path,
            columns=1, transpose=True, height=400)

        print(':: ACTIVITY: Plotting intermediate results')
        path = os.path.join(self.output_folder, "kalman_states_smoothed")
        self.save_json(
            self.associated_process.state_sequence_smoothed,
            path, transpose=True)

        print(":: STATUS: Space-state analysis finished")
        if self.brain_surface_reference is None or self.brain_surface_reference == "":
            return

        print(':: ACTIVITY: Plotting brain surface image')
        dataseries = self.associated_process.coherence_estimated
        for n in range(dataseries.shape[-1]):
            path = os.path.join(self.output_folder, "coherence_state_{0}".format(n + 1))
            plotutils.coherence_matrix_surface(
                dataseries[:, :, n], path, self.brain_surface_reference,
                convert_to_png=False)
        

    def do_default_output(self, path):
        path = os.path.join(self.output_folder,path)
        #print(path)
        if os.path.exists(path):
            #print("    removed")
            os.remove(path)
        """
        with open(path, "w") as f:
            f.write(self.loading_html)
        """

    def save_json(self, data, filename, transpose=False):
        x = np.array(data)
        if transpose:
            x = x.T
        print("{0}: {1}".format(filename, x.shape))
        with open(filename + ".json", "w") as f:
            json.dump(x.tolist(), f, indent=4)


class Oct2PyPatch(oct2py.Oct2Py):
    def __init__(self, *args, **kwargs):
        super(Oct2PyPatch, self).__init__(*args, **kwargs)
    
    def feval(self, name, *args, **kwargs):
        args = [a + 0.0 if np.any(np.isreal(a)) else a for a in args]
        return super(Oct2PyPatch, self).feval(name, *args, **kwargs)

class ExperimentSettings:
    def __init__(self, settings):
        self.settings = settings
        self.expand_arguments()
        self.var_order = self.parse("--var-order", int)
        self.window_length = self.parse("--window-length", int)
        self.window_shift = self.parse("--window-shift", int)
        self.number_states = self.parse("--number-states", int)
        self.em_tolerance = self.parse("--em-tolerance", float)
        self.em_max_iterations = self.parse("--em-max-iterations", int)
        self.labels = self.parse_text("--labels")
        self.output_folder = settings["--output-dir"]
        self.clean_output_folder = settings["--clean-output-dir"]
        self.dataset_field = settings["--matlab-field"]
        self.brain_surface_reference = settings["--brain-surface"]
        self.matlab_engine_path = os.path.realpath(settings["--matlab-model-path"])
        self.datasets = settings["<datasets>"]
        self.onlyHash = settings["--only-get-hash"]
        self.useMatlabEngine = ((settings["--use-octave-engine"] or "T").upper()[0] == "T")
        print(self.settings)
        print(self.useMatlabEngine)

    def expand_arguments(self):
        def _fix_quotations(x):
            v = x[:]
            if v.startswith("'") or v.startswith('"'):
                x = x[1:]
            if v.endswith("'") or v.endswith('"'):
                x = x[:-1]
            return x
        for k, v in self.settings.items():
            if not isinstance(v, str): continue
            self.settings[k] = _fix_quotations(v)
        pathArgs = [
            "--output-dir",
            "--matlab-model-path",
            "--labels",
            "--brain-surface",
        ]
        for key in pathArgs:
            self.settings[key] = os.path.realpath(self.settings[key]) if self.settings[key].strip() != "" else ""
        for i, path in enumerate(self.settings["<datasets>"]):
            print(path)
            self.settings["<datasets>"][i] = os.path.realpath(_fix_quotations(path))

    @staticmethod
    def output_folder_updated(arguments, filename):
        return arguments["--output-dir"].replace("[[dataset]]", os.path.splitext(os.path.basename(filename))[0])

    @staticmethod
    def get_hash(arguments, filename):
        notActualArgs = ["--only-get-hash", "<datasets>"]
        arguments = arguments.copy()
        for key in notActualArgs:
            arguments.pop(key)
        arguments["<datasets>"] = [filename]
        arguments["--output-dir"] = ExperimentSettings.output_folder_updated(arguments, filename)
        arguments = [(k, v) for (k, v) in arguments.items()]
        arguments = sorted(arguments, key=lambda x: x[0])
        hasher = hashlib.sha1(json.dumps(arguments).encode())
        return base64.urlsafe_b64encode(hasher.digest()[0:10]).decode()

    def parse_text(self, key):
        if self.settings[key].strip() == "": return
        try:
            return SVARConnectivity.read_txt(self.settings[key])
        except Exception as e:
            print(":: ERROR: {0} must be a valid label file".format(key))
            sys.exit(1)

    def parse(self, key, type):
        try:
            return type(self.settings[key])
        except:
            print(":: ERROR: {0} must be of type: {1}".format(key, type))
            sys.exit(1)

    def run(self):
        if self.onlyHash:
            print(self.get_hash(self.settings, self.datasets[0]))
            return

        connectivity = SVARConnectivity(
            self.var_order, self.window_length, self.window_shift,
            self.number_states, self.em_tolerance, self.em_max_iterations, self.labels,
            self.output_folder, self.brain_surface_reference, self.matlab_engine_path, self.useMatlabEngine)

        for dataset in self.datasets:
            print(":: START: Loading file:", dataset)
            updated_output_folder = self.output_folder_updated(self.settings, dataset)
            connectivity.output_folder = updated_output_folder
            connectivity.hash_code = self.get_hash(self.settings, dataset)
            if self.clean_output_folder and os.path.exists(updated_output_folder):
                shutil.rmtree(updated_output_folder)
            if not os.path.exists(connectivity.experiment_path):
                os.makedirs(connectivity.experiment_path)
            if connectivity.experiment_cached:
                print(":: STATUS: Experiment cached! Nothing to do here.")
            else:
                connectivity.run(dataset, self.dataset_field)
                connectivity.save_experiment_cache(self.settings)
            print(":: FINISH: Simulation model finished!")
            print(":: EXPERIMENT-ID: " + connectivity.hash_code)



class SVARConnectivity:
    def __init__(self, var_order, window_length, window_shift, number_states, em_tolerance, em_max_iterations, labels, output_folder, brain_surface_reference, matlab_engine_path, use_matlab_engine):
        self.engine = Oct2PyPatch()
        #self.engine = matlab.engine.start_matlab()
        print(matlab_engine_path)
        print(os.path.exists(matlab_engine_path))
        self.engine.addpath(matlab_engine_path)
        #self.engine.pkg("pkg", "-forge", "install", "control", "signal", "statistics", "io")
        self.engine.pkg("load", "control", "signal", "statistics", "io")
        if use_matlab_engine:
            TVVAR = MatlabTVVAR
            Clustering = MatlabClustering
            SVAR = MatlabSVAR
        else:
            #self.engine = object()
            TVVAR = PythonTVVAR
            Clustering = PythonClustering
            SVAR = PythonSVAR

        self.use_matlab_engine = use_matlab_engine
        self.time_coeff_engine = TVVAR(self.engine, var_order, window_length, window_shift)
        self.clustering_engine = Clustering(self.engine, var_order, number_states)
        self.time_space_coeff_engine = SVAR(self.engine, var_order, number_states, em_tolerance, em_max_iterations)
        self.output_folder = output_folder
        self.brain_surface_reference = brain_surface_reference
        self.labels = labels
        self.process = None
        self.hash_code = None
    
    @property
    def experiment_path(self):
        path = "{0}/{1}/".format(self.output_folder, self.hash_code)
        return path
    
    @property
    def experiment_conclusion_path(self):
        filename = "experiment.json"
        path = "{0}/{1}/{2}".format(self.output_folder, self.hash_code, filename)
        return path

    @property
    def experiment_cached(self):
        return os.path.exists(self.experiment_conclusion_path)

    def save_experiment_cache(self, configuration):
        with open(self.experiment_conclusion_path, "w") as f:
            f.write(json.dumps(configuration, sort_keys=True, indent=4))

    def run(self, filename, fieldname=None):
        reader_engine = AutofileReader(target=fieldname)
        self.process = IndividualProcess(
            reader_engine,
            self.time_coeff_engine,
            self.clustering_engine,
            self.time_space_coeff_engine,
        )
        self.process.logger = SVARConnectivityPlotter(
            self.process, self.labels, self.experiment_path, self.brain_surface_reference)
        self.process.run(filename)

    @staticmethod
    def read_txt(path):
        with open(path, "r") as f:
            lines = [line.strip() for line in f.readlines()]
            return [line for line in lines if not line.startswith("#") and line != ""]
