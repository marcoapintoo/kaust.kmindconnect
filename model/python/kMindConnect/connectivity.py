import os
import json
import numpy as np
from .loaders import AutofileReader
from .plotting import Plotting
from .individual import IndividualProcess, IndividualProcessLogger
import oct2py
#from .matlab_engine import TVVAR, Clustering, SVAR
from .python_engine import TVVAR, Clustering, SVAR

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

    def after_load_data(self):
        path = os.path.join(self.output_folder, "dataset_signals.html")
        print(":: Plotting", path)
        Plotting.matrix_series(
            self.associated_process.data, path, columns=1)

    def after_get_time_coefficients(self):
        path = os.path.join(self.output_folder, "time_coefficients.html")
        print(":: Plotting", path)
        Plotting.matrix_series(
            self.associated_process.time_coefficients, path,
            columns=len(self.associated_process.data), skip=50, staticPlot=True)

    def after_get_cluster_coefficients(self):
        path = os.path.join(self.output_folder, "centroids.html")
        print(":: Plotting", path)
        Plotting.clusters(
            self.associated_process.time_coefficients.T,
            self.associated_process.state_sequence.ravel(),
            self.associated_process.centroids,
            filename=path,
            title="PCA projection of the TVVAR coefficients")

    def after_get_state_space_coefficients(self):
        path = os.path.join(self.output_folder, "kalman_estimated_coherence.html")
        print(":: Plotting", path)
        Plotting.heatmap(
            self.associated_process.coherence_estimated,
            path,
            labels=self.labels)

        path = os.path.join(self.output_folder, "kalman_states_filtered.html")
        print(":: Plotting", path)
        Plotting.matrix_series(
            self.associated_process.state_sequence_filtered,
            path,
            columns=1, transpose=True, height=400)

        path = os.path.join(self.output_folder, "kalman_states_series_smoothed.html")
        print(":: Plotting", path)
        Plotting.matrix_series(
            self.associated_process.state_sequence_smoothed,
            path,
            columns=1, transpose=True, height=400)

        path = os.path.join(self.output_folder, "kalman_states_smoothed.html")
        print(":: Plotting", path)
        Plotting.multiary_series(
            self.associated_process.state_sequence_smoothed,
            path,
            columns=1, transpose=True, height=400)

        path = os.path.join(self.output_folder, "kalman_states_smoothed")
        print(":: Plotting", path)
        self.save_json(
            self.associated_process.state_sequence_smoothed,
            path, transpose=True)

        if self.brain_surface_reference is None:
            return
        dataseries = self.associated_process.coherence_estimated
        for n in range(dataseries.shape[-1]):
            path = os.path.join(self.output_folder, "coherence_state_{0}".format(n + 1))
            print(":: Plotting", path)
            Plotting.coherence_matrix_surface(
                dataseries[:, :, n], path, self.brain_surface_reference,
                convert_to_png=False)
        

    def do_default_output(self, path):
        path = os.path.join(self.output_folder,path)
        print(path)
        if os.path.exists(path):
            print("    removed")
            #os.remove(path)
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

class SVARConnectivity:
    def __init__(self, var_order, window_length, window_shift, number_states, em_tolerance, em_max_iterations, labels, output_folder, brain_surface_reference, matlab_engine_path):
        #self.engine = matlab.engine.start_matlab()
        #self.engine = oct2py.Oct2Py()
        self.engine = Oct2PyPatch()
        self.engine.addpath(matlab_engine_path)
        #self.engine.pkg("pkg", "-forge", "install", "control", "signal", "statistics", "io")
        self.engine.pkg("load", "control", "signal", "statistics", "io")
        self.time_coeff_engine = TVVAR(self.engine, var_order, window_length, window_shift)
        self.clustering_engine = Clustering(self.engine, var_order, number_states)
        self.time_space_coeff_engine = SVAR(self.engine, var_order, number_states, em_tolerance, em_max_iterations)
        self.output_folder = output_folder
        self.brain_surface_reference = brain_surface_reference
        self.labels = labels
        self.process = None

    def run(self, filename, fieldname=None):
        reader_engine = AutofileReader(target=fieldname)
        self.process = IndividualProcess(
            reader_engine,
            self.time_coeff_engine,
            self.clustering_engine,
            self.time_space_coeff_engine,
        )
        self.process.logger = SVARConnectivityPlotter(
            self.process, self.labels, self.output_folder, self.brain_surface_reference)
        self.process.run(filename)

    @staticmethod
    def read_txt(path):
        with open(path, "r") as f:
            lines = [line.strip() for line in f.readlines()]
            return [line for line in lines if not line.startswith("#") and line != ""]
