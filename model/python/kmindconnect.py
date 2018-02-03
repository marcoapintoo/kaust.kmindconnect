"""
kMindConnect 0.2

Usage:
    kmindconnect [options] --labels=<l> --output-dir=<o> <datasets>...
    kmindconnect --help

-p <p>, --var-order=<p>             TVVAR/SVAR order [default: 1]
-w <w>, --window-length=<w>         Window length in TVVAR [default: 50]
-s <s>, --window-shift=<s>          Window shift in TTVAR [default: 1]
-n <n>, --number-states=<n>         Number of states [default: 3]
-t <t>, --em-tolerance=<t>          Minimum improvement required to finish EM [default: 0.0001]
-i <i>, --em-max-iterations=<i>     Maximum number of iterations of EM [default: 15]
-z <z>, --matlab-field=<z>          Matlab field (only necessary for .mat datasets)
-l <l>, --labels=<l>                Path of a CSV-file with labels
-x <x>, --brain-surface=<x>         Path of the SVG brain surface template
-o <o>, --output-dir=<o>            Image output path
-m, --matlab-model-path=<m>         MATLAB model path [default: ../matlab/algorithm]
-c, --clean-output-dir              Clean image output path [default: False]
-h, --help                          Show this help
"""
# conda install -c conda-forge oct2py
# python kmindconnect.py --matlab-field="mean_roi" --em-max-iterations=1  --labels="/Users/pinto.marco/KAUSTProjects/TonitruumUI0/kEEGConnect/matlab/data/ROI.txt" --brain-surface="./brain_structure_16.svg" --output-dir="./outputs/results-[[dataset]]" "/Users/pinto.marco/KAUSTProjects/TonitruumUI0/kEEGConnect/matlab/data/FS_16ROI_mean/6791_mean_fs.mat"
"""
python kmindconnect.py --matlab-field="mean_roi" \
--em-max-iterations=1  \
--labels="/Users/pinto.marco/KAUSTProjects/TonitruumUI0/kEEGConnect/matlab/data/ROI.txt" \
--brain-surface="./brain_structure_16.svg" \
--output-dir="./outputs/results-[[dataset]]" \
"/Users/pinto.marco/KAUSTProjects/TonitruumUI0/kEEGConnect/matlab/data/FS_16ROI_mean/6791_mean_fs.mat"
"""
from kMindConnect import SVARConnectivity
import sys
import os


def parse_text(arguments, key):
    try:
        return SVARConnectivity.read_txt(arguments[key])
    except:
        print("{0} must be a valid file".format(key))
        sys.exit(1)


def parse(arguments, key, type):
    try:
        return type(arguments[key])
    except:
        print("{0} must be of type: {1}".format(key, type))
        sys.exit(1)

from docopt import docopt
import shutil
if __name__ == "__main__":
    arguments = docopt(__doc__, version='kMindConnect')
    var_order = parse(arguments, "--var-order", int)
    window_length = parse(arguments, "--window-length", int)
    window_shift = parse(arguments, "--window-shift", int)
    number_states = parse(arguments, "--number-states", int)
    em_tolerance = parse(arguments, "--em-tolerance", float)
    em_max_iterations = parse(arguments, "--em-max-iterations", int)
    labels= parse_text(arguments, "--labels")
    output_folder = arguments["--output-dir"]
    clean_output_folder = arguments["--clean-output-dir"]
    dataset_field = arguments["--matlab-field"]
    brain_surface_reference = arguments["--brain-surface"]
    matlab_engine_path = os.path.realpath(arguments["--matlab-model-path"])
    datasets = arguments["<datasets>"]

    print(arguments)


    connectivity = SVARConnectivity(
        var_order, window_length, window_shift,
        number_states, em_tolerance, em_max_iterations, labels, output_folder, brain_surface_reference, matlab_engine_path)
    
    for dataset in datasets:
        updated_name = output_folder.replace("[[dataset]]", os.path.splitext(os.path.basename(dataset))[0])
        connectivity.output_folder = updated_name
        if clean_output_folder and os.path.exists(connectivity.output_folder):
            shutil.rmtree(connectivity.output_folder)
        connectivity.run(dataset, dataset_field)

