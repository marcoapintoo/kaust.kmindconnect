"""
kMindConnect 0.2

Usage:
    kmindconnect [options] --labels=<l> --output-dir=<o> <datasets>...
    kmindconnect --only-get-hash [options] --labels=<l> --output-dir=<o> <datasets>...
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
--use-octave-engine=<e>             Use Octave engine rather than Python version [default: True]
--only-get-hash                     Only calculate the configuration hash [default: False]
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
import sys
sys.path.extend(['',
'/Users/ombaohc/anaconda/lib/python36.zip',
'/Users/ombaohc/anaconda/lib/python3.6',
'/Users/ombaohc/anaconda/lib/python3.6/lib-dynload',
'/Users/ombaohc/.local/lib/python3.6/site-packages',
'/Users/ombaohc/anaconda/lib/python3.6/site-packages',
'/Users/ombaohc/anaconda/lib/python3.6/site-packages/Sphinx-1.5.6-py3.6.egg',
'/Users/ombaohc/anaconda/lib/python3.6/site-packages/aeosa'])

from kMindConnect import ExperimentSettings
from docopt import docopt

if __name__ == "__main__":
    arguments = docopt(__doc__, version='kMindConnect')
    experiment = ExperimentSettings(arguments)
    experiment.run()
