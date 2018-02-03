import os
import gc
import sys
import numpy as np

def openFile(arrows, filename_png):
	brainFile = "Brain2.svg"
	brainOutFile = "brainModified.svg"
	brainOutFilePNG = filename_png
	codeToReplace = "<!--REPLACE-->"
	brainSvg = open(brainFile, "rt").read()
	for arrow in arrows:
		brainSvg = brainSvg.replace(codeToReplace, arrow + codeToReplace)
	open(brainOutFile, "wt").write(brainSvg)
	brainSvg = None
	gc.collect()
	command_line = "inkscape --file='{0}' --export-width=700 --export-height=368 --export-png='{1}'".format(brainOutFile, brainOutFilePNG)
	#command_line = "inkscape --file='{0}' --export-pdf='{1}'".format(brainOutFile, brainOutFilePNG.replace(".png", ".pdf"))
	os.system(command_line)

"""
    <path
       style="fill:none;fill-rule:evenodd;stroke:#000000;stroke-width:3;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1;marker-end:url(#Arrow2Lend);stroke-miterlimit:4;stroke-dasharray:none"
       d="m -303.5349,143.85983 408.94221,4.36205"
       id="path6112"
       inkscape:connector-curvature="0" />
"""
"""
    <line x1="100" y1="50" x2="250" y2="50" stroke="#000" stroke-width="3" 
       marker-end="url(#Arrow2Lend)" marker-start="url(#DotM)" />
"""
def transformCode(fromROI, fromHem, toROI, toHem, strength):
	ROIs = {
		"L": {
			"PreCG": (288,414),
			"CG": (271,386),
			"PoCG": (247,380),
			"TTG": (230,325),
			"TTS": (215,320),
			"PT": (162,353),
			"STG": (256,268),
			"PP": (340,257),
		},
		"R": {
			"PreCG": (680,437),
			"CG": (675,382),
			"PoCG": (758,433),
			"TTG": (699,305),
			"TTS": (739,319),
			"PT": (802,355),
			"STG": (679,275),
			"PP": (617,251),
		}
	}
	print strength, max(0.5, 4 * strength)
	return """
    <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#000" stroke-width="{width}" style="opacity:{strength};" 
       marker-end="url(#Arrow2Lend)" marker-start="url(#DotM)" />
""".format(
		x1=ROIs[fromHem][fromROI][0]-575,
		y1=540-ROIs[fromHem][fromROI][1],
		x2=ROIs[toHem][toROI][0]-575,
		y2=540-ROIs[toHem][toROI][1],
		strength=strength,
		width=max(0.5, 4 * strength),
	)
	return """
    <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#000" stroke-width="2" style="opacity:{strength};"
       marker-end="url(#Arrow2Lend)" marker-start="url(#DotM)" />
""".format(
		x1=ROIs[fromHem][fromROI][0]-575,
		y1=540-ROIs[fromHem][fromROI][1],
		x2=ROIs[toHem][toROI][0]-575,
		y2=540-ROIs[toHem][toROI][1],
		strength=strength,
	)
	return "%s%d%d" % (fromHem, ROIs[fromROI], ROIs[toROI])

"""
arrows = [
	transformCode("CG", "L", "PreCG", "R",1),
	transformCode("CG", "L", "CG", "R", 1),
	transformCode("CG", "L", "PoCG", "R", 0.1),
	transformCode("CG", "L", "TTG", "R", 0.5),
	transformCode("CG", "L", "TTS", "R", 0.7),
	transformCode("CG", "L", "PT", "R", 0.01),
	transformCode("CG", "L", "STG", "R", 0.1),
	transformCode("CG", "L", "PP", "R", 0.4),
	transformCode("TTS", "L", "PreCG", "R", 0.1),
	#transformCode("TTS", "L", "PP", "L"),
]
openFile(arrows, "brainTest1.png")
"""

def read_csv(fileCSV, filePNG):
	matrix = np.array(np.genfromtxt(fileCSV))
	order = [
	"CG",
	"PP",
	"PT",
	"PoCG",
	"PreCG",
	"STG",
	"TTG",
	"TTS",
	]
	h = ["L", "R"]
	matrix = np.exp(matrix)
	p = np.eye(len(matrix))
	minM, maxM = np.min(matrix + 1000 * p), np.max(matrix - 1000 * p)
	matrix = (matrix - minM) / (maxM - minM)
	matrix = matrix - matrix * p 
	print np.round(matrix,1), np.min(matrix), np.max(matrix)
	arrows = []
	for k1, fromH in enumerate(h):
		for j, fromV in enumerate(order):
			for k2, toH in enumerate(h):
				for i, toV in enumerate(order):
					arrows.append(transformCode(fromV, fromH, toV, toH, matrix[j + k1 * len(order)][i + k2 * len(order)]))
	openFile(arrows, filePNG)

read_csv("StateMatrix01.csv", "BrainState1.png")
read_csv("StateMatrix02.csv", "BrainState2.png")
read_csv("StateMatrix03.csv", "BrainState3.png")

