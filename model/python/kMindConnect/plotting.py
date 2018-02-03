import os
import gc
import numpy as np
from plotly import tools
import plotly.offline as offline
import plotly.graph_objs as go
import sklearn.decomposition

class Plotting:
    @staticmethod
    def _plot(fig, filename, **config):
        dataconfig = {
            'displaylogo': False,
            'scrollZoom': True,
            'displayModeBar': True,
            'editable': not True,
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
        }
        for k, v in config.items():
            dataconfig[k] = v
        offline.plot(
            fig,
            show_link=False,
            link_text='Plot.ly',
            validate=False,
            output_type='file',
            include_plotlyjs=True,
            filename=filename,
            auto_open=False,
            image=None,
            image_filename=None,#filename.replace(".html", ".png"),
            image_width=800,
            image_height=800,
            config=dataconfig,
        )
        filename = filename.replace(".html", ".div.html")
        html = offline.plot(
            fig,
            show_link=False,
            link_text='Plot.ly',
            validate=False,
            output_type='div',
            include_plotlyjs=False,
            filename=filename,
            auto_open=False,
            image=None,
            image_filename=None,  #filename.replace(".html", ".png"),
            image_width=800,
            image_height=800,
            config={
                'displaylogo': False,
                'scrollZoom': True,
                'displayModeBar': not True,
                'editable': not True
            }
        )
        with open(filename, "w") as f:
            f.write(html)
        
    @staticmethod
    def scatter(dataseries, names, filename, title=""):
        traces = []
        for data, name in zip(dataseries, names):
            trace = go.Scatter(
                x=data[:, 0],
                y=data[:, 1],
                name=name,
                mode='markers',
                marker=dict(
                    size=20 if name == "centroids" else 5,
                )
            )
            traces.append(trace)
        layout = dict(
            title=title,
            yaxis=dict(zeroline=False),
            xaxis=dict(zeroline=False),
            #height=400,
            #width=400,
            autosize=not False,
            showlegend=False,
            font=dict(size=16),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        fig = dict(data=traces, layout=layout)
        Plotting._plot(fig, filename)

    @staticmethod
    def one_heatmap(dataseries, filename, title="", labels=None, reduce_margins=False, **kwargs):
        dataseries = np.array(dataseries)
        vmin, vmax = np.min(dataseries), np.max(dataseries)
        fig = tools.make_subplots(
            print_grid=False,
            rows=1, cols=1,
            vertical_spacing=0)
        y = dataseries
        if labels is None:
            x = np.arange(len(y))
        else:
            x = labels[:len(y)]
        print(x)
        trace = go.Heatmap(
            x=x,
            y=x,
            z=y,
            zmin=vmin,
            zmax=vmax,
            showscale=not reduce_margins,
        )
        fig.append_trace(trace, 1, 1)
        fig['layout'].update(margin=dict(t=10, pad=4),)
        if reduce_margins:
            fig['layout'].update(margin=dict(l=10, r=10, b=10, t=10, pad=4),)
        fig['layout'].update(
            #height=400,
            #width=400,
            autosize=not False,
            title=title,
            showlegend=False,
            font=dict(size=16),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',)
        Plotting._plot(fig, filename, **kwargs)

    @staticmethod
    def heatmap(dataseries, filename, title="", labels=None):
        dataseries = np.array(dataseries)
        vmin, vmax = np.min(dataseries), np.max(dataseries)
        n = dataseries.shape[-1]
        fig = tools.make_subplots(
            print_grid=False,
            rows=1, cols=n,
            subplot_titles=["State {0}".format(i + 1) for i in range(n)],
            #vertical_spacing=0
            shared_xaxes=False,
            shared_yaxes=False,
        )
        for i in range(n):
            # Visualization purposes
            Plotting.one_heatmap(dataseries[:, :, i],
                                 #title="State {0}".format(i+1),
                                 filename=filename.replace(
                                     ".html", "-state-{0}.html".format(i + 1)),
                                 reduce_margins=False,
                                 displayModeBar=False, labels=labels)
            Plotting.one_heatmap(dataseries[:, :, i],
                                 #title="State {0}".format(i+1),
                                 filename=filename.replace(
                                     ".html", "-state-no-margin-{0}.html".format(i + 1)),
                                 reduce_margins=True,
                                 displayModeBar=False, labels=labels)
            #
            y = dataseries[:, :, i]
            if labels is None:
                x = np.arange(len(y))
            else:
                x = labels[:len(y)]
            trace = go.Heatmap(
                x=x,
                y=x,
                z=y,
                zmin=vmin,
                zmax=vmax,
                #showscale=False,
            )
            fig.append_trace(trace, 1, i + 1)
        fig['layout'].update(
            height=400,
            #width=600,
            autosize=not False,
            title=title,
            showlegend=False,
            font=dict(size=16))
        Plotting._plot(fig, filename)

        

    @staticmethod
    def heatmap_v(dataseries, filename, title="", labels=None):
        dataseries = np.array(dataseries)
        vmin, vmax = np.min(dataseries), np.max(dataseries)
        n = dataseries.shape[-1]
        fig = tools.make_subplots(
            print_grid=False,
            rows=n, cols=1,
            subplot_titles=["State {0}".format(i + 1) for i in range(n)],
            #vertical_spacing=0
            shared_xaxes=False,
            shared_yaxes=False,
        )
        for i in range(n):
            y = dataseries[:, :, i]
            if labels is None:
                x = np.arange(len(y))
            else:
                x = labels[:len(y)]
            trace = go.Heatmap(
                x=x,
                y=x,
                z=y,
                zmin=vmin,
                zmax=vmax,
                showscale=False,
            )
            fig.append_trace(trace, i + 1, 1)
        fig['layout'].update(
            height=n * 400,
            width=600,
            autosize=not False,
            title=title,
            showlegend=False,
            font=dict(size=16))
        Plotting._plot(fig, filename)

    @staticmethod
    def series(dataseries, filename, title=""):
        dataseries = np.array(dataseries)
        n = len(dataseries)
        traces = []
        fig = tools.make_subplots(
            print_grid=False,
            rows=n, cols=1,
            shared_xaxes=True,
            shared_yaxes=False,
            vertical_spacing=0)
        for i in range(n):
            y = dataseries[i]
            trace = go.Scatter(
                x=np.arange(len(y)),
                y=y
            )
            fig.append_trace(trace, i + 1, 1)
        fig['layout'].update(
            height=n * 200,
            #width=800,
            autosize=not False,
            title=title,
            showlegend=False,
            font=dict(size=16))
        Plotting._plot(fig, filename)

    @staticmethod
    def matrix_series(dataseries, filename, columns, title="", skip=1, staticPlot=False, transpose=False, height=200):
        dataseries = np.array(dataseries)
        if transpose:
            dataseries = dataseries.T
        n = len(dataseries) // columns
        fig = tools.make_subplots(
            print_grid=False,
            rows=n, cols=columns,
            shared_xaxes=True,
            shared_yaxes=False,
            vertical_spacing=0)
        for i in range(n):
            for j in range(columns):
                y = dataseries[i * columns + j]
                x = np.arange(0, len(y), skip)
                trace = go.Scatter(
                    x=x,
                    y=y[x]
                )
                fig.append_trace(trace, i + 1, j + 1)
                fig['layout']['yaxis%d' % (i * columns + j + 1)].update(
                    autorange=True,
                    showgrid=False,
                    zeroline=False,
                    showline=False,
                    autotick=True,
                    ticks='',
                    showticklabels=False
                )
        fig['layout'].update(
            height=n * height,
            #width=800,
            autosize=not False,
            title=title,
            showlegend=False,
            font=dict(size=16))
        Plotting._plot(fig, filename)
    
    @staticmethod
    def multiary_series(dataseries, filename, columns, title="", skip=1, staticPlot=False, transpose=False, height=200):
        y = np.array(dataseries)
        states = np.unique(y)
        dataseries = np.ones((len(states), len(y)))
        for k, v in enumerate(states):
            dataseries[k] = ((y == v) * (k + 1)).ravel()
        n = len(dataseries) // columns
        #fig = go.Figure()
        traces = []
        for i in range(n):
            y = dataseries[i]
            x = np.arange(0, len(y), skip)
            trace = go.Bar(
                x=x,
                y=y[x],# * (i + 1),
                name="State {0}".format(i + 1),
            )
            traces.append(trace)
        layout = go.Layout(
            barmode='stack',
            #height=n * height,
            #width=800,
            autosize=not False,
            title=title,
            showlegend=False,
            font=dict(size=16),
            yaxis=dict(
                autorange=True,
                showgrid=False,
                zeroline=False,
                showline=False,
                autotick=True,
                ticks='',
                showticklabels=False
            )
        )
        fig = go.Figure(data=traces, layout=layout)
        Plotting._plot(fig, filename)

    @staticmethod
    def clusters(points, labels, centroids, filename, title):
        if np.min(labels) > 0:
            labels = labels - np.min(labels)
        pca = sklearn.decomposition.PCA(n_components=2)
        pca.fit(points)
        cdata = [pca.transform(centroids)]
        clabels = ['centroids']
        for i, c in enumerate(centroids):
            cdata.append(pca.transform(points[labels == i]))
            clabels.append("cluster {0}".format(i + 1))
        Plotting.scatter(
            cdata,
            clabels,
            filename,
            title=title
        )

    @classmethod
    def _svg_roi_plot_arrows(cls, arrows, filename_png, brain_file_input, brain_file_output):
        codeToReplace = "<!--REPLACE-->"
        brainSvg = open(brain_file_input, "rt").read()
        for arrow in arrows:
            brainSvg = brainSvg.replace(codeToReplace, arrow + codeToReplace)
        open(brain_file_output, "wt").write(brainSvg)
        brainSvg = None
        gc.collect()
        if filename_png is None:
            return
        command_line = "inkscape --file='{0}' --export-width=700 --export-height=368 --export-png='{1}'".format(
            brain_file_output, filename_png)
        os.system(command_line)


    @classmethod
    def _svg_roi_transform_code(cls, fromROI, fromHem, toROI, toHem, strength):
        ROIs = {
            "L": {
                "PreCG": (288, 414),
                "CG": (271, 386),
                "PoCG": (247, 380),
                "TTG": (230, 325),
                "TTS": (215, 320),
                "PT": (162, 353),
                "STG": (256, 268),
                "PP": (340, 257),
            },
            "R": {
                "PreCG": (680, 437),
                "CG": (675, 382),
                "PoCG": (758, 433),
                "TTG": (699, 305),
                "TTS": (739, 319),
                "PT": (802, 355),
                "STG": (679, 275),
                "PP": (617, 251),
            }
        }
        if strength < 0.5:
            strength = 0
        #print(strength, max(0.5, 4 * strength))
        return """
        <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#000" stroke-width="{width}" style="opacity:{strength};" 
        marker-end="url(#Arrow2Lend)" marker-start="url(#DotM)" />
        """.format(
            x1=ROIs[fromHem][fromROI][0] - 575,
            y1=540 - ROIs[fromHem][fromROI][1],
            x2=ROIs[toHem][toROI][0] - 575,
            y2=540 - ROIs[toHem][toROI][1],
            strength=strength,
            width=max(0.5, 4 * strength),
        )

    @classmethod
    def coherence_matrix_surface(cls, coherence_matrix, file_output, brain_file_reference, convert_to_png=False):
        matrix = coherence_matrix
        if matrix.shape != (16, 16):
            raise ValueError("It is not implemented yet to plot a coherence matrix for this kind of data")
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
        #print(np.round(matrix, 1), np.min(matrix), np.max(matrix))
        arrows = []
        for k1, fromH in enumerate(h):
            for j, fromV in enumerate(order):
                for k2, toH in enumerate(h):
                    for i, toV in enumerate(order):
                        arrows.append(cls._svg_roi_transform_code(fromV, fromH, toV, toH,
                                                                matrix[j + k1 * len(order)][i + k2 * len(order)]))
        cls._svg_roi_plot_arrows(
            arrows, (file_output + ".png") if convert_to_png else None, brain_file_reference, file_output + ".svg")
