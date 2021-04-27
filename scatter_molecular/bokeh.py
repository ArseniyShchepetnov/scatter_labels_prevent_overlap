"""Prepare plots"""
from typing import List

import numpy as np

from bokeh.models import Arrow, Label, NormalHead
from bokeh.plotting import Figure, figure
from scatter_molecular.molecular import (Anchor, AnchorAttraction, Particle,
                                         ParticleRepulsion, System)


class RowPlot:
    """Plot row with text labels."""

    def __init__(self,
                 text: List[str],
                 x0: float = 0,
                 y0: float = 0,
                 stride: float = 1,
                 anchor_rad: float = 0.1,
                 repulsion_k: float = 0.1,
                 anchor_k: float = 0.5):
        """
        Parameters
        ----------
        text : List[str]
            List of text labels.
        x0 : float, optional
            Start X-axis anchors position, by default 0.
        y0 : float, optional
            Y-axis position for anchors, by default 0.
        stride : float, optional
            Interval between anchors, by default 1.
        anchor_rad : float, optional
            Anchor effective radius, by default 0.1.
        repulsion_k : float, optional
            Labels repulsion coefficient, by default 0.1.
        anchor_k : float, optional
            Anchor attraction coefficient, by default 0.5.
        """

        particles = []
        anchors = []
        connections = {}

        for ind in range(len(text)):

            y_offset = (np.random.random() - 0.5) * anchor_rad

            label_pos = np.array([x0 + stride * ind, y0 + y_offset])
            anchor_pos = np.array([x0, y0])

            label = Particle(label_pos, mass=1)
            anchor = Anchor(anchor_pos)

            particles.append(label)
            anchors.append(anchor)

            connections[ind] = ind

        repulsion_force = ParticleRepulsion(repulsion_k)
        anchor_force = AnchorAttraction(anchor_k, anchor_rad)

        self.system = System(particles,
                             anchors=anchors,
                             connections=connections,
                             particle_force=repulsion_force,
                             anchor_force=anchor_force)

        self.text = text

    def prepare(self, n_iter):
        """Prepare system."""
        self.system.run(n_iter=n_iter)

    def plot(self, **fig_args) -> Figure:
        """Plot labels with anchors."""

        fig = figure(**fig_args)

        x = [ptcl.pos[0] for ptcl in self.system.particles]
        y = [ptcl.pos[1] for ptcl in self.system.particles]

        a_x = [ptcl.pos[0] for ptcl in self.system.anchors]
        a_y = [ptcl.pos[1] for ptcl in self.system.anchors]

        fig.circle(a_x, a_y)

        for ind in range(len(x)):

            labels = Label(x=x[ind], y=y[ind], text=self.text[ind])

            fig.add_layout(labels)
            fig.add_layout(Arrow(end=NormalHead(),
                                 x_start=x[ind],
                                 y_start=y[ind],
                                 x_end=a_x[ind],
                                 y_end=a_y[ind]))

        return fig
