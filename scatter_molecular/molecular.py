"""Molecular-like system to prevent objects overlap."""
import abc
from typing import Dict, List, Optional

import numpy as np


class Force(metaclass=abc.ABCMeta):
    """Base force class."""

    @abc.abstractmethod
    def calc(self, src: np.ndarray, tar: np.ndarray) -> np.ndarray:
        """
        Calculate force from coordinates.

        Parameters
        ----------
        src : np.ndarray
            Vector of source coordinates.
        tar : np.ndarray
            Vector of target coordinates.

        Return
        ------
        np.ndarray
            Force values vector.
        """


class AnchorAttraction(Force):
    """
    Attraction force to anchors.

    When distance to target is higher then radius -
    linear attraction force is observed.
    When distance to target is lower than radius, then
    linear repulsion force happens.
    """

    def __init__(self, k: float, rad: float = 0):
        """
        Parameters
        ----------
        k : float
            Force coefficient.
        rad : float, optional
            Radius of the anchor, by default 0.
            Repulsion applied when target entering radius.
        """
        self.k = k
        self.rad = rad

    def calc(self,
             src: np.ndarray,
             tar: np.ndarray) -> np.ndarray:
        """Force realization."""

        dist = np.linalg.norm(src - tar)

        if dist >= self.rad:
            result = - self.k * (src - tar) / dist * dist
        else:
            result = self.k * (src - tar) / dist * dist

        return result


class ParticleRepulsion(Force):

    def __init__(self, k: float):
        """[summary]

        Parameters
        ----------
        k : float
            Force coefficient.
        """
        self.k = k

    def calc(self,
             src: np.ndarray,
             tar: np.ndarray) -> np.ndarray:

        dist = np.linalg.norm(src - tar)

        result = self.k * (src - tar) / dist / dist

        return result


class Body:
    """Movable body."""

    def __init__(self,
                 pos: np.ndarray,
                 vel: Optional[np.ndarray] = None,
                 mass: float = 0):
        """
        Parameters
        ----------
        pos : np.ndarray
            Position vector.
        vel : Optional[np.ndarray], optional
            Velocity vector, by default None.
        mass : float, optional
            Mass, by default 0.
        """

        self._pos = pos.astype(np.float32)

        if vel is None:
            vel = np.zeros(shape=self._pos.shape)

        self._vel = vel.astype(np.float32)

        self._mass = mass

    @property
    def pos(self) -> np.ndarray:
        """Postion vector."""
        return self._pos

    @property
    def vel(self) -> np.ndarray:
        """Velocity vector."""
        return self._vel

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}: "
                f"pos({self._pos}), "
                f"vel({self._vel}), "
                f"mass({self._mass})")


class Anchor(Body):
    """Anchor class. No move, position only."""


class Particle(Body):
    """Interactive particle."""

    def move(self, dt: float):
        """
        Move particle with time increment. Change position using velocity.

        Parameters
        ----------
        dt : float
            Time increment
        """
        self._pos += self._vel * dt

    def apply_force(self, force: np.ndarray, dt: float):
        """Change velocity using force value."""
        self._vel += (force / self._mass * dt)


class System:
    """System of anchors and particles."""

    dt = 1
    dim = 2

    def __init__(self,
                 particles: List[Particle],
                 anchors: List[Anchor],
                 connections: Dict[int, int],
                 particle_force: Force,
                 anchor_force: Force):
        """
        Parameters
        ----------
        particles : List[Particle]
            List of particles.
        anchors : List[Anchor]
            List of anchors.
        connections : Dict[int, int]
            Dictionary of index relations between anchors and particles
            from lists.
        particle_force : Force
            Force between particles.
        anchor_force : Force
            Force between anchors and particles.
        """

        self._particles = particles
        self._anchors = anchors
        self._connections = connections
        self._particle_force = particle_force
        self._anchor_force = anchor_force

        n_particles = len(self._particles)
        self._force = np.zeros(shape=(n_particles, ))

    @property
    def particles(self) -> List[Particle]:
        """List of particles."""
        return self._particles

    @property
    def anchors(self) -> List[Anchor]:
        """List of anchors."""
        return self._anchors

    def _one_particle_force(self, index: int) -> np. ndarray:
        """
        Calculate full force applied to one particle with `index` in list.
        """

        force_repulsion = np.zeros(shape=(self.dim, ))

        for other in range(len(self._particles)):

            if index != other:

                force_repulsion += (
                    self._particle_force.calc(self._particles[index].pos,
                                              self._particles[other].pos)
                )

        anchor = self._connections[index]
        force_attraction = self._anchor_force.calc(self._particles[index].pos,
                                                   self._anchors[anchor].pos)

        result = force_repulsion + force_attraction

        return result

    def _calc_particle_forces(self):
        """Calculate full forces for particles."""

        forces = np.zeros(shape=(len(self._particles), self.dim),
                          dtype=np.float32)

        for index in range(len(self._particles)):

            force = self._one_particle_force(index)

            forces[index, :] += force

        return forces

    def _apply_force_particles(self, forces: np.ndarray):
        """Apply forces to particles."""

        for i in range(len(self._particles)):

            self._particles[i].apply_force(forces[i, :], self.dt)

    def _move_particles(self):
        """Move particles."""

        for i in range(len(self._particles)):

            self._particles[i].move(self.dt)

    def next(self):
        """One step in time."""

        forces = self._calc_particle_forces()
        self._apply_force_particles(forces)
        self._move_particles()

    def run(self, n_iter: int = 5):
        """Run multiple steps."""

        for _ in range(n_iter):
            self.next()
