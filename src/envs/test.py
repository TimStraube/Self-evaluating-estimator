import random
import numpy

class Test:
    def __init__(self, size=3):
        self.size = size
        self.aktionsraum = (
            self.size * self.size
        )

    def __repr__(self):
        return "test"

    def restart(self):
        zustand_umwelt = numpy.zeros(
            (self.size, self.size),
            dtype=numpy.uint8
        )
        return zustand_umwelt

    def step(self, zustand, aktion):
        zustand = zustand.copy()
        # Aktion ggf. aus Liste extrahieren
        if isinstance(aktion, (list, tuple, numpy.ndarray)):
            aktion = aktion[0]
        x = aktion // self.size
        y = aktion % self.size
        zustand[x, y] += 1
        return zustand

    def terminated(self, zustand, aktion):
        terminated = False
        return terminated