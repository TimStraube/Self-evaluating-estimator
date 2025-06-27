import random

from src.types.env import Umweltzustand

class Umwelt():
    def __init__(self, umwelt):
        self.umwelt = umwelt
        self.zustand_welt = self.umwelt.restart()
        self.aktionsraum = self.umwelt.aktionsraum
        self.beobachtungsraum = self.zustand_welt.shape
        self.zustand : Umweltzustand = Umweltzustand.BEREIT

    def observe(self):
        return self.zustand_welt

    def act(self, aktion):
        self.zustand = Umweltzustand.AKTIV
        # Erwartet jetzt einen Integer als Aktion!
        self.zustand_welt = self.umwelt.step(
            self.zustand_welt, aktion)
            
    def simuliere(self):
        # Simuliere nur einen Schritt!
        if self.zustand == Umweltzustand.FERTIG:
            self.zustand_welt = self.umwelt.restart()
            self.zustand = Umweltzustand.BEREIT
        else:
            self.zustand = Umweltzustand.AKTIV
            aktion = random.randint(0, self.aktionsraum - 1)
            self.zustand_welt = self.umwelt.step(self.zustand_welt, aktion)

