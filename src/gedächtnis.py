from src.erinnerung import Erinnerung
import numpy as np
from src.erinnerung import Erinnerung

class Gedächtnis():
    def __init__(self, kapazität):
        self.kapazität = kapazität
        self.speicher = np.array([Erinnerung(np.zeros((3, 3)), 0) for _ in range(kapazität)], dtype=object)
        self.schreibZeiger = 0
        self.belohnungen = []

    def update(self, erfahrung):
        # print("Updating memory with state of mind: " + str(stateOfMind))
        # For a new input a stateOfMind is forgotten
        self.speicher[self.schreibZeiger] = erfahrung

    def getErinnerung(self):
        return self.speicher[self.schreibZeiger]
    
    def getBild(self):
        return self.speicher[self.schreibZeiger].getBild()

    def setSchreibZeiger(self, schreibZeiger):
        self.schreibZeiger = schreibZeiger

    def getKapazität(self):
        return self.kapazität

    def getReward(self, schreibZeiger):
        return self.speicher[schreibZeiger].getBelohnung()

    def getMeanReward(self):
        for i in range(self.kapazität):
            self.belohnungen.append(self.speicher[i].getBelohnung())
        return np.mean(self.belohnungen)
