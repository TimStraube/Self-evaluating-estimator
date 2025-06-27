from enum import Enum


class Trainingszustand(Enum):
    BEREIT = "bereit"
    AKTIV = "aktiv"
    INAKTIV = "inaktiv"
    FERTIG = "fertig"
    FEHLER = "fehler"