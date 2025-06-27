from typing import Tuple

import numpy
from enum import Enum

# Define the Aktion type
Aktion = Tuple[numpy.uint32, numpy.ndarray]
# Define the observation type
Beobachtung = numpy.ndarray

class Umweltzustand(Enum):
    BEREIT = "bereit"
    AKTIV = "aktiv"
    INAKTIV = "inaktiv"
    FERTIG = "fertig"
    FEHLER = "fehler"