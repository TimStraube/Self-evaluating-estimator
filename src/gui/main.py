import sys
import os
import importlib
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QVBoxLayout,
    QWidget,
    QPushButton,
    QComboBox,
    QMessageBox,
    QHBoxLayout,
    QGroupBox,
    QSpinBox,
    QScrollArea,  # hinzugefügt
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QColor, QPixmap, QPainter, QImage, QFont
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.types.agent import Trainingszustand

from umwelt import Umwelt
from src.agent import Agent
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Electric Columbus GUI")

        # Hauptlayout horizontal: links Agent, rechts Umwelt
        self.main_layout = QHBoxLayout()
        self.layout = QVBoxLayout()  # Für die rechte Seite (Umwelt)

        # Agent-Box mit Label und Memory-Visualisierung
        self.agent_box = QGroupBox("Agent")
        self.agent_layout = QVBoxLayout()

        # Trainingszustand-Statuspunkt
        self.training_status_label = QLabel()
        self.training_status_label.setFixedHeight(24)
        self.agent_layout.addWidget(self.training_status_label)
        self.trainingszustand = Trainingszustand.INAKTIV
        self.update_training_status_point(self.trainingszustand)

        # Eingabe für Gedächtnisgröße und Reset-Button
        self.memory_control_layout = QHBoxLayout()
        self.memory_size_spin = QSpinBox()
        self.memory_size_spin.setMinimum(1)
        self.memory_size_spin.setMaximum(100)
        self.memory_size_spin.setValue(5)
        self.memory_size_spin.setPrefix("Zellen: ")
        self.memory_reset_button = QPushButton("Gedächtnis resetten")
        self.memory_reset_button.clicked.connect(self.reset_memory)
        self.memory_control_layout.addWidget(self.memory_size_spin)
        self.memory_control_layout.addWidget(self.memory_reset_button)
        self.agent_layout.addLayout(self.memory_control_layout)

        # Trainings-Toggle-Button
        self.train_toggle_button = QPushButton("Training starten")
        self.train_toggle_button.setCheckable(True)
        self.train_toggle_button.clicked.connect(self.toggle_training)
        self.agent_layout.addWidget(self.train_toggle_button)

        # Gedächtnis-Visualisierung als scrollbarer Bereich
        self.memory_label = QLabel()
        self.memory_label.setWordWrap(True)
        self.memory_scroll = QScrollArea()
        self.memory_scroll.setWidgetResizable(True)
        self.memory_scroll.setWidget(self.memory_label)
        self.memory_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.memory_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.agent_layout.addWidget(self.memory_scroll)

        # Matplotlib-Plot für Rewards
        self.reward_fig, self.reward_ax = plt.subplots()
        self.reward_canvas = FigureCanvas(self.reward_fig)
        self.reward_canvas.setFixedHeight(160)
        self.agent_layout.addWidget(self.reward_canvas)
        self.agent_box.setLayout(self.agent_layout)

        # Umwelt-Box (analog zur Agent-Box)
        self.umwelt_box = QGroupBox("Umwelt")
        self.umwelt_layout = QVBoxLayout()
        # Statuspunkt für Umweltzustand (Emoji + Text)
        self.status_label = QLabel()
        self.status_label.setFixedHeight(30)
        self.umwelt_layout.addWidget(self.status_label)
        self.update_status_point("inaktiv")  # Initialer Zustand

        self.label = QLabel("Umweltzustand wird hier angezeigt.")
        self.umwelt_layout.addWidget(self.label)

        # Dropdown für Umwelten
        self.env_combo = QComboBox()
        self.envs = self.get_env_classes()
        self.env_combo.addItems(list(self.envs.keys()))
        self.umwelt_layout.addWidget(self.env_combo)

        self.load_button = QPushButton("Umwelt laden")
        self.load_button.clicked.connect(self.load_umwelt)
        self.umwelt_layout.addWidget(self.load_button)

        self.sim_button = QPushButton("Simulation starten")
        self.sim_button.clicked.connect(self.toggle_simulation)
        self.sim_button.setEnabled(False)
        self.umwelt_layout.addWidget(self.sim_button)
        self.umwelt_box.setLayout(self.umwelt_layout)

        # Layouts zusammenführen
        self.main_layout.addWidget(self.agent_box)
        self.main_layout.addWidget(self.umwelt_box)

        container = QWidget()
        container.setLayout(self.main_layout)
        self.setCentralWidget(container)

        # Agent initialisieren mit Gedächtnisgröße
        self.agent = Agent()
        self.agent.gedächtnis = self.create_memory(self.memory_size_spin.value())
        self.umwelt = self.agent.umwelt

        self.timer = QTimer()
        self.timer.timeout.connect(self.simulation_step)
        self.sim_steps = 0
        self.max_steps = 10
        self.sim_running = False

        self.gedächtnis = self.create_memory(self.memory_size_spin.value())

        # Visualisierungsmodus-Auswahl
        self.memory_view_mode_combo = QComboBox()
        self.memory_view_mode_combo.addItems(["Bild (Pixel)", "Zahlenmatrix"])
        self.memory_view_mode_combo.currentIndexChanged.connect(self.update_memory_visualization)
        self.agent_layout.addWidget(self.memory_view_mode_combo)

    def get_env_classes(self):
        env_dir = os.path.join(os.path.dirname(__file__), "../envs")
        env_files = [
            f
            for f in os.listdir(env_dir)
            if f.endswith(".py") and not f.startswith("__")
        ]
        envs = {}
        for file in env_files:
            modulename = f"envs.{file[:-3]}"
            module = importlib.import_module(modulename)
            # Suche nach einer Klasse mit gleichem Namen wie das File (Case-insensitive)
            classname = file[:-3].capitalize()
            for attr in dir(module):
                if attr.lower() == file[:-3].lower():
                    envs[attr] = (modulename, attr)
        return envs

    def update_status_point(self, zustand):
        emojis = {
            "bereit": "🟢",
            "aktiv": "🟢",
            "inaktiv": "🟡",
            "fertig": "🟢",
            "fehler": "🔴",  
        }
        emoji = emojis.get(str(zustand).lower(), "⚪️")
        self.status_label.setText(f"{emoji}  {zustand.capitalize()}")

    def load_umwelt(self):
        env_name = self.env_combo.currentText()
        modulename, classname = self.envs[env_name]
        try:
            module = importlib.import_module(modulename)
            env_class = getattr(module, classname)
            self.agent.umwelt = Umwelt(env_class())
            self.umwelt = self.agent.umwelt
            self.label.setText(str(self.umwelt.zustand_welt))
            if hasattr(self.umwelt, "zustand"):
                self.update_status_point(self.umwelt.zustand.name if hasattr(self.umwelt.zustand, "name") else self.umwelt.zustand)
            self.update_memory_visualization()
            self.update_reward_plot()
            self.sim_button.setEnabled(True)
        except Exception as e:
            self.umwelt = None
            self.sim_button.setEnabled(False)
            self.status_label.setText("🔴 Fehler beim Laden der Umwelt")
            QMessageBox.critical(self, "Fehler", f"Umwelt konnte nicht geladen werden:\n{e}")

    def toggle_simulation(self):
        if not self.sim_running:
            self.sim_running = True
            self.sim_button.setText("Simulation stoppen")
            self.sim_steps = 0
            self.timer.start(500)
        else:
            self.sim_running = False
            self.sim_button.setText("Simulation starten")
            self.timer.stop()
            # Reset: Umwelt neu laden und Status zurücksetzen
            self.load_umwelt()

    def simulation_step(self):
        if self.sim_steps >= self.max_steps:
            self.timer.stop()
            self.sim_running = False
            self.sim_button.setText("Simulation starten")
            return
        # Beispielaktion: Zufällige Aktion für Agent
        aktion = [np.random.randint(self.agent.gedächtnis.getKapazität()), 0, np.random.randint(self.umwelt.aktionsraum)]
        _, reward, _, _, _ = self.agent.step(aktion)
        self.label.setText(str(self.umwelt.zustand_welt))
        if hasattr(self.umwelt, "zustand"):
            self.update_status_point(self.umwelt.zustand.name if hasattr(self.umwelt.zustand, "name") else self.umwelt.zustand)
        self.sim_steps += 1
        self.update_memory_visualization()
        self.update_reward_plot()

    def create_memory(self, size):
        try:
            from gedächtnis import Gedächtnis
            from erinnerung import Erinnerung
            import numpy as np
            return Gedächtnis(size)
        except Exception:
            return None

    def reset_memory(self):
        size = self.memory_size_spin.value()
        self.agent.gedächtnis = self.create_memory(size)
        self.update_memory_visualization()

    def update_memory_visualization(self):
        memory = self.agent.gedächtnis
        if memory is None:
            self.memory_label.clear()
            return
        mode = self.memory_view_mode_combo.currentText()
        try:
            kap = memory.getKapazität()
            # Hole die Form des Beobachtungsarrays der aktuellen Umwelt
            obs_shape = None
            try:
                obs_shape = tuple(self.umwelt.observe().shape)
            except Exception:
                obs_shape = (1,)
            if len(obs_shape) == 1:
                rows, cols = 1, obs_shape[0]
            elif len(obs_shape) == 2:
                rows, cols = obs_shape
            elif len(obs_shape) == 3:
                # Zeige nur die erste Schicht, falls 3D
                rows, cols = obs_shape[1], obs_shape[2]
            else:
                rows, cols = 1, 1
            schreibzeiger = getattr(memory, "schreibZeiger", None)
            # Matrix-Grid als HTML-Tabelle
            html = "<div style='font-family:monospace; font-size:12px;'>"
            for i in range(kap):
                try:
                    erinnerung = memory.speicher[i]
                    bild = erinnerung.getBild()
                    belohnung = erinnerung.getBelohnung()
                    arr = np.array(bild)
                    # Wähle nur die erste Schicht, falls 3D
                    if arr.ndim == 3:
                        arr = arr[0]
                    arr = arr.reshape((rows, cols))
                    # Highlight für aktuelle Erinnerung
                    if schreibzeiger == i:
                        html += f"<div style='background:#cce5ff; border:2px solid #3399ff; margin:4px; display:inline-block; padding:2px;'>"
                    else:
                        html += f"<div style='background:#f5f5f5; border:1px solid #ccc; margin:4px; display:inline-block; padding:2px;'>"
                    html += f"<b>Zelle {i} (Belohnung: {belohnung:.1f})</b><br>"
                    html += "<table style='border-collapse:collapse;'>"
                    for r in range(rows):
                        html += "<tr>"
                        for c in range(cols):
                            val = arr[r, c]
                            html += f"<td style='border:1px solid #bbb; min-width:18px; text-align:center; padding:2px;'>{val:.1f}</td>"
                        html += "</tr>"
                    html += "</table></div>"
                except Exception:
                    html += f"<div>Zelle {i}: Fehler beim Lesen</div>"
            html += "</div>"
            self.memory_label.setText(html)
            self.memory_label.setMinimumWidth(max(400, (cols+1)*kap*20))
            self.memory_label.setMinimumHeight(min(1200, (rows+1)*kap*20+100))
            self.memory_label.setMaximumHeight(2000)
        except Exception:
            self.memory_label.clear()

    def update_training_status_point(self, zustand):
        emojis = {
            Trainingszustand.BEREIT: "🟢",
            Trainingszustand.AKTIV: "🟢",
            Trainingszustand.INAKTIV: "🟡",
            Trainingszustand.FERTIG: "🟢",
            Trainingszustand.FEHLER: "🔴",
        }
        emoji = emojis.get(zustand, "⚪️")
        self.training_status_label.setText(f"{emoji}  {zustand.value.capitalize()}")

    def toggle_training(self):
        if self.trainingszustand != Trainingszustand.AKTIV:
            self.trainingszustand = Trainingszustand.AKTIV
            self.train_toggle_button.setText("Training stoppen")
        else:
            self.trainingszustand = Trainingszustand.INAKTIV
            self.train_toggle_button.setText("Training starten")
        self.update_training_status_point(self.trainingszustand)

    def update_reward_plot(self):
        rewards = self.agent.belohnungen
        self.reward_ax.clear()
        if len(rewards) > 0:
            self.reward_ax.plot(rewards, label="Reward", color="#0077bb")
            self.reward_ax.set_xlabel("Step")
            self.reward_ax.set_ylabel("Reward")
            self.reward_ax.set_title("Rewards over time")
            self.reward_ax.legend()
        else:
            self.reward_ax.text(0.5, 0.5, "Noch keine Rewards", ha='center', va='center', transform=self.reward_ax.transAxes, fontsize=10, color='gray')
        self.reward_fig.tight_layout()
        self.reward_canvas.draw_idle()
        self.reward_canvas.flush_events()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
