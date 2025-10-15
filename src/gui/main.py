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
    QDoubleSpinBox,  # <--- importiert für Lernrate
    QScrollArea,
    QToolBar,
    QMenu,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QComboBox as QComboBoxWidget,
    QCheckBox,
    QStackedWidget,  # <--- hinzufügen
    QListWidget, QInputDialog,  # für Interface-Liste und Dialog
    QTextEdit,  # für Code-Ansicht im Umweltfenster
)
from PyQt6.QtGui import QColor, QPixmap, QPainter, QImage, QFont, QAction, QIcon
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal
import numpy as np
import markdown  # für Markdown-zu-HTML-Konvertierung
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.types.agent import Trainingszustand

from src.umwelt import Umwelt
from src.agent import Agent
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt


class TrainingThread(QThread):
    finished = pyqtSignal()
    def __init__(self, agent, steps=1000, learning_rate=0.0003):
        super().__init__()
        self.agent = agent
        self.steps = steps
        self.learning_rate = learning_rate
    def run(self):
        try:
            from stable_baselines3 import PPO
            model = PPO("MlpPolicy", self.agent, verbose=1, learning_rate=self.learning_rate)
            model.learn(total_timesteps=self.steps, progress_bar=False)
        except Exception as e:
            import traceback
            print("Fehler im TrainingThread:", e)
            traceback.print_exc()
        self.finished.emit()


class SettingsDialog(QDialog):
    def __init__(self, parent=None, show_plot_separately=False):
        super().__init__(parent)
        self.setWindowTitle("Einstellungen")
        layout = QFormLayout(self)
        # Spracheinstellung
        self.language_combo = QComboBoxWidget()
        self.language_combo.addItems(["Deutsch", "Englisch"])
        layout.addRow("Sprache:", self.language_combo)
        # Theme (Hell/Dunkel)
        self.theme_combo = QComboBoxWidget()
        self.theme_combo.addItems(["Hell", "Dunkel"])
        layout.addRow("Theme:", self.theme_combo)
        # Plot-Option
        self.plot_separate_checkbox = QCheckBox("Reward-Plot in separatem Fenster anzeigen")
        self.plot_separate_checkbox.setChecked(show_plot_separately)
        layout.addRow(self.plot_separate_checkbox)
        # Buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)
    def get_settings(self):
        return {
            "language": self.language_combo.currentText(),
            "theme": self.theme_combo.currentText(),
            "plot_separate": self.plot_separate_checkbox.isChecked(),
        }


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # --- Plot-Optionen und Reward-Plot-Fenster VOR jeglicher Nutzung initialisieren ---
        self.show_plot_separately = False
        self.reward_window = None

        # --- Simulation/Step-Status früh initialisieren, damit update_step_label funktioniert ---
        # --- Agenten-Statusbereich als Container ---
        self.agent_status_box = QGroupBox("Agentenstatus")
        self.agent_status_layout = QVBoxLayout()
        # Trainingszustand-Statuspunkt
        self.training_status_label = QLabel()
        self.training_status_label.setFixedHeight(24)
        self.agent_status_layout.addWidget(self.training_status_label)
        # --- Aktueller Step ---
        # --- Initialisiere sim_steps und verwandte Attribute VOR update_step_label ---
        self.sim_steps = 0
        self.max_steps = 10
        self.sim_running = False
        self.step_label = QLabel()
        self.step_label.setFixedHeight(24)
        self.agent_status_layout.addWidget(self.step_label)
        self.update_step_label()
        # Gedächtnis-Visualisierung als scrollbarer Bereich
        self.memory_label = QLabel()
        self.memory_label.setWordWrap(True)
        self.memory_scroll = QScrollArea()
        self.memory_scroll.setWidgetResizable(True)
        self.memory_scroll.setWidget(self.memory_label)
        self.memory_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.memory_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.agent_status_layout.addWidget(self.memory_scroll)
        # Wahrnehmungsdissonanzmatrix-Anzeige
        self.dissonanz_label = QLabel()
        self.dissonanz_label.setWordWrap(True)
        self.dissonanz_label.setMinimumHeight(80)
        self.agent_status_layout.addWidget(QLabel("Wahrnehmungsdissonanzmatrix:"))
        self.agent_status_layout.addWidget(self.dissonanz_label)
        self.agent_status_box.setLayout(self.agent_status_layout)
        # --- Ende Agentenstatus-Container ---

        # --- Plots & Statistiken Container ---
        self.plots_stats_box = QGroupBox("Plots & Statistiken")
        self.plots_stats_layout = QVBoxLayout()
        # Matplotlib-Plot für Rewards
        self.reward_fig, self.reward_ax = plt.subplots()
        self.reward_canvas = FigureCanvas(self.reward_fig)
        self.reward_canvas.setFixedHeight(160)
        self.plots_stats_layout.addWidget(self.reward_canvas)
        # Hier können weitere Statistiken/Plots ergänzt werden
        self.plots_stats_box.setLayout(self.plots_stats_layout)
        # --- Ende Plots & Statistiken Container ---

        # --- Agenten-Steuerungsbereich als Container (unverändert) ---
        self.agent_controls_box = QGroupBox("Steuerung & Einstellungen")
        self.agent_controls_layout = QVBoxLayout()
        from PyQt6.QtWidgets import QLineEdit
        # Neue horizontale Layout-Reihe für Inputs und Training-Button
        self.memory_control_layout = QHBoxLayout()
        self.memory_input = QLineEdit()
        self.memory_input.setText("5")
        self.memory_input.setFixedWidth(120)  # 25% der typischen Breite (bei ca. 480px Layout)
        self.memory_input.setPlaceholderText("Zellen")
        self.memory_input.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.memory_control_layout.addWidget(self.memory_input)
        # Lernrate-Input
        self.lr_label = QLabel("Lernrate:")
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(5)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setMinimum(0.00001)
        self.lr_spin.setMaximum(1.0)
        self.lr_spin.setValue(0.0003)  # Standardwert für PPO
        self.memory_control_layout.addWidget(self.lr_label)
        self.memory_control_layout.addWidget(self.lr_spin)
        # Max Steps Input
        self.max_steps_label = QLabel("max. Schritte:")
        self.max_steps_spin = QSpinBox()
        self.max_steps_spin.setMinimum(1)
        self.max_steps_spin.setMaximum(100000)
        self.max_steps_spin.setValue(10)
        self.memory_control_layout.addWidget(self.max_steps_label)
        self.memory_control_layout.addWidget(self.max_steps_spin)
        # Training starten Button ganz rechts
        self.train_toggle_button = QPushButton("Training starten")
        self.train_toggle_button.setCheckable(True)
        self.train_toggle_button.clicked.connect(self.toggle_training)
        self.memory_control_layout.addStretch(1)
        self.memory_control_layout.addWidget(self.train_toggle_button)
        self.agent_controls_layout.addLayout(self.memory_control_layout)
        self.agent_controls_box.setLayout(self.agent_controls_layout)
        # --- Ende Agenten-Steuerung ---

        # --- Hauptlayout initialisieren (Hinzufügen!) ---
        self.main_layout = QHBoxLayout()

        # Agentenbereich (vertikal): Steuerung, Status, Plots
        self.agent_box = QGroupBox("Agent")
        self.agent_layout = QVBoxLayout()
        self.agent_layout.addWidget(self.agent_controls_box)
        self.agent_layout.addWidget(self.agent_status_box)
        self.agent_layout.addWidget(self.plots_stats_box)
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

        # --- Interface-Liste und Hinzufügen-Button ---
        self.interface_list_label = QLabel("Verfügbare Interfaces:")
        self.umwelt_layout.addWidget(self.interface_list_label)
        self.interface_list = QListWidget()
        self.umwelt_layout.addWidget(self.interface_list)
        self.add_interface_button = QPushButton("Interface hinzufügen")
        self.add_interface_button.clicked.connect(self.add_interface)
        self.umwelt_layout.addWidget(self.add_interface_button)
        # Status-Verwaltung für Interfaces
        self.interfaces = []  # Liste von Dicts: {"name": ..., "active": ...}
        self.interface_list.itemClicked.connect(self.toggle_interface_status)
        # --- Ende Interface-Liste ---

        # Button zum Öffnen des Umwelt-Fensters
        self.open_umwelt_window_button = QPushButton("Umwelt-Fenster öffnen")
        self.open_umwelt_window_button.clicked.connect(self.show_umwelt_view)
        self.umwelt_layout.addWidget(self.open_umwelt_window_button)

        self.umwelt_box.setLayout(self.umwelt_layout)

        # Layouts zusammenführen
        self.main_layout.addWidget(self.agent_box)
        self.main_layout.addWidget(self.umwelt_box)

        self.main_widget = QWidget()
        self.main_widget.setLayout(self.main_layout)
        
        # --- Zusätzliche zentrale Views initialisieren (About, Settings, Umwelt) ---
        self.about_widget = QWidget()
        self.settings_widget = QWidget()
        self.umwelt_widget = QWidget()
        # --- About-Ansicht: Markdown laden und anzeigen ---
        about_layout = QVBoxLayout()
        self.about_text = QTextEdit()
        self.about_text.setReadOnly(True)
        about_md = self._read_markdown_file(os.path.join(os.path.dirname(__file__), '../../docs/about.md'))
        if about_md:
            about_html = markdown.markdown(about_md)
            self.about_text.setHtml(about_html)
        else:
            self.about_text.setText("Keine Info verfügbar.")
        about_layout.addWidget(self.about_text)
        self.about_widget.setLayout(about_layout)
        # --- Settings-Ansicht: einfache Settings anzeigen ---
        settings_layout = QFormLayout()
        self.language_combo = QComboBoxWidget()
        self.language_combo.addItems(["Deutsch", "Englisch"])
        settings_layout.addRow("Sprache:", self.language_combo)
        self.theme_combo = QComboBoxWidget()
        self.theme_combo.addItems(["Hell", "Dunkel"])
        settings_layout.addRow("Theme:", self.theme_combo)
        self.plot_separate_checkbox = QCheckBox("Reward-Plot in separatem Fenster anzeigen")
        settings_layout.addRow(self.plot_separate_checkbox)
        self.settings_widget.setLayout(settings_layout)
        # --- Ende zentrale Views ---

        # --- Toolbar und Navigation initialisieren ---
        self.toolbar = self.addToolBar("Navigation")
        # Training/Home
        self.training_action = QAction(QIcon(), "Training", self)
        self.training_action.triggered.connect(self.show_main_view)
        self.toolbar.addAction(self.training_action)
        # Umwelt
        self.umwelt_action = QAction(QIcon(), "Umwelt", self)
        self.umwelt_action.triggered.connect(self.show_umwelt_view)
        self.toolbar.addAction(self.umwelt_action)
        # Einstellungen
        self.settings_action = QAction(QIcon(), "Einstellungen", self)
        self.settings_action.triggered.connect(self.show_settings_view)
        self.toolbar.addAction(self.settings_action)
        # Info/About
        self.about_action = QAction(QIcon(), "Info", self)
        self.about_action.triggered.connect(self.show_about_view)
        self.toolbar.addAction(self.about_action)
        # Back-Action (wie gehabt, aber ans Ende)
        self.back_action = QAction("Zurück", self)
        self.back_action.setVisible(False)
        self.back_action.triggered.connect(self.show_main_view)
        self.toolbar.addAction(self.back_action)
        
        # --- Zentrales Stack-Widget für alle Views ---
        self.central_stack = QStackedWidget()
        self.central_stack.addWidget(self.main_widget)      # Index 0
        self.central_stack.addWidget(self.about_widget)     # Index 1
        self.central_stack.addWidget(self.settings_widget)  # Index 2
        self.central_stack.addWidget(self.umwelt_widget)    # Index 3
        self.setCentralWidget(self.central_stack)
        # --- Ende Stack-Widget ---

        # Agent initialisieren mit Gedächtnisgröße
        self.agent = Agent()
        # Gedächtnis-Input-Events erst jetzt verbinden
        self.memory_input.textChanged.connect(self._on_memory_input_changed)
        self.reset_memory()  # Initialwert übernehmen
        self.umwelt = self.agent.umwelt

        self.timer = QTimer()
        self.timer.timeout.connect(self.simulation_step)
        self.sim_steps = 0
        self.max_steps = 10
        self.sim_running = False


        # --- Layout und Felder für Umwelt-Ansicht initialisieren (jetzt direkt nach Widget-Erstellung!) ---
        self.umwelt_layout_view = QVBoxLayout()
        self.umwelt_info_label = QLabel()
        self.umwelt_info_label.setWordWrap(True)
        self.umwelt_code_view = QTextEdit()
        self.umwelt_code_view.setReadOnly(True)
        self.umwelt_code_view.setMinimumHeight(200)
        self.umwelt_layout_view.addWidget(self.umwelt_info_label)
        self.umwelt_layout_view.addWidget(self.umwelt_code_view)
        self.umwelt_widget.setLayout(self.umwelt_layout_view)

    def get_env_classes(self):
        env_dir = os.path.join(os.path.dirname(__file__), "../envs")
        env_files = [
            f
            for f in os.listdir(env_dir)
            if f.endswith(".py") and not f.startswith("__")
        ]
        envs = {}
        for file in env_files:
            modulename = f"src.envs.{file[:-3]}"
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
            neue_umwelt = Umwelt(env_class())
            self.agent.set_umwelt(neue_umwelt)
            self.umwelt = self.agent.umwelt
            # Gedächtnis nach Umweltwechsel neu initialisieren
            self.agent.gedächtnis = self.create_memory(self.memory_size_spin.value())
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
            self.max_steps = self.max_steps_spin.value()  # Wert aus UI übernehmen
            self.timer.start(500)
        else:
            self.sim_running = False
            self.sim_button.setText("Simulation starten")
            self.timer.stop()
            # Reset: Umwelt neu laden und Status zurücksetzen
            self.load_umwelt()

    def update_step_label(self):
        self.step_label.setText(f"Aktueller Step: {self.sim_steps}")

    def simulation_step(self):
        if self.sim_steps >= self.max_steps:
            self.timer.stop()
            self.sim_running = False
            self.sim_button.setText("Simulation starten")
            return
        # Beispielaktion: Zufällige Aktion für Agent
        aktion = [np.random.randint(self.agent.gedächtnis.getKapazität()), np.random.randint(self.umwelt.aktionsraum)]
        _, reward, _, _, _ = self.agent.step(aktion)
        self.label.setText(str(self.umwelt.zustand_welt))
        if hasattr(self.umwelt, "zustand"):
            self.update_status_point(self.umwelt.zustand.name if hasattr(self.umwelt.zustand, "name") else self.umwelt.zustand)
        self.sim_steps += 1
        self.update_step_label()
        self.update_memory_visualization()
        self.update_reward_plot()
        self.update_dissonanz_matrix()
        # --- NEU: Immer nach jedem Schritt updaten, auch bei Einzelschritt/Training ---
        QApplication.processEvents()

    def create_memory(self, size):
        try:
            from src.gedächtnis import Gedächtnis
            from src.erinnerung import Erinnerung
            import numpy as np
            return Gedächtnis(size)
        except Exception as e:
            print(f"Fehler beim Erstellen des Gedächtnisses: {e}")
            return None

    def reset_memory(self):
        try:
            size = int(self.memory_input.text())
            if size < 1:
                size = 1
        except Exception:
            size = 1
        self.memory_input.setText(str(size))
        self.agent.gedächtnis = self.create_memory(size)
        self.update_memory_visualization()

    def _on_memory_input_changed(self):
        # Automatisches Reset des Gedächtnisses bei Änderung
        self.reset_memory()

    def update_memory_visualization(self):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib import cm
        from PyQt6.QtGui import QPixmap
        import io
        from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget
        memory = self.agent.gedächtnis
        if memory is None:
            empty = QWidget()
            self.memory_scroll.setWidget(empty)
            return
        try:
            kap = memory.getKapazität()
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
                rows, cols = obs_shape[1], obs_shape[2]
            else:
                rows, cols = 1, 1
            schreibzeiger = getattr(memory, "schreibZeiger", None)
            # Finde min/max Wert aller Grids für Farbskala
            all_values = []
            for i in range(kap):
                try:
                    erinnerung = memory.speicher[i]
                    bild = erinnerung.getBild()
                    arr = np.array(bild)
                    if arr.ndim == 3:
                        arr = arr[0]
                    arr = arr.reshape((rows, cols))
                    all_values.extend(arr.flatten())
                except Exception:
                    continue
            if all_values:
                min_val = float(np.min(all_values))
                max_val = float(np.max(all_values))
            else:
                min_val, max_val = 0, 1
            # Neues Layout für Heatmaps
            heatmap_layout = QHBoxLayout()
            heatmap_layout.setSpacing(2)
            heatmap_layout.setContentsMargins(0,0,0,0)
            for i in range(kap):
                try:
                    erinnerung = memory.speicher[i]
                    bild = erinnerung.getBild()
                    arr = np.array(bild)
                    if arr.ndim == 3:
                        arr = arr[0]
                    arr = arr.reshape((rows, cols))
                    fig = Figure(figsize=(1.2, 1.2), dpi=66)
                    ax = fig.add_subplot(111)
                    im = ax.imshow(arr, cmap=cm.coolwarm, vmin=min_val, vmax=max_val, aspect='auto')
                    ax.axis('off')
                    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
                    buf.seek(0)
                    pixmap = QPixmap()
                    pixmap.loadFromData(buf.getvalue())
                    label = QLabel()
                    label.setPixmap(pixmap)
                    label.setFixedSize(80, 80)
                    if schreibzeiger == i:
                        label.setStyleSheet("border: 2px solid #3399ff; margin:0px 1px;")
                    else:
                        label.setStyleSheet("border: 1px solid #555; margin:0px 1px;")
                    heatmap_layout.addWidget(label)
                except Exception:
                    label = QLabel(f"Zelle {i}: Fehler")
                    heatmap_layout.addWidget(label)
            # Setze das neue Layout in den Scrollbereich
            container = QWidget()
            container.setLayout(heatmap_layout)
            self.memory_scroll.setWidget(container)
            self.memory_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
            self.memory_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        except Exception as e:
            empty = QWidget()
            self.memory_scroll.setWidget(empty)
            print(f"Fehler in update_memory_visualization: {e}")

    def update_dissonanz_matrix(self):
        matrix = getattr(self.agent, "wahrnehmungsdissonanzmatrix", None)
        if matrix is None:
            self.dissonanz_label.setText("-")
            return
        arr = np.array(matrix)
        # Bei 3D: nur erste Schicht anzeigen
        if arr.ndim == 3:
            arr = arr[0]
        # Farbskala: grau (0) bis rot (Abweichung von 0)
        max_abs = np.max(np.abs(arr)) if arr.size > 0 else 1
        def dissonanz_color(val):
            # Skala: dunkelgrau (0) zu rot (Abweichung von 0)
            if val == 0 or max_abs == 0:
                return "#222222"
            f = min(abs(val) / max_abs, 1.0)
            r = int(34 + (255-34)*f)
            g = int(34 + (51-34)*f)
            b = int(34 + (51-34)*f)
            return f"#{r:02x}{g:02x}{b:02x}"
        html = "<table style='border-collapse:collapse; font-family:monospace; font-size:12px; background:#222;'>"
        for row in arr:
            html += "<tr>"
            for val in row:
                color = dissonanz_color(val)
                html += f"<td style='border:1px solid #555; min-width:18px; text-align:center; padding:2px; background:{color}; color:#fff;'>{val:.2f}</td>"
            html += "</tr>"
        html += "</table>"
        self.dissonanz_label.setText(html)

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
        # Training mit aktueller Umwelt im Thread starten
        self.trainingszustand = Trainingszustand.AKTIV
        self.train_toggle_button.setText("Training läuft...")
        self.update_training_status_point(self.trainingszustand)
        # Lernrate aus UI übernehmen
        lr = self.lr_spin.value()
        self.training_thread = TrainingThread(self.agent, steps=1000, learning_rate=lr)
        self.training_thread.finished.connect(self.training_finished)
        self.training_thread.start()

    def training_finished(self):
        self.trainingszustand = Trainingszustand.FERTIG
        self.train_toggle_button.setText("Training starten")
        self.update_training_status_point(self.trainingszustand)
        self.update_reward_plot()
        self.update_dissonanz_matrix()
        self.update_memory_visualization()  # NEU: Gedächtnis nach Training updaten
        QApplication.processEvents()

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
        # Plot ggf. in separatem Fenster anzeigen
        if self.show_plot_separately:
            if self.reward_window is None:
                self.reward_window = QMainWindow(self)
                self.reward_window.setWindowTitle("Reward-Plot")
                self.reward_window.setCentralWidget(self.reward_canvas)
                self.reward_window.resize(500, 300)
            if not self.reward_window.isVisible():
                self.reward_window.show()
        else:
            # Plot wieder im Plots & Statistiken-Container anzeigen
            if self.reward_window is not None and self.reward_window.isVisible():
                self.reward_window.hide()
            if self.reward_canvas.parent() != self.plots_stats_layout:
                self.plots_stats_layout.addWidget(self.reward_canvas)

    def show_main_view(self):
        self.central_stack.setCurrentIndex(0)
        self.current_view = "main"
        self.back_action.setVisible(False)
        self.training_action.setChecked(True)
        self.umwelt_action.setChecked(False)
        self.settings_action.setChecked(False)
        self.about_action.setChecked(False)

    def show_about_view(self):
        self.central_stack.setCurrentIndex(1)
        self.current_view = "about"
        self.back_action.setVisible(True)
        self.training_action.setChecked(False)
        self.umwelt_action.setChecked(False)
        self.settings_action.setChecked(False)
        self.about_action.setChecked(True)

    def show_settings_view(self):
        self.central_stack.setCurrentIndex(2)
        self.current_view = "settings"
        self.back_action.setVisible(True)
        self.training_action.setChecked(False)
        self.umwelt_action.setChecked(False)
        self.settings_action.setChecked(True)
        self.about_action.setChecked(False)

    def show_umwelt_view(self):
        self.update_umwelt_info()
        self.central_stack.setCurrentIndex(3)
        self.current_view = "umwelt"
        self.back_action.setVisible(True)
        self.training_action.setChecked(False)
        self.umwelt_action.setChecked(True)
        self.settings_action.setChecked(False)
        self.about_action.setChecked(False)

    def update_umwelt_info(self):
        # Zeige die Python-Definitionsklasse der aktuellen Umwelt
        if self.umwelt is not None:
            # Das eigentliche Environment-Objekt steckt in self.umwelt.umwelt
            env_obj = getattr(self.umwelt, "umwelt", self.umwelt)
            klassentext = type(env_obj).__name__
            modulename = type(env_obj).__module__
            doc = getattr(env_obj, "__doc__", "")
            info = f"<b>Klasse:</b> {klassentext}<br><b>Modul:</b> {modulename}<br>"
            if doc:
                info += f"<pre style='font-size:12px'>{doc}</pre>"
            self.umwelt_info_label.setText(info)
            # Lade und zeige den Python-Code der Environment-Klasse aus src/envs
            try:
                # Ermittle den Dateipfad aus dem Modulnamen (z.B. src.envs.test -> src/envs/test.py)
                if modulename.startswith("src.envs."):
                    envfile = os.path.join(os.path.dirname(__file__), "..", "envs", modulename.split(".")[-1] + ".py")
                    envfile = os.path.abspath(envfile)
                else:
                    envfile = importlib.util.find_spec(modulename).origin
                with open(envfile, "r", encoding="utf-8") as f:
                    code = f.read()
                self.umwelt_code_view.setPlainText(code)
            except Exception as e:
                self.umwelt_code_view.setPlainText(f"Fehler beim Laden der Datei: {e}")
        else:
            self.umwelt_info_label.setText("Keine Umwelt geladen.")
            self.umwelt_code_view.setPlainText("")

    def open_settings_dialog(self):
        self.show_settings_view()

    def show_about_dialog(self):
        self.show_about_view()

    def add_interface(self):
        # Dialog zur Eingabe des Interface-Namens
        text, ok = QInputDialog.getText(self, "Interface hinzufügen", "Name des Interfaces (z.B. USB, API, ...):")
        if ok and text:
            # Prüfe auf Duplikate
            if any(i["name"] == text for i in self.interfaces):
                QMessageBox.warning(self, "Interface existiert", f"Das Interface '{text}' existiert bereits.")
                return
            self.interfaces.append({"name": text, "active": False})
            self.update_interface_list()

    def update_interface_list(self):
        self.interface_list.clear()
        for interface in self.interfaces:
            emoji = "🟢" if interface["active"] else "⚪️"
            # Name links, Emoji rechtsbündig
            item_text = f"{interface['name']}    {emoji}"
            self.interface_list.addItem(item_text)

    def toggle_interface_status(self, item):
        # Finde das Interface anhand des Namens (ohne Emoji)
        name = item.text().rsplit(" ", 1)[0].strip()
        for interface in self.interfaces:
            if interface["name"] == name:
                interface["active"] = not interface["active"]
                break
        self.update_interface_list()

    def _read_markdown_file(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return None


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Electric Columbus Desktop")
    app.setApplicationDisplayName("Electric Columbus GUI")
    # Setze das App-Icon für das Dock (macOS) und Taskleiste (Windows/Linux)
    icon_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../public/img/favicon.ico'))
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
