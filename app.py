import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QLineEdit, QPushButton
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp


class DoublePendulumSimulation(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Double Pendulum Simulation")
        self.setGeometry(100, 100, 1200, 600)

        self.length1 = 1.0
        self.length2 = 1.0
        self.mass1 = 1.0
        self.mass2 = 1.0
        self.gravity = 9.81

        self.initUI()

    def initUI(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)

        control_panel = self.createControlPanel()
        main_layout.addLayout(control_panel)

        self.canvas = FigureCanvas(Figure(figsize=(5, 4)))
        main_layout.addWidget(self.canvas)

        self.ax = self.canvas.figure.add_subplot(111)
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_title("Double Pendulum Simulation")

        self.setCentralWidget(main_widget)

    def createControlPanel(self):
        control_layout = QVBoxLayout()

        control_layout.addWidget(QLabel("Length 1"))
        self.length1_slider = QSlider(Qt.Horizontal)
        self.length1_slider.setMinimum(1)
        self.length1_slider.setMaximum(200)
        self.length1_slider.setValue(100)
        self.length1_slider.valueChanged.connect(self.updateLength1)
        control_layout.addWidget(self.length1_slider)

        self.length1_input = QLineEdit("1.0")
        self.length1_input.returnPressed.connect(self.updateLength1FromInput)
        control_layout.addWidget(self.length1_input)

        control_layout.addWidget(QLabel("Length 2"))
        self.length2_slider = QSlider(Qt.Horizontal)
        self.length2_slider.setMinimum(1)
        self.length2_slider.setMaximum(200)
        self.length2_slider.setValue(100)
        self.length2_slider.valueChanged.connect(self.updateLength2)
        control_layout.addWidget(self.length2_slider)

        self.length2_input = QLineEdit("1.0")
        self.length2_input.returnPressed.connect(self.updateLength2FromInput)
        control_layout.addWidget(self.length2_input)

        control_layout.addWidget(QLabel("Mass 1"))
        self.mass1_slider = QSlider(Qt.Horizontal)
        self.mass1_slider.setMinimum(1)
        self.mass1_slider.setMaximum(200)
        self.mass1_slider.setValue(100)
        self.mass1_slider.valueChanged.connect(self.updateMass1)
        control_layout.addWidget(self.mass1_slider)

        self.mass1_input = QLineEdit("1.0")
        self.mass1_input.returnPressed.connect(self.updateMass1FromInput)
        control_layout.addWidget(self.mass1_input)

        control_layout.addWidget(QLabel("Mass 2"))
        self.mass2_slider = QSlider(Qt.Horizontal)
        self.mass2_slider.setMinimum(1)
        self.mass2_slider.setMaximum(200)
        self.mass2_slider.setValue(100)
        self.mass2_slider.valueChanged.connect(self.updateMass2)
        control_layout.addWidget(self.mass2_slider)

        self.mass2_input = QLineEdit("1.0")
        self.mass2_input.returnPressed.connect(self.updateMass2FromInput)
        control_layout.addWidget(self.mass2_input)

        control_layout.addWidget(QLabel("Gravity"))
        self.gravity_slider = QSlider(Qt.Horizontal)
        self.gravity_slider.setMinimum(1)
        self.gravity_slider.setMaximum(200)
        self.gravity_slider.setValue(98)
        self.gravity_slider.valueChanged.connect(self.updateGravity)
        control_layout.addWidget(self.gravity_slider)

        self.gravity_input = QLineEdit("9.81")
        self.gravity_input.returnPressed.connect(self.updateGravityFromInput)
        control_layout.addWidget(self.gravity_input)

        self.start_button = QPushButton("Start Simulation")
        self.start_button.clicked.connect(self.startSimulation)
        control_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Simulation")
        self.stop_button.clicked.connect(self.stopSimulation)
        control_layout.addWidget(self.stop_button)

        return control_layout

    def updateLength1(self):
        self.length1 = self.length1_slider.value() / 100.0
        self.length1_input.setText(str(self.length1))

    def updateLength1FromInput(self):
        self.length1 = float(self.length1_input.text())
        self.length1_slider.setValue(int(self.length1 * 100))

    def updateLength2(self):
        self.length2 = self.length2_slider.value() / 100.0
        self.length2_input.setText(str(self.length2))

    def updateLength2FromInput(self):
        self.length2 = float(self.length2_input.text())
        self.length2_slider.setValue(int(self.length2 * 100))

    def updateMass1(self):
        self.mass1 = self.mass1_slider.value() / 100.0
        self.mass1_input.setText(str(self.mass1))

    def updateMass1FromInput(self):
        self.mass1 = float(self.mass1_input.text())
        self.mass1_slider.setValue(int(self.mass1 * 100))

    def updateMass2(self):
        self.mass2 = self.mass2_slider.value() / 100.0
        self.mass2_input.setText(str(self.mass2))

    def updateMass2FromInput(self):
        self.mass2 = float(self.mass2_input.text())
        self.mass2_slider.setValue(int(self.mass2 * 100))

    def updateGravity(self):
        self.gravity = self.gravity_slider.value() / 10.0
        self.gravity_input.setText(str(self.gravity))

    def updateGravityFromInput(self):
        self.gravity = float(self.gravity_input.text())
        self.gravity_slider.setValue(int(self.gravity * 10))

    def startSimulation(self):
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.ax.clear()
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_title("Double Pendulum Simulation")

        y0 = [np.pi / 2, 0, np.pi / 2, 0]
        t_span = (0, 10)
        t_eval = np.linspace(0, 10, 400)

        def equations(t, y):
            theta1, z1, theta2, z2 = y
            c, s = np.cos(theta1 - theta2), np.sin(theta1 - theta2)

            theta1_dot = z1
            z1_dot = (self.mass2 * self.gravity * np.sin(theta2) * c - self.mass2 * s * (self.length1 * z1**2 * c + self.length2 * z2**2) -
                      (self.mass1 + self.mass2) * self.gravity * np.sin(theta1)) / self.length1 / (self.mass1 + self.mass2 * s**2)
            theta2_dot = z2
            z2_dot = ((self.mass1 + self.mass2) * (self.length1 * z1**2 * s - self.gravity * np.sin(theta2) + self.gravity * np.sin(theta1) * c) +
                      self.mass2 * self.length2 * z2**2 * s * c) / self.length2 / (self.mass1 + self.mass2 * s**2)
            return [theta1_dot, z1_dot, theta2_dot, z2_dot]

        sol = solve_ivp(equations, t_span, y0, t_eval=t_eval)

        theta1, theta2 = sol.y[0], sol.y[2]
        x1 = self.length1 * np.sin(theta1)
        y1 = -self.length1 * np.cos(theta1)
        x2 = x1 + self.length2 * np.sin(theta2)
        y2 = y1 - self.length2 * np.cos(theta2)

        self.line, = self.ax.plot([], [], 'o-', lw=2)
        self.tip_line, = self.ax.plot([], [], 'r-', lw=1)
        self.tip_x, self.tip_y = [], []

        def update(frame):
            self.line.set_data([0, x1[frame], x2[frame]], [0, y1[frame], y2[frame]])
            self.tip_x.append(x2[frame])
            self.tip_y.append(y2[frame])
            self.tip_line.set_data(self.tip_x, self.tip_y)
            return self.line, self.tip_line

        self.anim = FuncAnimation(self.canvas.figure, update, frames=len(t_eval), interval=20, blit=True)
        self.canvas.draw()

    def stopSimulation(self):
        if hasattr(self, 'anim'):
            self.anim.event_source.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.ax.clear()
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_title("Double Pendulum Simulation")
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DoublePendulumSimulation()
    window.show()
    sys.exit(app.exec_())
