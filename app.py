import numpy as np
from scipy.integrate import solve_ivp
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.slider import Slider
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.graphics import Line, Ellipse
from kivy.clock import Clock

# Pendulum simulation code
def equations(t, y, m1, m2, L1, L2, g):
    theta1, z1, theta2, z2 = y
    c, s = np.cos(theta1 - theta2), np.sin(theta1 - theta2)

    theta1_dot = z1
    z1_dot = (m2 * g * np.sin(theta2) * c - m2 * s * (L1 * z1**2 * c + L2 * z2**2) -
              (m1 + m2) * g * np.sin(theta1)) / L1 / (m1 + m2 * s**2)
    theta2_dot = z2
    z2_dot = ((m1 + m2) * (L1 * z1**2 * s - g * np.sin(theta2) + g * np.sin(theta1) * c) +
              m2 * L2 * z2**2 * s * c) / L2 / (m1 + m2 * s**2)
    return [theta1_dot, z1_dot, theta2_dot, z2_dot]

class PendulumSimulation(Widget):
    def __init__(self, **kwargs):
        super(PendulumSimulation, self).__init__(**kwargs)
        self.m1 = 1.0
        self.m2 = 1.0
        self.L1 = 1.0
        self.L2 = 1.0
        self.g = 9.81
        self.theta1 = np.pi
        self.theta2 = np.pi
        self.sol = None
        self.simulating = False

    def start_simulation(self):
        y0 = [self.theta1, 0, self.theta2, 0]
        t_span = (0, 10)
        t_eval = np.linspace(0, 10, 1000)
        self.sol = solve_ivp(equations, t_span, y0, args=(self.m1, self.m2, self.L1, self.L2, self.g), t_eval=t_eval)
        self.simulating = True
        Clock.schedule_interval(self.update, 1.0 / 60.0)

    def update(self, dt):
        if self.simulating and self.sol is not None:
            self.theta1, self.theta2 = self.sol.y[0][-1], self.sol.y[2][-1]
            self.draw_pendulum()

    def draw_pendulum(self):
        self.canvas.clear()
        with self.canvas:
            x1 = 300 + self.L1 * 100 * np.sin(self.theta1)
            y1 = 300 - self.L1 * 100 * np.cos(self.theta1)
            x2 = x1 + self.L2 * 100 * np.sin(self.theta2)
            y2 = y1 - self.L2 * 100 * np.cos(self.theta2)
            Line(points=[300, 300, x1, y1], width=2)
            Line(points=[x1, y1, x2, y2], width=2)
            Ellipse(pos=(x1 - 10, y1 - 10), size=(20, 20))
            Ellipse(pos=(x2 - 10, y2 - 10), size=(20, 20))

class PendulumApp(App):
    def build(self):
        layout = BoxLayout(orientation='vertical')

        self.simulation = PendulumSimulation()
        layout.add_widget(self.simulation)

        self.m1_input = TextInput(text='1.0', multiline=False)
        self.m1_slider = Slider(min=0.1, max=10, value=1.0)
        self.m1_slider.bind(value=self.on_m1_change)
        layout.add_widget(Label(text='Mass 1'))
        layout.add_widget(self.m1_input)
        layout.add_widget(self.m1_slider)

        self.m2_input = TextInput(text='1.0', multiline=False)
        self.m2_slider = Slider(min=0.1, max=10, value=1.0)
        self.m2_slider.bind(value=self.on_m2_change)
        layout.add_widget(Label(text='Mass 2'))
        layout.add_widget(self.m2_input)
        layout.add_widget(self.m2_slider)

        self.L1_input = TextInput(text='1.0', multiline=False)
        self.L1_slider = Slider(min=0.1, max=10, value=1.0)
        self.L1_slider.bind(value=self.on_L1_change)
        layout.add_widget(Label(text='Length 1'))
        layout.add_widget(self.L1_input)
        layout.add_widget(self.L1_slider)

        self.L2_input = TextInput(text='1.0', multiline=False)
        self.L2_slider = Slider(min=0.1, max=10, value=1.0)
        self.L2_slider.bind(value=self.on_L2_change)
        layout.add_widget(Label(text='Length 2'))
        layout.add_widget(self.L2_input)
        layout.add_widget(self.L2_slider)

        self.g_input = TextInput(text='9.81', multiline=False)
        self.g_slider = Slider(min=1, max=20, value=9.81)
        self.g_slider.bind(value=self.on_g_change)
        layout.add_widget(Label(text='Gravity'))
        layout.add_widget(self.g_input)
        layout.add_widget(self.g_slider)

        start_button = Button(text='Start Simulation')
        start_button.bind(on_press=self.start_simulation)
        layout.add_widget(start_button)

        return layout

    def on_m1_change(self, instance, value):
        self.simulation.m1 = value
        self.m1_input.text = str(value)

    def on_m2_change(self, instance, value):
        self.simulation.m2 = value
        self.m2_input.text = str(value)

    def on_L1_change(self, instance, value):
        self.simulation.L1 = value
        self.L1_input.text = str(value)

    def on_L2_change(self, instance, value):
        self.simulation.L2 = value
        self.L2_input.text = str(value)

    def on_g_change(self, instance, value):
        self.simulation.g = value
        self.g_input.text = str(value)

    def start_simulation(self, instance):
        self.simulation.start_simulation()

if __name__ == '__main__':
    PendulumApp().run()
