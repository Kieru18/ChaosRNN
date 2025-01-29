import numpy as np
from scipy.integrate import solve_ivp
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.slider import Slider
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.graphics import Line, Ellipse, Color
from kivy.clock import Clock
from kivy.core.window import Window

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
        self.draw_pendulum()  # Draw pendulum on start

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
            # Set the color to black for all pendulum elements
            Color(0, 0, 0, 1)  # Set color to black

            # Calculate the pendulum positions
            x1 = self.width / 2 + self.L1 * 100 * np.sin(self.theta1)
            y1 = self.height / 2 - self.L1 * 100 * np.cos(self.theta1)
            x2 = x1 + self.L2 * 100 * np.sin(self.theta2)
            y2 = y1 - self.L2 * 100 * np.cos(self.theta2)

            # Draw the pendulum lines (in black)
            Line(points=[self.width / 2, self.height / 2, x1, y1], width=2)
            Line(points=[x1, y1, x2, y2], width=2)

            # Draw the pendulum masses (in black)
            Ellipse(pos=(x1 - 10, y1 - 10), size=(20, 20))
            Ellipse(pos=(x2 - 10, y2 - 10), size=(20, 20))

class PendulumApp(App):
    def build(self):
        # Main layout (vertical)
        main_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        # Pendulum simulation widget
        self.simulation = PendulumSimulation(size_hint=(1, 0.7))
        main_layout.add_widget(self.simulation)

        # Control panel (vertical)
        control_panel = BoxLayout(orientation='vertical', size_hint=(1, 0.3), spacing=10)

        # Masses row
        masses_row = BoxLayout(orientation='horizontal', spacing=10)
        masses_row.add_widget(self.create_input_group('Mass 1', 'm1'))
        masses_row.add_widget(self.create_input_group('Mass 2', 'm2'))
        control_panel.add_widget(masses_row)

        # Lengths row
        lengths_row = BoxLayout(orientation='horizontal', spacing=10)
        lengths_row.add_widget(self.create_input_group('Length 1', 'L1'))
        lengths_row.add_widget(self.create_input_group('Length 2', 'L2'))
        control_panel.add_widget(lengths_row)

        # Gravity and start button
        gravity_group = self.create_input_group('Gravity', 'g')
        control_panel.add_widget(gravity_group)

        start_button = Button(text='Start Simulation', size_hint=(1, None), height=50)
        start_button.bind(on_press=self.start_simulation)
        control_panel.add_widget(start_button)

        # Add control panel to main layout
        main_layout.add_widget(control_panel)

        return main_layout

    def create_input_group(self, label_text, parameter):
        """Helper function to create a label, input field, and slider group."""
        group = BoxLayout(orientation='vertical', spacing=5)

        # Label
        label = Label(text=label_text, color=(0, 0, 0, 1), size_hint=(1, None), height=30)
        group.add_widget(label)

        # Text input
        input_field = TextInput(text='1.0', multiline=False, size_hint=(1, None), height=30)
        setattr(self, f'{parameter}_input', input_field)
        group.add_widget(input_field)

        # Slider
        slider = Slider(min=0.1, max=10, value=1.0, size_hint=(1, None), height=30)
        slider.bind(value=lambda instance, value: self.on_parameter_change(parameter, value))
        setattr(self, f'{parameter}_slider', slider)
        group.add_widget(slider)

        return group

    def on_parameter_change(self, parameter, value):
        """Update the simulation parameter when the slider changes."""
        setattr(self.simulation, parameter, value)
        getattr(self, f'{parameter}_input').text = str(value)

    def start_simulation(self, instance):
        self.simulation.start_simulation()

if __name__ == '__main__':
    Window.clearcolor = (1, 1, 1, 1)  # Set background color to white
    PendulumApp().run()