import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import SimpleRNN, Dense
from scipy.integrate import solve_ivp
import os

NUM_TRAJECTORIES = 120
T_SPAN = (0, 10)
T_EVAL = np.linspace(0, 10, 400)
SEQ_LEN = 10
PATH = 'backend/models/trained_modelLSTM_50.keras'
MODEL = 'LSTM'

class DoublePendulumPredictor:
    def __init__(self, length_1, length_2, mass_1, mass_2, gravity, theta_1, theta_2, velocity_1, velocity_2, model_path=PATH):
        self.model_path = model_path
        self.length_1 = length_1
        self.length_2 = length_2
        self.mass_1 = mass_1
        self.mass_2 = mass_2
        self.gravity = gravity
        self.theta_1 = theta_1
        self.theta_2 = theta_2
        self.velocity_1 = velocity_1
        self.velocity_2 = velocity_2
        self.sequence_length = SEQ_LEN
        self.load_model()
        
    def load_model(self):
        if not os.path.exists(os.path.join(os.getcwd(), self.model_path)):
            print(f"Ścieżka do bieżącego katalogu: {os.getcwd()}")
            print(f"Ścieżka do modelu: {self.model_path}")
            print(f"Ścieżka : {os.path.join(os.getcwd(), self.model_path)}")
            print(f"Model nie istnieje. Tworzenie nowego modelu.")
            self.compose_model()
        
        self.model = load_model(self.model_path)
        print("Model wczytany.")
        
    def compose_model(self, units=64, epochs=50, batch_size=32):
        x_train, y_train = self.get_training_set()
        match MODEL:
            case 'LSTM':
                model = self.get_LSTM(x_train.shape[2], y_train.shape[1], units)
                model.compile(optimizer='adam', loss=tf.keras.losses.Huber(delta=1.0), metrics=['mae'])
            case 'SRNN':
                model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=['mae'])
                model = self.get_simpleRNN(x_train.shape[2], y_train.shape[1], units)
            case _:
                model = self.get_simpleRNN(x_train.shape[2], y_train.shape[1], units)
                model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=['mae'])
       
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
        model.save(self.model_path)
        self.model = model
    
    def get_training_set(self):
        trajectories = []
        for i in range(NUM_TRAJECTORIES):
            pendulum_position = [np.pi + np.random.uniform(-1.0, 1.0), 0.0, np.pi + np.random.uniform(-1.0, 1.0), 0.0]
                
            solution = solve_ivp(fun=self.pendulum_equation, 
                        t_span=T_SPAN, 
                        y0=pendulum_position,
                        t_eval=T_EVAL,
                        args=(self.mass_1, self.mass_2, self.length_1, self.length_2, self.gravity))
            trajectories.append(np.column_stack((solution.y[0], solution.y[1], solution.y[2], solution.y[3])))

        trajectories = np.array(trajectories)
        X, y = [], []
        for traj in trajectories:
            for i in range(len(traj) - self.sequence_length):
                X.append(traj[i:i + self.sequence_length])
                y.append(traj[i + self.sequence_length])
                
        return np.array(X), np.array(y)

    def pendulum_equation(self, t, pendulum_position, mass_1, mass_2, length_1, lenght_2, gravity):
        theta_1, velocity_1, theta_2, velocity_2 = pendulum_position
        c, s = np.cos(theta_1 - theta_2), np.sin(theta_1 - theta_2)

        theta1_dot = velocity_1
        z1_dot = (mass_2 * gravity * np.sin(theta_2) * c - mass_2 * s * (length_1 * velocity_1**2 * c + lenght_2 * velocity_2**2) -
                (mass_1 + mass_2) * gravity * np.sin(theta_1)) / length_1 / (mass_1 + mass_2 * s**2)
        theta2_dot = velocity_2
        z2_dot = ((mass_1 + mass_2) * (length_1 * velocity_1**2 * s - gravity * np.sin(theta_2) + gravity * np.sin(theta_1) * c) +
                mass_2 * lenght_2 * velocity_2**2 * s * c) / lenght_2 / (mass_1 + mass_2 * s**2)
        return [theta1_dot, z1_dot, theta2_dot, z2_dot]
    
    def generate_solution(self):
        pendulum_position = [self.theta_1, self.velocity_1, self.theta_2, self.velocity_2]
        solution = solve_ivp(fun=self.pendulum_equation, 
                            t_span=T_SPAN, 
                            y0=pendulum_position,
                            t_eval=T_EVAL,
                            args=(self.mass_1, self.mass_2, self.length_1, self.length_2, self.gravity))
        sliced_solution = np.column_stack((solution.y[0], solution.y[1], solution.y[2], solution.y[3]))
        return sliced_solution
    
    def predict(self, predict_path=None):

        if predict_path is None:
            predict_path = self.generate_solution()
        
        X, y = [], []
        for i in range(len(predict_path) - self.sequence_length):
            X.append(predict_path[i:i + self.sequence_length])
            y.append(predict_path[i + self.sequence_length])

        X = np.array(X)
        return self.model.predict(X)
        
    def get_simpleRNN(self, input_shape, output_shape, units):
        return tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.sequence_length, input_shape)),
            tf.keras.layers.SimpleRNN(units, activation='tanh', return_sequences=True),
            tf.keras.layers.SimpleRNN(units, activation='tanh'),
            tf.keras.layers.Dense(output_shape)
        ])
    
    def get_LSTM(self, input_shape, output_shape, units):
        return tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.sequence_length, input_shape)),
            tf.keras.layers.LSTM(units, activation='tanh', return_sequences=True),
            tf.keras.layers.LSTM(units, activation='tanh'),
            tf.keras.layers.Dense(output_shape)
        ])
    