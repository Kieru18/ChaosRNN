import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from DoublePendulumPredictor import DoublePendulumPredictor

MASS_1 = 1.0
MASS_2 = 1.0
LENGTH_1 = 1.0
LENGTH_2 = 1.0
GRAVITY  = 9.81
THETA_1 = np.pi * 0.66
THETA_2 = np.pi * 1.1412
Z_1 = 1
Z_2 = 1

def showcase(y_test, y_pred, theta_num):
    plt.figure(figsize=(20, 6))
    plt.plot(y_test[:, theta_num], label=f'Theta{theta_num + 1} - rzeczywiste', alpha=0.5, linewidth=2.5)
    plt.plot(y_pred[:, theta_num], label=f'Theta{theta_num + 1} - predykcja', linestyle='dashed', alpha=0.7) 
    plt.title('Porównanie rzeczywistych i przewidywanych trajektorii')
    plt.xlabel('Czas')
    plt.ylabel(f'Kąt Theta{theta_num + 1}')
    plt.xlim(0, 500)
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    predictor = DoublePendulumPredictor(LENGTH_1, LENGTH_2, MASS_1, MASS_2, GRAVITY, 
                                        THETA_1, THETA_2, Z_1, Z_2)
    
    y_test = predictor.generate_solution()
    y_pred = predictor.predict()
    
    start_index = 0
    y_test_sample = y_test[start_index+10:start_index + 510]
    y_pred_sample = y_pred[start_index:start_index + 500]
    
    showcase(y_test_sample, y_pred_sample, 0)
    showcase(y_test_sample, y_pred_sample, 1)
