import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import cv2
import threading
from scipy.optimize import minimize

class ImageRecoveryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Recovery App")

        # Schimbarea aspectului ferestrei
        self.root.configure(bg="blue")  # Setarea culorii de fundal a ferestrei

        self.n = 70
        self.A_bar = self.load_image("pisica.jpg")

        # Buttons
        self.run_cvx_button = tk.Button(root, text="Run CVX", command=self.run_cvx, bg="yellow")  # Setarea culorii de fundal pentru butoane
        self.run_cvx_button.pack()

        self.run_accelerated_gradient_button = tk.Button(root, text="Run Accelerated Gradient", command=self.run_accelerated_gradient, bg="yellow")  # Setarea culorii de fundal pentru butoane
        self.run_accelerated_gradient_button.pack()

        self.run_sgd_button = tk.Button(root, text="Run SGD", command=self.run_sgd, bg="yellow")  # Setarea culorii de fundal pentru butoane
        self.run_sgd_button.pack()

        # Canvas for image display
        self.fig = Figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack()

    def load_image(self, path):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (self.n, self.n))
        return image / 255.0

    def show_image(self, image, title, scale_factor=1.0):
        resized_image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
        self.ax.clear()
        self.ax.imshow(resized_image, cmap='gray')
        self.ax.set_title(title)
        self.canvas.draw()

    def run_cvx(self):
        threading.Thread(target=self.execute_cvx).start()

    def execute_cvx(self):
        try:
            B = self.cvx_method(self.A_bar)
            self.show_image(B, 'Image Recovered with CVX', scale_factor=0.5)  # Example scale_factor set to 0.5
        except Exception as e:
            messagebox.showerror("Error", str(e))
    def run_accelerated_gradient(self):
        threading.Thread(target=self.execute_accelerated_gradient).start()

    def execute_accelerated_gradient(self):
        try:
            X_new = self.accelerated_gradient_method(self.A_bar)
            self.show_image(X_new, 'Image Recovered with Accelerated Gradient', scale_factor=0.5)  # Example scale_factor set to 0.5
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run_sgd(self):
        threading.Thread(target=self.execute_sgd).start()

    def execute_sgd(self):
        try:
            X_sgd = self.sgd_method(self.A_bar)
            self.show_image(X_sgd, 'Image Recovered with SGD', scale_factor=0.5)  # Example scale_factor set to 0.5
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def cvx_method(self, A_bar):
        nrintraricunoscute = 3000
        rPerm = np.random.permutation(self.n*self.n)
        omega = np.sort(rPerm[:nrintraricunoscute])
        A = np.full((self.n, self.n), np.nan)
        A.flat[omega] = A_bar.flat[omega]

        def objective(B_flat):
            B = B_flat.reshape((self.n, self.n))
            return np.linalg.norm(B, ord='nuc')

        def constraint(B_flat):
            B = B_flat.reshape((self.n, self.n))
            return B.flat[omega] - A.flat[omega]

        B0 = np.random.randn(self.n, self.n).flatten()
        cons = {'type': 'eq', 'fun': constraint}
        result = minimize(objective, B0, constraints=cons, method='SLSQP', options={'maxiter': 1000})
        B = result.x.reshape((self.n, self.n))
        return B

    def accelerated_gradient_method(self, A_bar):
        nrintraricunoscute = 3000
        rPerm = np.random.permutation(self.n*self.n)
        omega = np.sort(rPerm[:nrintraricunoscute])
        A = np.full((self.n, self.n), np.nan)
        A.flat[omega] = A_bar.flat[omega]

        maxiter = 1000
        eps = 1e-3
        alpha = 0.1
        beta = 0.9

        A_initial = np.random.randn(self.n, self.n)
        A_initial.flat[omega] = A.flat[omega]
        Y = A_initial
        X_old = A_initial
        oprire = 1
        iter = 0

        while oprire >= eps and iter < maxiter:
            grad = Y - A_bar
            X_new = Y - alpha * grad
            X_new.flat[omega] = A.flat[omega]
            Y = X_new + beta * (X_new - X_old)
            oprire = np.linalg.norm(X_new - X_old)
            X_old = X_new
            iter += 1

        return X_new

    def sgd_method(self, A_bar):
        nrintraricunoscute = 3000
        rPerm = np.random.permutation(self.n*self.n)
        omega = np.sort(rPerm[:nrintraricunoscute])
        A = np.full((self.n, self.n), np.nan)
        A.flat[omega] = A_bar.flat[omega]

        alpha_sgd = 0.1
        maxiter_sgd = 1000
        eps_sgd = 1e-3
        batch_size = 200
        X_sgd = np.random.randn(self.n, self.n)
        X_sgd.flat[omega] = A.flat[omega]

        oprire_sgd = 1
        iter_sgd = 0

        while oprire_sgd >= eps_sgd and iter_sgd < maxiter_sgd:
            idx = np.random.choice(omega, batch_size, replace=False)
            xi_batch = A_bar.flat[idx]
            grad_sgd = X_sgd - np.mean(xi_batch)
            X_new_sgd = X_sgd - alpha_sgd * grad_sgd
            X_new_sgd.flat[omega] = A.flat[omega]
            oprire_sgd = np.linalg.norm(X_new_sgd - X_sgd)
            X_sgd = X_new_sgd
            iter_sgd += 1

        return X_sgd

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageRecoveryApp(root)
    root.mainloop()

