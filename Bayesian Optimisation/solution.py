import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process.kernels import *
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm

domain = np.array([[0, 5]])
SAFETY_THRESHOLD = 1.2
SEED = 0

""" Solution """


class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """

        # TODO: enter your code here
        self.dataset = []
        # self.gp_f = GaussianProcessRegressor(
        #     kernel=Matern(length_scale=0.5, nu=2.5) + WhiteKernel(noise_level=0.0225,noise_level_bounds="fixed"),
        #     alpha=0.5,
        #     optimizer=None,
        #     normalize_y=True) # GP model for f
        
        self.gp_f = GaussianProcessRegressor(
            kernel=ConstantKernel(0.5,"fixed")*Matern(0.5,length_scale_bounds="fixed",nu=2.5)+ WhiteKernel(noise_level=0.0225,noise_level_bounds="fixed"),
            optimizer=None,
            normalize_y=True) # GP model for f
        

        # self.gp_v = GaussianProcessRegressor(
        #     kernel=Matern(length_scale=0.5, nu=2.5)*ConstantKernel(constant_value=1.5) + WhiteKernel(noise_level=0.0001**2,noise_level_bounds="fixed"),
        #     alpha=1.414,
        #     optimizer=None,
        #     normalize_y=True) # GP model for v

        self.gp_v = GaussianProcessRegressor(
            kernel=ConstantKernel(1.414,"fixed")*Matern(0.5,length_scale_bounds="fixed",nu=2.5)+ WhiteKernel(noise_level=0.0001**2,noise_level_bounds="fixed"),
            optimizer=None,
            normalize_y=True) # GP model for v

        self.X_train = []
        self.f_train = []
        self.v_train = []

        self.t = 0 #Keeping track of time
        self.best_score = 0
        self.violation_counter = 0


    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """

        # TODO: enter your code here
        # In implementing this function, you may use optimize_acquisition_function() defined below.
        # if len(self.dataset) == 0:
        #     x = np.array([[np.random.uniform(domain[0,0], domain[0,1])]], dtype==object)
        # else:
        #     x = self.optimize_acquisition_function()

        return self.optimize_acquisition_function()


    def optimize_acquisition_function(self):
        """
        Optimizes the acquisition function.

        Returns
        -------
        x_opt: np.ndarray
            1 x domain.shape[0] array containing the point that maximize the acquisition function.
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * \
                 np.random.rand(domain.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=domain,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *domain[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        return np.atleast_2d(x_values[ind])

    def acquisition_function(self, x):
        """
        Compute the acquisition function.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f

        Returns
        ------
        af_value: float
            Value of the acquisition function at x
        """

        # TODO: enter your code here

        f_mean, f_std = self.gp_f.predict([x], return_std=True)
        v_mean, v_std = self.gp_v.predict([x], return_std=True)
        #We use upper confidence bound with a penalty if an optimistic estimate of the speed doesn't fulfill the constraints
        return f_mean[0][0] + 0.5*f_std[0] - 10*(v_mean[0][0] + v_std[0] +1.5 < 1.20015) + 10*min(v_mean[0][0] + v_std[0] +1.5-1.2, 0)

        # v_mean, v_std = self.gp_v.predict(x.reshape(1,-1), return_std=True)
        # v_prob = (1 - norm.cdf(0.0, v_mean, v_std))*100 # Proba of x satisfying constraint c^(x) <= 0
        
        # f_max = np.max(self.f_train) # Get current min fct value
        
        # f_mean, f_std = self.gp_f.predict(x.reshape(1,-1), return_std=True)
        # z = (f_mean - f_max) / f_std # Standardized
        # ei_x = f_std * (z * norm.cdf(z) + norm.pdf(z)) # Expected improvement for obs x
        # # af_x = ei_x * v_prob # Acquisition function
        # af_x = ei_x + v_prob # Acquisition function

        # return float(af_x)

    def add_data_point(self, x, f, v):
        """
        Add data points to the model.

        Parameters
        ----------
        x: np.ndarray
            Hyperparameters
        f: np.ndarray
            Model accuracy
        v: np.ndarray
            Model training speed
        """

        # TODO: enter your code here
        if v<1.2:
            self.violation_counter +=1
        self.X_train.append(x.flatten()[0])
        self.f_train.append(f.flatten()[0])
        self.v_train.append(v.flatten()[0])
        self.gp_f.fit([ [elt] for elt in self.X_train],[[elt] for elt in self.f_train])
        self.gp_v.fit([ [elt] for elt in self.X_train], [[elt-1.5] for elt in self.v_train])
        if (v>1.2):
            self.best_score = max(self.best_score,f)
        self.t += 1

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """

        # TODO: enter your code here

        return self.X_train[np.argmax(np.array(self.f_train)*(np.array(self.v_train)>1.2001))]


""" Toy problem to check code works as expected """

def check_in_domain(x):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= domain[None, :, 0]) and np.all(x <= domain[None, :, 1])


def f(x):
    """Dummy objective"""
    mid_point = domain[:, 0] + 0.5 * (domain[:, 1] - domain[:, 0])
    return - np.linalg.norm(x - mid_point, 2)  # -(x - 2.5)^2


def v(x):
    """Dummy speed"""
    return 2.0

def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*domain[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val > SAFETY_THRESHOLD]
    np.random.seed(SEED)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]
    return x_init



def main():
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    print(x_init)
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)


    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, domain.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {domain.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x)
        cost_val = v(x)
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = np.atleast_2d(agent.get_solution())
    assert solution.shape == (1, domain.shape[0]), \
        f"The function get solution must return a numpy array of shape (" \
        f"1, {domain.shape[0]})"
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the ' \
        f'domain, {solution} returned instead'

    # Compute regret
    if v(solution) < 1.2:
        regret = 1
    else:
        regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret{regret}')


if __name__ == "__main__":
    main()