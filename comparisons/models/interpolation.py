import os
import warnings
import numpy as np
try:
    from scipy.interpolate import RBFInterpolator
except ImportError:
    RBFInterpolator = None

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
    from sklearn.exceptions import ConvergenceWarning
except ImportError:
    GaussianProcessRegressor = None
    ConvergenceWarning = Warning

class BaseInterpolator:
    def __init__(self, grid_size=(32, 32, 32)):
        self.grid_size = grid_size
        self.D, self.H, self.W = grid_size
        # Create grid coordinates once
        z, y, x = np.meshgrid(
            np.arange(self.D), 
            np.arange(self.H), 
            np.arange(self.W), 
            indexing='ij'
        )
        self.grid_coords = np.stack([z.ravel(), y.ravel(), x.ravel()], axis=1)

    def fit_predict(self, obs_grid):
        """
        Args:
            obs_grid: (2, D, H, W) numpy array. 
                      Channel 0 is values, Channel 1 is mask (1=observed).
        Returns:
            (1, D, H, W) numpy array of interpolated probabilities [0, 1].
        """
        raise NotImplementedError

    def _extract_points(self, obs_grid):
        values = obs_grid[0]
        mask = obs_grid[1]
        
        # mask indices
        z_idx, y_idx, x_idx = np.where(mask > 0.5)
        
        if len(z_idx) == 0:
            return None, None
            
        points = np.stack([z_idx, y_idx, x_idx], axis=1)
        vals = values[z_idx, y_idx, x_idx]
        
        return points, vals

class RBFModel(BaseInterpolator):
    def __init__(self, grid_size=(32, 32, 32), kernel='linear'):
        super().__init__(grid_size)
        self.kernel = kernel
        if RBFInterpolator is None:
            raise ImportError("scipy >= 1.7.0 is required for RBFInterpolator")

    def fit_predict(self, obs_grid):
        points, vals = self._extract_points(obs_grid)
        if points is None:
            return np.zeros((1, *self.grid_size), dtype=np.float32)

        # RBF Interpolator
        # neighbors=None means global RBF (slow but standard)
        try:
            interp = RBFInterpolator(points, vals, kernel=self.kernel)
            pred_vals = interp(self.grid_coords)
            
            # Reshape
            pred_vol = pred_vals.reshape(self.D, self.H, self.W)
            
            # Clip/Sigmoid to 0-1
            pred_vol = np.clip(pred_vol, 0.0, 1.0)
            
            return pred_vol[None, ...]
        except Exception as e:
            print(f"RBF Error: {e}")
            return np.zeros((1, *self.grid_size), dtype=np.float32)

class KrigingModel(BaseInterpolator):
    """
    Using Gaussian Process Regression as a proxy for Ordinary Kriging.
    """
    def __init__(self, grid_size=(32, 32, 32)):
        super().__init__(grid_size)
        if GaussianProcessRegressor is None:
            raise ImportError("scikit-learn is required for KrigingModel")
        
        # Kernel: Constant * RBF + Noise
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=5.0, length_scale_bounds=(1e-1, 50.0))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, normalize_y=True)

    def fit_predict(self, obs_grid):
        points, vals = self._extract_points(obs_grid)
        if points is None:
            return np.zeros((1, *self.grid_size), dtype=np.float32)
            
        # Limit points for Kriging/GP as it is O(N^3)
        # If too many drill points, sample subset
        MAX_POINTS = 1000
        if points.shape[0] > MAX_POINTS:
            idx = np.random.choice(points.shape[0], MAX_POINTS, replace=False)
            points = points[idx]
            vals = vals[idx]

        try:
            show_gp_warnings = os.environ.get('SHOW_GP_WARNINGS', '0').lower() in ('1', 'true', 'yes', 'on')
            if show_gp_warnings:
                self.gp.fit(points, vals)
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=ConvergenceWarning)
                    self.gp.fit(points, vals)
            pred_vals, sigma = self.gp.predict(self.grid_coords, return_std=True)
            
            # Reshape
            pred_vol = pred_vals.reshape(self.D, self.H, self.W)
            
            # Clip/Sigmoid
            pred_vol = np.clip(pred_vol, 0.0, 1.0)
            
            return pred_vol[None, ...]
        except Exception as e:
            print(f"Kriging Error: {e}")
            return np.zeros((1, *self.grid_size), dtype=np.float32)

class IDWModel(BaseInterpolator):
    def __init__(self, grid_size=(32, 32, 32), power=2.0):
        super().__init__(grid_size)
        self.power = power

    def fit_predict(self, obs_grid):
        points, vals = self._extract_points(obs_grid)
        if points is None:
            return np.zeros((1, *self.grid_size), dtype=np.float32)

        # Vectorized simple IDW
        # grid_coords: (N_grid, 3)
        # points: (N_obs, 3)
        
        # We process in chunks to avoid OOM if grid is large
        N_grid = self.grid_coords.shape[0]
        N_obs = points.shape[0]
        CHUNK_SIZE = 10000 
        
        pred_vals = np.zeros(N_grid, dtype=np.float32)
        
        for i in range(0, N_grid, CHUNK_SIZE):
            end = min(i + CHUNK_SIZE, N_grid)
            chunk_coords = self.grid_coords[i:end] # (B, 3)
            
            # Dist: (B, N_obs)
            dists = np.sqrt(((chunk_coords[:, None, :] - points[None, :, :])**2).sum(axis=2))
            
            # Avoid divide by zero
            dists = np.maximum(dists, 1e-6)
            
            weights = 1.0 / (dists ** self.power)
            
            # Weighted average
            sum_weights = weights.sum(axis=1)
            weighted_vals = (weights * vals[None, :]).sum(axis=1)
            
            pred_vals[i:end] = weighted_vals / (sum_weights + 1e-8)

        pred_vol = pred_vals.reshape(self.D, self.H, self.W)
        pred_vol = np.clip(pred_vol, 0.0, 1.0)
        return pred_vol[None, ...]

