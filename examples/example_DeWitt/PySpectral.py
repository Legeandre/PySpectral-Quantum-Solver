#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Library-name: PySpectral Quantum Solver
Description: Library developed in Python with spectral methods for solving eigenvalue problems and wave functions in quantum wells.

Author:         Vagner Jandre Monteiro  
Contact:        <vagner.jandre@iprj.uerj.br>
Create date:    2025-03-24 
last updated:   2026-02-15 
Version:        2.0.0  
Licence:        MIT License  
Repository: ...
Dependencies:
  – Python stdlib:
      • warnings
      • datetime
      • os
      • numbers
      • time
      • inspect
  – NumPy >= 1.20
  – SymPy >= 1.8
  – Pandas >= 1.3
  – Matplotlib >= 3.3
      • matplotlib.animation
      • matplotlib.ticker
      • mpl_toolkits.mplot3d.Axes3D
  – SciPy >= 1.6
      • scipy.linalg.eig, scipy.linalg.eigh
      • scipy.integrate.quad, scipy.integrate.fixed_quad
      • scipy.sparse.linalg.eigsh
      • scipy.optimize.minimize_scalar

Changelog: ...

"""
#__author__    = "Vagner Jandre Monteiro <vagner.jandre@iprj.uerj.br>"
#__version__   = "2.0.0"
#__license__   = "MIT"
#__repo_url__  = "https://github.com/Legeandre/PySpectral-Quantum-Solver"


# =========================================================================== 
# Dependences
# =========================================================================== 
import warnings
warnings.simplefilter(action='ignore')

import os
import numpy as np
import sympy as sp
import datetime
import numbers
import time
import inspect
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scipy.linalg import eig, eigh
from scipy.integrate import fixed_quad, quad
from scipy.optimize import minimize_scalar
from matplotlib.ticker import MaxNLocator
from scipy.signal import find_peaks
from typing import Callable, Optional
from contextlib import contextmanager
from scipy.special import roots_legendre

import random
# Fix random seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# =========================================================================== 
# Safe Decorator
# =========================================================================== 
def safe_execution(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"[ERRO em {func.__name__}] {e}")
            return None 
    return wrapper

# =========================================================================== 
# Class
# =========================================================================== 
class SpectralMethod:

    def __init__(self, num_levels: int, length: float, 
                 f_function: Optional[Callable] = None, 
                 g_function: Optional[Callable] = None, 
                 num_digits: int = 15, 
                 weight: Optional[Callable] = None, 
                 root_filename: str = "output", 
                 label: str = "project", 
                 optimizer_L: str = 'n',
                 optimize_N: int = None,        # Specific N for optimization
                 L_bounds: tuple = (0.1, 50.0), # Bounds for L
                 ensure_reproducibility: bool = False):  
                 
        
        """
        Initializes the spectral problem with validation and setup.
        """
        self.num_levels = num_levels
    
        # Function definitions (with safe fallbacks)
        self.f_function = f_function if f_function is not None else lambda x: np.float64(0.0)
        self.g_function = g_function if g_function is not None else lambda x: np.float64(1.0)
        self.weight = weight if weight is not None else lambda x: np.float64(1.0)
    
        # Precision and state settings
        self.num_digits = num_digits
        self.digits_used = num_digits
        
        self.has_spectrum_been_calculated = False
        self.has_eigenvectors_been_calculated = False
        self.en_spectrum = None
        self.eigenvectors = None     
        self.version = "PySpectral"
        
        # --- File management ---
        self.root_filename = root_filename  
        self.label = label            
        self.calculator_name = "Python"
        
        # Ensure the output folder exists. If it does not exist, create it.
        if self.root_filename and not os.path.exists(self.root_filename):
            try:
                os.makedirs(self.root_filename)
                print(f"Directory '{self.root_filename}' created successfully.")
            except OSError as e:
                print(f"Warning: Could not create directory '{self.root_filename}'. Error: {e}")

        if ensure_reproducibility:
            # Configure environment for maximum reproducibility
            self._setup_reproducible_environment()
        
        # Cache for weight matrix
        self._weight_cache = {}
        self._f_integral_cache = {}
        self._g_integral_cache = {}

        # Precompute constants that will be used frequently
        self._pi = np.pi
        self._pi_squared = self._pi ** 2

        # Set the initial length provided by the user
        self.length = np.float64(length)
        
        if optimizer_L.upper() == 'Y':
            print(f"Optimizing the box length. Please wait…")
            
            opt_N = optimize_N if optimize_N is not None else num_levels
            
            # Clear caches
            self.clear_caches()

            self.length = self.optimize_length(
                num_levels_opt=opt_N,
                L_bounds=L_bounds,
                verbose=True
            )
            
            print(f"Optimized length: L = {self.length:.6f} (N={opt_N})")

    def define_problem(self, num_levels, length, f_function, g_function, variable, weight=1, label="Problem", num_digits=15):
        """
            Define the problem parameters.
            Args:
            - num_levels: Number of energy levels (positive integer).
            - length: Length of the interval (real, positive number).
            - f_function: Function representing `f`.
            - g_function: Function representing `g`.
            - variable: Independent variable.
            - weight: Optional weight function (default is 1).
            - label: Problem label (default is "Problem").
            - num_digits: Precision for calculations (default is 15).
        """
        if num_digits > 15:
            print("\nWARNING: Working with more than 15 digits may result in numerical instabilities.")

        # Store parameters
        self.problem = {
            "num_levels": num_levels,
            "length": length,
            "f_function": f_function,
            "g_function": g_function,
            "variable": variable,
            "weight": weight,
            "label": label,
            "num_digits": num_digits
        }
        print(f"Problem defined with label: {label}")

    @contextmanager
    def _temporary_state(self, **kwargs):
        """
        Context manager to temporarily modify class attributes.
        Clears the caches only if length is changed.
        """
        old_values = {}
        old_caches = {}
        
        try:
            # Save current values
            for key, value in kwargs.items():
                old_values[key] = getattr(self, key)
                setattr(self, key, value)
            
            # If length changes, clear all caches
            if 'length' in kwargs and kwargs['length'] != old_values.get('length'):
                old_caches = {
                    'weight': self._weight_cache.copy(),
                    'f_integral': self._f_integral_cache.copy(),
                    'g_integral': self._g_integral_cache.copy()
                }
                self._weight_cache.clear()
                self._f_integral_cache.clear()
                self._g_integral_cache.clear()
            
            yield self
            
        finally:
            # Restore previous values
            for key, value in old_values.items():
                setattr(self, key, value)
            
            # Restore caches if they were saved
            if old_caches:
                self._weight_cache = old_caches.get('weight', {})
                self._f_integral_cache = old_caches.get('f_integral', {})
                self._g_integral_cache = old_caches.get('g_integral', {})

    def _setup_reproducible_environment(self):
        """Configure the environment for reproducible calculations"""
        
        # Limit parallelism (avoids non-determinism)
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        
        # Configure NumPy for deterministic mode
        np.random.seed(42)  # Even if randomness is not explicitly used
        np.set_printoptions(precision=16)  # For debugging
        
        # Configure floating-point behavior
        try:
            np.seterr(all='raise')  # Capture numerical errors
        except:
            pass
        
        print("Reproducibility mode enabled: 1 thread, fixed seed.")
    
    def __get_integration_grid(self):
        """
        Generates Gauss–Legendre quadrature nodes mapped to the interval [0, L], ensuring exact accuracy for polynomials up to approximately degree 4*N.
        """
        # A high N_quad ensures that oscillatory integrals are resolved with machine precision
        n_quad = max(200, 2 * self.num_levels + 50)
        
        # Generate nodes and weights in [-1, 1]
        x_leg, w_leg = roots_legendre(n_quad)
        
        # Map them to [0, L]
        half_L = self.length * 0.5
        x_real = half_L * (x_leg + 1.0)
        weights = w_leg * half_L
        
        return x_real, weights, n_quad

    def __CIntegral(self, func, m, n):
        """
        Computes the integral associated with the elements of C.
            For both diagonal and off-diagonal elements, uses the same formulation
            with the function `func` (which can be `f` or `g`).

            Args:
                func (callable): The function (`f` or `g`) to be integrated.
                m, n (int): Indices of the matrix element.

            Returns:
                Numerical value of the integral.

            Note:
                In the original code, the integral is defined as:
                    2/L * ∫[0,L] sin(nπx/L) * func(x) * sin(mπx/L) dx
        """
        L = self.length
        m, n = int(m), int(n)
        
        # Identify which cache to use
        if func is self.f_function:
            cache_dict = self._f_integral_cache
        elif func is self.g_function:
            cache_dict = self._g_integral_cache
        else:
            cache_dict = self._weight_cache
        
        cache_key = (m, n, L)
        if cache_key in cache_dict:
            return cache_dict[cache_key]
        
        # Adaptive number of points
        max_index = max(m, n)
        n_points = min(100 + max_index * 30, 5000)
        
        pi_L = self._pi / L
        
        # CORRECT formulas for the sine basis
        if m == n:
            # ∫₀ᴸ sin²(mπx/L) f(x) dx = ∫₀ᴸ 0.5(1 - cos(2mπx/L)) f(x) dx
            integrand = lambda x: 0.5 * (1.0 - np.cos(2.0 * m * pi_L * x)) * func(x)
        else:
            # ∫₀ᴸ sin(mπx/L) sin(nπx/L) f(x) dx 
            # = ∫₀ᴸ 0.5[cos((m-n)πx/L) - cos((m+n)πx/L)] f(x) dx
            integrand = lambda x: 0.5 * (np.cos((m - n) * pi_L * x) - np.cos((m + n) * pi_L * x)) * func(x)
        
        # Compute the integral
        result, _ = fixed_quad(integrand, 0.0, L, n=n_points)
        
        # Normalization factor: (2/L) for the orthonormal basis
        result = (2.0 / L) * result
        
        # Cache
        cache_dict[cache_key] = result
        return result

    # --- This method is deprecated and has been replaced by higher-precision implementations.
    def __C_from_f_offdiagonal(self, m, n):
        """
            Off-diagonal, for f.
            Definitions for the elements constructed from function f.
        """
        return self.__CIntegral(self.f_function, m, n)
    
    def __C_from_f_diagonal(self, m):
        """
            Diagonal element, for f.
        """
        return self.__CIntegral(self.f_function, m, m)
    
    def __C_from_g_offdiagonal(self, m, n):
        """
            Off-diagonal, for g.
            Definitions for the elements constructed from function g.
        """
        return self.__CIntegral(self.g_function, m, n)
    
    def __C_from_g_diagonal(self, m):
        """
            Equation (30), diagonal element, for g.
        """
        return self.__CIntegral(self.g_function, m, m)
    
    def __C(self, n, m):
        """
            Computes the element C[n,m] from Equation (29) of [Pedran2008].  
            If n == m, uses the diagonal version; otherwise, the off-diagonal version.  
            (n and m are integers, 1-based.)
        """
        if n == m:
            return self.__C_from_f_diagonal(m)
        else:
            return self.__C_from_f_offdiagonal(m, n)
    
    def __C2(self, n, m):
        """
            Computes the element C'[n,m] from Equation (30) of [Pedran2008].  
            If n == m, uses the diagonal version; otherwise, the off-diagonal version.
        """
        if n == m:
            return self.__C_from_g_diagonal(m)
        else:
            return self.__C_from_g_offdiagonal(m, n)
    # ---
    
    # ----- Methods for Weight Matrix -----
    
    # --- This method is deprecated and has been replaced by higher-precision implementations.
    def __weight_matrix_offdiagonal(self, m, n):
        """
            Returns the **(m, n) element** of the **off-diagonal weight matrix**.
        """
        return self.__CIntegral(self.weight, m, n)

    def __weight_matrix_diagonal(self, m):
        """
            Returns the **(m, m) diagonal element** of the **weight matrix**.
        """
        return self.__CIntegral(self.weight, m, m)

    def __weight_function(self, n, m):
        """
            Returns the **(n, m) element** of the **weight matrix**,  
            using **memorization** to avoid repeated recalculations.
        """
        key = (n, m)
        if key in self._weight_cache:
            return self._weight_cache[key]
        # If `n == m`, use the diagonal case; otherwise, use the off-diagonal case.
        if n == m:
            value = self.__weight_matrix_diagonal(m)
        else:
            value = self.__weight_matrix_offdiagonal(m, n)
        self._weight_cache[key] = value
        return value
    # --- 

    def __scalar_product(self, u, v):
        """
        Computes the dot product of vectors **u** and **v**,  
        considering the weight:  
        Σ₍i,j₎ uᵢ * WeightFunction(i,j) * vⱼ.
        The vectors are expected as lists (or arrays), and indexing is **1-based**.
        
        Computes u^T * W * v using a cached Weight Matrix.
        Optimized to build the Weight matrix via vectorized integration.
        """
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        dim = len(u)

        if len(v) != dim:
            raise ValueError("Vectors have different dimensions.")
        
        # Verify that the W matrix is already built and has the correct shape
        if not hasattr(self, '_W_matrix') or self._W_matrix.shape[0] != dim:          
            # Retrieve the integration grid
            x, w, _ = self.__get_integration_grid()
            
            # Evaluate the weight function
            try:
                w_vals = self.weight(x)
                if np.isscalar(w_vals): w_vals = np.full_like(x, w_vals)
            except:
                w_vals = np.array([self.weight(val) for val in x])
                
            # Build the sine basis for the requested dimension
            n_indices = np.arange(1, dim + 1, dtype=np.float64)
            arg = np.outer(x, n_indices) * (self._pi / self.length)
            B = np.sin(arg)
            
            # Perform integration using BLAS
            norm_factor = 2.0 / self.length
            weighted_B = B * (w * w_vals)[:, None]
            W_matrix = (B.T @ weighted_B) * norm_factor
            
            # Store in cache while enforcing exact symmetry
            self._W_matrix = 0.5 * (W_matrix + W_matrix.T)
        
        # Perform fast multiplication using BLAS
        return u @ self._W_matrix @ v
    
    # ----- Construction of matrices D and D' (Equation (32) from [Pedran2008]). -----
    
    # --- First
    def __FirstBigMatrix(self):
        """
        Builds the matrix D (of order N x N) according to Eq.(32) from [Pedran2008].
        Redirects to the ultra-fast vectorized construction.
        """
        return self.__build_matrix_d()

    def __SecondBigMatrix(self):
        """
        Builds the D′ matrix (N × N) according to Eq. (32).
        Redirects to the ultra-fast vectorized construction.
        """
        return self.__build_matrix_d_prime()
    
    # --- Second
    def __build_matrix_d(self):
        """
        Builds the D matrix (kinetic energy + f potential), defined in Eq. (32) of [Pedran2008], using a vectorized implementation.
        """
        N = self.num_levels
        
        # Obtém grade de integração 
        x, w, _ = self.__get_integration_grid()
        
        # Avalia a função f(x)
        try:
            f_vals = self.f_function(x)
            if np.isscalar(f_vals): f_vals = np.full_like(x, f_vals)
        except:
            f_vals = np.array([self.f_function(val) for val in x])
            
        # Constrói a Matriz de Base (Seno)
        n_indices = np.arange(1, N + 1, dtype=np.float64)
        arg = np.outer(x, n_indices) * (self._pi / self.length)
        B = np.sin(arg)
        
        # Integração Numérica via Álgebra Linear (BLAS) 
        norm_factor = 2.0 / self.length
        weighted_B = B * (w * f_vals)[:, None]
        matrix_d = (B.T @ weighted_B) * norm_factor
        
        # Adiciona o Termo Cinético na Diagonal
        if self.length > 1e6:
            pi_over_L = np.exp(np.log(self._pi) - np.log(self.length))
        else:
            pi_over_L = self._pi / self.length
            
        kinetic_term = (n_indices * pi_over_L) ** 2
        np.fill_diagonal(matrix_d, matrix_d.diagonal() + kinetic_term)
        
        # Força Simetria exata (Remove flutuações de 1e-16 da máquina)
        matrix_d = 0.5 * (matrix_d + matrix_d.T)
        
        return matrix_d

    def __build_matrix_d_prime(self):
        """
        Builds the D′ matrix (g potential), defined in Eq. (32) of [Pedran2008], using a vectorized implementation.
        """
        N = self.num_levels
        
        # Obtém grade (Desempacota exatamente 3 valores)
        x, w, _ = self.__get_integration_grid()
        
        # Avalia g(x)
        try:
            g_vals = self.g_function(x)
            if np.isscalar(g_vals): g_vals = np.full_like(x, g_vals)
        except:
            g_vals = np.array([self.g_function(val) for val in x])
            
        # Base Seno
        n_indices = np.arange(1, N + 1, dtype=np.float64)
        arg = np.outer(x, n_indices) * (self._pi / self.length)
        B = np.sin(arg)
        
        # Integração Vetorizada
        norm_factor = 2.0 / self.length
        weighted_B = B * (w * g_vals)[:, None]
        matrix_d_prime = (B.T @ weighted_B) * norm_factor
        
        # Simetria
        matrix_d_prime = 0.5 * (matrix_d_prime + matrix_d_prime.T)
        
        return matrix_d_prime
    
    # ----- Methods for Spectrum Calculation and Eigenvectors -----

    @safe_execution
    def is_solved(self, number_of_digits=None):
        """
        Solves the eigenvalue and eigenvector problem.
        Updates precision if provided, tracks time, and reports status.

            - If a value for number_of_digits (a positive integer) is provided, it updates the number of digits (precision) to be used in calculations.  

            Procedure:  
                1. Optional Updates self.num_digits with the provided value.  
                2. Computes and times the spectrum (eigenvalues) by calling self.calculate_spectrum().  
                3. Computes and times the eigenvectors by calling self.calculate_eigenvectors().  
                4. Displays a final message indicating that the problem has been solved, including the number of digits used.  
        """
        # Update Precision if requested
        if number_of_digits is not None:
            if isinstance(number_of_digits, int) and number_of_digits > 0:
                self.num_digits = number_of_digits
            else:
                raise ValueError("Number of digits must be a positive integer.")

        # Helper to run and time a step (reduces repeated code blocks)
        def run_step(name, check_flag, func):
            if not check_flag:
                print(f"Calculating {name} ...")
                t0 = time.time()
                func()
                print(f"Time elapsed for {name} calculation: {time.time() - t0:.6f} seconds.")
            else:
                print(f"{name.capitalize()} already calculated. Skipping.")

        # Execute Steps
        run_step("eigenvalues", self.has_spectrum_been_calculated, self.calculate_spectrum)
        run_step("eigenvectors", self.has_eigenvectors_been_calculated, self.calculate_eigenvectors)

        # Final Report
        self.calculator_name = "Python"
        print("=" * 80)
        print(f"The eigenvalue/eigenvector problem has been completely solved with {self.digits_used} digits used.")

    @safe_execution
    def calculate_spectrum(self, save_to_file: bool = True):
        """
        Computes the eigenvalue spectrum, prints status, and optionally saves to file.
        Wraps the logic of calculate_spectrum_silently to avoid code duplication.

        Computes the eigenvalue spectrum for the generalized eigenvalue problem,  
            solving A*v = λ*B*v, where A and B are the matrices D and D′  
            (built using the functions build_matrix_d() and build_matrix_d_prime(), respectively).  

            The method performs several attempts:  
                1. Initially, it sets the number of digits (precision) and informs the user.  
                2. Computes the spectrum using scipy.linalg.eig.  
                3. If the number of roots found differs from num_levels or if any eigenvalue  
                has a nonzero imaginary part beyond tolerance, the number of digits is increased,  
                and the calculation is retried.  
                4. When all eigenvalues are real and the correct number of roots is found,  
                the spectrum is sorted, stored, and returned.  

            Computes the eigenvalue spectrum of the generalized problem A*v = λ*B*v,  
            where A = D and B = D′. If the number of eigenvalues found or their nature  
            (all real) does not meet the conditions, it increases "precision" (simulated by the `digits` variable)  
            and tries again.  

            At the end, it sets:  
            - `self.en_spectrum`: ordered list of eigenvalues.  
            - `self.digits_used`.  
            - `self.has_spectrum_been_calculated = True`.  
            
            # digits is a symbolic indicator of precision control,
            # used to retry with simulated "increased effort"
        """
        screen_width = 80
        print("_" * screen_width)
        print("Calculating energy eigenvalues...")
        print("_" * screen_width)

        # Calls the robust silent method to do the heavy lifting
        self.calculate_spectrum_silently()

        # Reporting
        nroots = len(self.en_spectrum)
        if nroots != self.num_levels:
            print(f"Warning: only {nroots} eigenvalues found, expected {self.num_levels}.")
        
        print("Finished eigenvalue calculation.")
        print("_" * screen_width)

        # --- Save to File TXT ---
        if save_to_file:
            filename = f"{self.root_filename}/{self.root_filename}_Spectrum.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"# {filename}\n")
                f.write(f"# Program version : {self.version}\n")
                f.write(f"# Date: {datetime.datetime.now().strftime('%c')}\n")
                f.write("# (1) Index, (2) Eigenvalue\n")
                f.write("# " + "="*80 + "\n")
                for idx, val in enumerate(self.en_spectrum, start=1):
                    f.write(f"{idx}\t{val:.12e}\n")

            print(f"Spectrum saved to: {filename}")

        return self.en_spectrum

    @safe_execution
    def calculate_spectrum_silently(self):
        """
        Core worker: Computes the eigenvalue spectrum (D*v = lambda*D'*v).
        Includes logic to increase precision (digits) and retry if roots are complex or missing.

        Computes the eigenvalue spectrum silently, meaning without  
            displaying messages to the user.  

            The method solves the generalized problem:  
            A*v = λ*B*v,  
            where A and B are the matrices D and D′, respectively.  

            For each attempt, if:  
            - The number of eigenvalues found differs from num_levels, or  
            - Any eigenvalue has an imaginary part (nonzero within tolerance),  
              
            The precision (digits) is adjusted if the number of eigenvalues doesn't match num_levels or if any eigenvalue has an imaginary part.
        
            At completion:  
            - self.digits_used stores the precision used,  
            - self.has_spectrum_been_calculated is set to True,  
            - self.en_spectrum receives the ordered spectrum.  

            Returns:  
            An ordered list containing the eigenvalues.  

        """
        digits = self.num_digits
        digits_percent_increase = 0.1
        
        # Iterative loop to ensure numerical stability
        while True:
            # Rebuild matrices (assumes build_matrix uses 'digits' or self.digits_used if updated)
            
            A = self.__build_matrix_d()
            B = self.__build_matrix_d_prime()

            # Eigenvalue calculation (right=False is faster if we only want spectrum)
            evs = eig(A, B, right=False)

            # Filter: clean small imaginary parts and non-finites
            res = []
            has_complex = False
            for u in evs:
                if not np.isfinite(u): continue
                if np.isclose(u.imag, 0, atol=1e-12):
                    res.append(u.real)
                else:
                    res.append(u)
                    has_complex = True

            nroots = len(res)
            
            # Retry condition: wrong number of roots or presence of complex roots
            if nroots != self.num_levels or has_complex:
                # Increases precision and loops again
                digits = int(digits * (1 + digits_percent_increase))
                continue
            
            # If execution reaches this point, the operation was successful
            break

        # Final step
        self.digits_used = digits
        self.en_spectrum = sorted(res)
        self.has_spectrum_been_calculated = True

        return self.en_spectrum

    @safe_execution
    def its_eigenvalues(self, *args):
        """
        Returns the calculated eigenvalue spectrum with flexible indexing.
        Supports: int, slice, range, or list of ints (1-based indexing).

        Args:
            *args: int, slice, range, or list of ints in 1-based indexing.

            Returns:
                Single eigenvalues (if single int), or list of eigenvalues.

            Examples:  
                self.its_eigenvalues() : returns the complete list of eigenvalues.  
                self.its_eigenvalues(1) : returns the 1st eigenvalue.  
                self.its_eigenvalues(slice(1,4)) : returns the eigenvalues for levels 1, 2, and 3.  
                self.its_eigenvalues(range(1,4)) : returns the eigenvalues for levels 1, 2, and 3.  
                self.its_eigenvalues(1, 3, 5) : returns a list with the 1st, 3rd, and 5th eigenvalue.  

            Use a slice or range object, for example: slice(1,4) or range(1,4).

        """
        if not self.has_spectrum_been_calculated:
            self.calculate_spectrum_silently()

        # Case 0: No arguments -> Return full spectrum
        if not args:
            return self.en_spectrum

        arg = args[0]

        # Case 1: Multiple integers passed as args -> its_eigenvalues(1, 3, 5)
        if len(args) > 1:
            return [self.en_spectrum[i - 1] for i in args if isinstance(i, int)]

        # Case 2: Single argument handling
        if isinstance(arg, int):
            if arg <= 0:
                raise ValueError("Index must be a positive integer.")
            return self.en_spectrum[arg - 1]
        
        elif isinstance(arg, slice):
            # Convert 1-based slice to 0-based
            start = arg.start - 1 if arg.start is not None else None
            stop = arg.stop - 1 if arg.stop is not None else None
            return self.en_spectrum[slice(start, stop, arg.step)]
        
        elif isinstance(arg, (range, list, tuple)):
            # Handle list/range/tuple of indices
            return [self.en_spectrum[i - 1] for i in arg]
        
        else:
            raise ValueError(f"Unsupported argument type: {type(arg)}. Use int, slice, range, or list.")

    @safe_execution
    def its_eigenvalues_silently(self, *args):
        """
        Silently retrieve eigenvalues. Typically used internally.
        Args:
            - args: Can be empty or a single positive integer.
            Returns:
            - List of eigenvalues or a specific eigenvalue.
        """
        if not self.has_spectrum_been_calculated:
             raise RuntimeError("Spectrum hasn't been calculated yet.")

        if args and isinstance(args[0], int) and args[0] > 0:
            return self.en_spectrum[args[0] - 1]
        
        return self.en_spectrum

    @safe_execution
    def calculate_eigenvectors(self, save_to_file: bool = True):
        """
        Computes eigenvalues AND eigenvectors.
        Note: Even if spectrum was calculated, we must run eig(right=True) again to get vectors.
        Computes the eigenvectors for the generalized eigenvalue problem A*v = λ*B*v.
        After obtaining eigenvalues and eigenvectors (via scipy.linalg.eig), the results are
        sorted according to the eigenvalues, and each eigenvector is normalized so that
        the first nonzero element is positive.

        The results are stored in self.eigenvectors as a list of pairs [E, eigenvector],
        and the flag self.has_eigenvectors_been_calculated is set to True.
        
        If save_to_file=True (Default), saves results in "<root_filename>_Eigenvectors.txt".
        """
        # 1. Ensure spectrum is calculated first.
        # This is crucial because calculate_spectrum determines the optimal 'digits'
        # needed to avoid complex numbers. We want to use that same stability here.
        if not self.has_spectrum_been_calculated:
            print("The energy spectrum hasn't been calculated yet. Calculating first to stabilize precision...")
            self.calculate_spectrum()
        
        if self.has_eigenvectors_been_calculated:
            print("Eigenvectors already calculated. Returning cached results.")
            return self.eigenvectors

        print("_" * 80)
        print("Calculating energy eigenvectors now ...")
        
        # 2. Build matrices (using the stable precision found in calculate_spectrum)
        A = self.__build_matrix_d()
        B = self.__build_matrix_d_prime()
        
        # 3. Solve (right=True gets vectors)
        w, V = eig(A, B, right=True)

        # 4. Filter / Clean
        w = np.real_if_close(w, tol=1e-12)
        V = np.real_if_close(V, tol=1e-12)

        # 5. Sort based on eigenvalues
        sorted_idx = np.argsort(w)
        w_sorted = w[sorted_idx]
        V_sorted = V[:, sorted_idx]

        # 6. Normalize
        eigenvectors = []
        for i in range(self.num_levels):
            # Safeguard in case w returns more roots than expected
            if i >= len(w_sorted): break 
            
            vec = V_sorted[:, i]
            
            # Normalization logic inline
            norm = self.__scalar_product(vec, vec)
            if norm > 1e-12:
                vec = vec / np.sqrt(norm)
                # Ensure first significant element is positive
                first_nonzero = vec[np.argmax(np.abs(vec) > 1e-12)]
                if first_nonzero < 0:
                    vec = -vec
            
            eigenvectors.append([w_sorted[i], vec])

        # Store
        self.eigenvectors = eigenvectors
        self.has_eigenvectors_been_calculated = True
        
        # Update spectrum logic too (consistency check)
        self.en_spectrum = list(w_sorted[:self.num_levels])

        # --- Save to File ---
        if save_to_file:
            filename = f"{self.root_filename}/{self.root_filename}_Eigenvectors.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"# {filename}\n")
                f.write(f"# Program version : {self.version}\n")
                f.write(f"# Date: {datetime.datetime.now().strftime('%c')}\n")
                f.write("# (1) Eigenvalue, (2) Eigenvector (full)\n")
                f.write("# " + "="*80 + "\n")
                for eigval, eigvec in eigenvectors:
                    # Optimized formatting for faster file writing
                    vec_str = ", ".join(f"{x:.12e}" for x in eigvec.flatten())
                    f.write(f"{eigval:.12e}\t[{vec_str}]\n")

            print(f"Eigenvectors saved to: {filename}")
            
        print("_" * 80)
        print("Done!")
        print("=" * 80)

        return eigenvectors

    @safe_execution
    def its_eigenpairs(self, *args, save_to_file: bool = True):
        """
        Returns eigenvalue/eigenvector pairs (λ, v) in 1-based indexing.
        Master function for data retrieval.
        
        Optionally saves all results in a TXT file with column format.

        Args:
            *args: Optional integer indices, slice, range, or list of ints.
            save_to_file (bool): If True, saves results to "<root_filename>_Eigenpairs.txt".

        Returns:
            - A single [λ, v] pair if a single int is provided.
            - A list of [λ, v] pairs otherwise.
        
        Examples:
            self.its_eigenpairs()             -> full list of [λ, v] pairs
            self.its_eigenpairs(1)            -> first pair [λ₁, v₁]
            self.its_eigenpairs(slice(1,4))   -> pairs [λ₁, v₁], [λ₂, v₂], [λ₃, v₃]
            self.its_eigenpairs(range(1,4))   -> same as above
            self.its_eigenpairs(1, 3, 5)      -> list with 1st, 3rd, and 5th pairs
        """
        if not self.has_eigenvectors_been_calculated:
            self.calculate_eigenvectors()

        # Logic matches exactly what was done for eigenvalues, handling all cases.
        if not args:
            result = self.eigenvectors
        else:
            arg = args[0]
            # Case: Multiple arguments passed -> its_eigenpairs(1, 3, 5)
            if len(args) > 1:
                indices = [i for i in args if isinstance(i, int)]
                result = [self.eigenvectors[i - 1] for i in indices]
            
            # Case: Single argument
            elif isinstance(arg, int):
                if arg <= 0: raise ValueError("Index must be positive.")
                result = self.eigenvectors[arg - 1] # Returns single pair [λ, v]
            elif isinstance(arg, slice):
                start = arg.start - 1 if arg.start is not None else None
                stop = arg.stop - 1 if arg.stop is not None else None
                result = self.eigenvectors[slice(start, stop, arg.step)]
            elif isinstance(arg, (list, tuple, range)):
                result = [self.eigenvectors[i - 1] for i in arg]
            else:
                raise ValueError(f"Unsupported argument type: {type(arg)}")

        # --- Save to File ---
        if save_to_file:
            filename = f"{self.root_filename}/{self.root_filename}_Eigenpairs.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"# {filename}\n")
                f.write(f"# Program version : {self.version}\n")
                f.write("# (1) Eigenvalue, (2) Eigenvector (full)\n")
                f.write("# " + "="*80 + "\n")

                # Normalize to list for iteration (handle single pair case)
                # If result is [E, v] (single pair), wrap it in list -> [[E, v]]
                # Check: List of lists/arrays?
                pairs_to_write = [result] if (isinstance(result, list) and len(result) == 2 and isinstance(result[0], (int, float, complex))) else result
                
                # If it's a list of pairs (standard case)
                if not isinstance(pairs_to_write, list): 
                    # Fallback for slice causing ndarray or similar
                    pairs_to_write = result

                for eigval, eigvec in pairs_to_write:
                    # Flatten and format string
                    vec_str = "[" + ", ".join(f"{comp:.12e}" for comp in np.array(eigvec).flatten()) + "]"
                    f.write(f"{eigval:.12e}\t{vec_str}\n")

            print(f"Eigenpairs saved in: {filename}")

        return result

    @safe_execution
    def its_eigenvectors(self, *args):
        """
        Returns the computed eigenvectors (or a subset) in 1-based indexing.
        Wrapper around its_eigenpairs to avoid code duplication.

        Args:
            *args: int, slice, range, or list of ints in 1-based indexing.

        Returns:
            Single eigenvector (if single int), or list of eigenvectors.
            
        Examples:
            self.its_eigenvectors()             -> full list of eigenvectors
            self.its_eigenvectors(1)            -> first eigenvector
            self.its_eigenvectors(slice(1,4))   -> eigenvectors 1,2,3
            self.its_eigenvectors(range(1,4))   -> eigenvectors 1,2,3
            self.its_eigenvectors(1, 3, 5)      -> list [v1, v3, v5]
        """
        # Reuse logic from its_eigenpairs
        pairs = self.its_eigenpairs(*args, save_to_file=False)

        # Extract just the vectors (v) from [λ, v]
        
        # Case A: Single pair returned (args was a single int)
        # Check: [scalar, vector] structure
        if isinstance(pairs, list) and len(pairs) == 2 and isinstance(pairs[0], (int, float, complex)):
             return pairs[1]

        # Case B: List of pairs returned (slice, range, or no args)
        return [p[1] for p in pairs]

    @safe_execution
    def its_eigenfunction(self, x, i, tolerance=1e-15):
        """
        Returns the i-th normalized eigenfunction (ψ_i).

            ψ_i(x) = sqrt(2/L) * Σₘ₌₁^N [aₘ * sin(m π x / L)]

        - Numeric x: Uses vectorized NumPy calculation (via an_eigenfunction).
        - Symbolic x: Returns a SymPy expression.

        Args:
            x (float, array, str, sympy.Symbol): Position(s).
            i (int): Quantum level (1-based index).
            tolerance (float): Threshold to ignore negligible coefficients (optimization).

        Returns:
            float, np.ndarray, or sympy.Expr
        """
        # Validate Attributes
        if not hasattr(self, 'length') or not hasattr(self, 'num_levels'):
            raise AttributeError("Attributes 'length' and 'num_levels' are missing.")
        
        # Get Coefficients for level i
        if not hasattr(self, 'eigenvectors') or not self.eigenvectors:
             # Try to calculate if missing, or raise error
            raise ValueError("Eigenvectors not found. Please run solve() first.")
        
        # Extract vector part (v) from [E, v]
        coeffs = self.eigenvectors[i - 1][1]

        # Numeric Path 
        if isinstance(x, (numbers.Number, np.ndarray, list)):
            return self.an_eigenfunction(x, coeffs, tolerance=tolerance)

        # Symbolic Path (SymPy)
        return self._symbolic_eigenfunction(x, coeffs, tolerance)

    @safe_execution
    def an_eigenfunction(self, x, basis_coeffs, tolerance=0.0):
        """
        Calculates ψ(x) numerically given specific basis coefficients.
        
        Physics: Represents the expansion in the Infinite Well basis (sine basis).
        Domain: 0 <= x <= L. Returns 0 outside this domain if enforcing physics.

        Args:
            x (float or array_like): Positions.
            basis_coeffs (array_like): Eigenvector coefficients (a_m).
            tolerance (float): Coefficients with absolute value < tolerance are ignored.

        Returns:
            float or np.ndarray: The wavefunction value(s).
        """
        L = self.length
        
        # Prepare Inputs
        x_arr = np.atleast_1d(x).astype(float)
        coeffs = np.asarray(basis_coeffs, dtype=float)
        
        # Filter small coefficients to reduce matrix operations
        m_indices = np.arange(1, len(coeffs) + 1)
        
        if tolerance > 0:
            mask_c = np.abs(coeffs) > tolerance
            if not np.any(mask_c): return np.zeros_like(x_arr) if x_arr.size > 1 else 0.0
            coeffs = coeffs[mask_c]
            m_indices = m_indices[mask_c]

        # Calculation: ψ(x) = sqrt(2/L) * Σ a_m * sin(m*π*x/L)
        # Note: (m * pi / L) is computed once per m
        args = np.outer(m_indices, x_arr) * (np.pi / L) 
        basis_values = np.sin(args)
        
        # Dot product: sum(coef * sin) over the 'm' axis
        psi_values = np.sqrt(2.0 / L) * np.dot(coeffs, basis_values)

        # Physics Enforce: Boundary Conditions
        # The particle cannot exist outside [0, L]. The sine function repeats, 
        # so we manually zero out values outside the box.
        outside_domain = (x_arr < 0) | (x_arr > L)
        if np.any(outside_domain):
            psi_values[outside_domain] = 0.0

        # Return scalar if input was scalar
        if np.ndim(x) == 0:
            return psi_values.item()
        return psi_values

    # --- Helper method for symbolic eigenfunction 
    def _symbolic_eigenfunction(self, x_str, coeffs, tolerance):
        """Helper to handle the slow SymPy construction separately."""
        x_sym = sp.sympify(x_str)
        L_sym = sp.Float(self.length)
        
        # Build terms generator
        terms = []
        const_factor = sp.sqrt(2 / L_sym)
        
        for m_idx, c_val in enumerate(coeffs):
            if abs(c_val) > tolerance:
                m = m_idx + 1
                term = sp.Float(c_val) * sp.sin(m * sp.pi * x_sym / L_sym)
                terms.append(term)
        
        return const_factor * sp.Add(*terms)

    @safe_execution
    def wave_function(self, x, t, coefficients, tolerance=1e-15, normalize=True, save_to_file=False):
        """
        Constructs ψ(x, t) efficiently using matrix operations.
        
        Physics:
            ψ(x, t) = Σ c_n * ψ_n(x) * exp(-i * E_n * t)
            
        Optimization:
            - Vectorized over Space (x), Time (t), and Levels (n).
            - Avoids re-integrating normalization at every time step (relies on unitarity).
        """
        # Validation
        if not self.has_eigenvectors_been_calculated:
            raise RuntimeError("Eigenvectors haven't been calculated yet.")
        if len(coefficients) != self.num_levels:
            raise ValueError(f"Expected {self.num_levels} coeffs, got {len(coefficients)}.")

        # 2. Normalize Coefficients Once (Conservation of Probability)
        # Assuming the basis set is orthonormal, we only need to normalize the vector c.
        coeffs = np.array(coefficients, dtype=complex)
        if normalize:
            norm_c = np.linalg.norm(coeffs)
            if not np.isclose(norm_c, 0):
                coeffs = coeffs / norm_c

        # --- Numeric Case ---
        if isinstance(x, (numbers.Number, np.ndarray, list)) and \
           isinstance(t, (numbers.Number, np.ndarray, list)):
            
            x_vals = np.atleast_1d(x)
            t_vals = np.atleast_1d(t)
            
            # A. Pre-calculate Spatial Basis Matrix (Levels x Space)
            psi_space_matrix = np.zeros((self.num_levels, len(x_vals)), dtype=float)
            
            # Only calculate levels with significant coefficients
            active_indices = np.where(np.abs(coeffs) > tolerance)[0]
            
            for i in active_indices:
                eig_vec = self.eigenvectors[i][1] # Get vector part
                psi_space_matrix[i, :] = self.an_eigenfunction(x_vals, eig_vec, tolerance)

            # B. Pre-calculate Temporal Phasors Matrix (Time x Levels)
            energies = np.array([self.eigenvectors[i][0] for i in range(self.num_levels)])
            
            phasors = np.exp(-1j * np.outer(t_vals, energies)) 

            # C. Combine: Psi(t, x) = (Phasors * Coeffs) @ Psi_Space
            # (Num_T, Num_Levels) * (Num_Levels,) -> (Num_T, Num_Levels) weighted
            # (Num_T, Num_Levels) @ (Num_Levels, Num_X) -> (Num_T, Num_X)
            weighted_phasors = phasors * coeffs[None, :] # Broadcast coefficients
            psi_grid = weighted_phasors @ psi_space_matrix

            # D. File I/O (Delegated to helper or inline if strictly necessary)
            if save_to_file:
                self._save_wavefunction(x_vals, t_vals, psi_grid)

            # E. Return shape handling
            return np.squeeze(psi_grid)

        # --- Symbolic Case ---
        else:
            return self._symbolic_wave_function(x, t, coeffs, tolerance)

    def _symbolic_wave_function(self, x, t, coefficients, tolerance):
        """Helper for symbolic construction."""
        x_sym = sp.sympify(x)
        t_sym = sp.sympify(t)
        psi_expr = 0
        
        # Normalize coefficients symbolically
        norm = sp.sqrt(sum(sp.Abs(c)**2 for c in coefficients))
        
        for i, c in enumerate(coefficients):
            if abs(c) > tolerance:
                E = self.eigenvectors[i][0]
                psi_n = self.its_eigenfunction(x_sym, i + 1, tolerance)
                # Note: Physics convention exp(-iEt)
                psi_expr += (c/norm) * psi_n * sp.exp(-sp.I * E * t_sym)
                
        return psi_expr

    @safe_execution
    def probability_density(self, x, t, coefficients, simplify_expr=False, save_to_file=False):
        """
        Calculates |ψ(x,t)|². Efficient wrapper around wave_function.
        """
        psi = self.wave_function(x, t, coefficients, normalize=True, save_to_file=False)

        # Symbolic
        if isinstance(psi, sp.Basic):
            rho = sp.Abs(psi)**2
            return sp.simplify(rho) if simplify_expr else rho
        
        # Numeric
        rho = np.abs(psi)**2
        
        if save_to_file:
            # Reusing the logic
            self._save_density(x, t, rho)
            
        return rho

    def normalize_eigenvector(self, u):
        """
        Normalizes a vector u based on the scalar product defined in the class.
        """
        u = np.asarray(u)
        if u.ndim != 1:
            raise ValueError("Eigenvector must be 1D.")

        # Compute norm squared <u|u>
        norm_sq = self.__scalar_product(u, u)
        
        # Avoid division by zero and sqrt of negative (precision errors)
        if np.isclose(norm_sq, 0, atol=1e-15):
            return u
            
        return u / np.sqrt(np.abs(norm_sq))

    @safe_execution
    def norm_of_wave_function(self, coefficients):
        """
        Returns a function norm_func(t) to check the norm preservation.
        
        Physics Note:
        For a Hermitian Hamiltonian, this value should be constant (approx 1.0) over time.
        Any deviation indicates numerical error or a non-Hermitian system.
        """
        coeffs = np.asarray(coefficients)
        # Pre-normalize coefficients so the expected result is 1.0
        coeffs = coeffs / np.linalg.norm(coeffs)

        def norm_func(t):
            # We integrate |Psi(x,t)|^2 * w(x) dx
            
            def integrand(x_Pos):
                # We calculate density at a single time t for varying x
                rho = self.probability_density(x_Pos, t, coeffs, save_to_file=False)
                # Handle potential weight function
                w = self.weight(x_Pos) if hasattr(self, 'weight') else 1.0
                return rho * w

            result, _ = fixed_quad(integrand, 0.0, self.length, n=5e3)
            return result

        return norm_func

    # --- Helper methods for clean I/O 
    def _save_wavefunction(self, x, t, psi_data):
        # Ensure the directory exists
        if not os.path.exists(self.root_filename):
            os.makedirs(self.root_filename, exist_ok=True)
            
        filename = os.path.join(self.root_filename, "WaveFunction.txt")
        print(f"Saving to {filename}...")
        
        # Create a coordinate grid for each data point
        # T_grid, X_grid = np.meshgrid(t, x, indexing='ij') 
        rows, cols = psi_data.shape # (Time, Space)
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"# {filename}\n# Date: {datetime.datetime.now()}\n")
            f.write("# x\t t\t Real(Psi)\t Imag(Psi)\t Abs(Psi)\n" + "-"*60 + "\n")
            
            # Loop only over time, using block writing or vectorized row-by-row writing
            for j in range(rows): # For each time step
                curr_t = t[j]
                curr_psi = psi_data[j, :] # Entire spatial slice
                
                # Build a temporary matrix to save this time block
                # Colums: x, t, real, imag, abs
                data_block = np.column_stack((
                    x, 
                    np.full_like(x, curr_t), 
                    curr_psi.real, 
                    curr_psi.imag, 
                    np.abs(curr_psi)
                ))
                
                np.savetxt(f, data_block, fmt='%.6f\t%.6f\t%.6e\t%.6e\t%.6e')

    def _save_density(self, x, t, rho_data):
        """
        Implementation of the missing method to save probability density.
        """
        if not os.path.exists(self.root_filename):
            os.makedirs(self.root_filename, exist_ok=True)

        filename = os.path.join(self.root_filename, "ProbabilityDensity.txt")
        print(f"Saving to {filename}...")
        
        rows, cols = rho_data.shape # (Time, Space)

        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"# {filename}\n# Date: {datetime.datetime.now()}\n")
            f.write("# x\t t\t Density(|Psi|^2)\n" + "-"*60 + "\n")
            
            for j in range(rows):
                curr_t = t[j]
                curr_rho = rho_data[j, :]
                
                # Build block: X, T, Rho
                data_block = np.column_stack((
                    x,
                    np.full_like(x, curr_t),
                    curr_rho
                ))
                
                np.savetxt(f, data_block, fmt='%.6f\t%.6f\t%.6e')
    # ---

    # ----- Plot Methods for WaveFunction and Probability Density ---

    @safe_execution
    def plot_eigenfunctions(self, num_levels: int = None, levels: list[int] = None, x_points: int = 800, save: bool = True, save_txt: bool = True):
        """
        Plots the stationary eigenfunctions ψ_n(x) and saves the data.
        """

        # Argument validation
        img_save_path = f"{self.root_filename}/{self.root_filename}_eigenfunctions.png"
        
        if levels is not None:
            idx = levels
        elif num_levels is not None:
            if num_levels < 1: raise ValueError("`num_levels` must be >= 1.")
            idx = list(range(1, num_levels + 1))
        else:
            raise ValueError("Provide `num_levels` or `levels`.")

        # Retrieve (Energy, Vector) pairs
        pairs = self.its_eigenpairs(*idx)
        if not isinstance(pairs, list): 
            pairs = [pairs]

        # Domain definition
        L = getattr(self, "length", None) or getattr(self, "L_optimal", None)
        if L is None: raise AttributeError("Set `self.length` before plotting.")
        
        # Domain vectorization
        x = np.linspace(0.0, L, x_points)

        plt.figure(figsize=(8, 5))
        print(f"--- Processing {len(idx)} levels ---")
        
        # Plotting and saving loop
        for level, (eigval, vec) in zip(idx, pairs):
            psi_complex = self.an_eigenfunction(x, vec)
            psi_n = psi_complex.real # Real part for 1D plotting
            
            n = level - 1 # Physical index (0, 1, 2, ...)
            
            # Plot
            plt.plot(x, psi_n, label=f"$\psi_{{{n}}}$ (E={eigval:.4f})")

            # Save TXT 
            if save_txt:
                txt_filename = f"{self.root_filename}/{self.root_filename}_psi_level_{n}.txt"
                header_info = (f"Level: {n}\nEnergy: {eigval:.8e}\n"
                               f"L_domain: {L:.6f}\nX            Psi(x)")
                np.savetxt(txt_filename, np.column_stack((x, psi_n)), 
                           fmt='%.6e', header=header_info)
                print(f"-> Saved TXT: Level {n}")

        # Aesthetic settings
        plotted_indices = [i - 1 for i in idx]
        plt.title(f"Eigenfunctions $\psi_n(x)$ — Levels {plotted_indices}")
        plt.xlabel("Position $x$ [a.u.]")
        plt.ylabel("$\psi_n(x)$")
        plt.legend(loc="upper left", bbox_to_anchor=(1.00, 1), fontsize='small')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save:
            plt.savefig(img_save_path, dpi=900, bbox_inches="tight")
            print(f"-> Image saved: {img_save_path}")
            
        else: 
            plt.close()

    @safe_execution
    def plot_eigenfunction_3d(self, level: int, x_points: int = 800, y_points: int = 50, show: bool = True, save: bool = True):
        """
        Plots the eigenfunction $\psi_n(x)$ as a 3D surface (extrusion along the Y axis).
        Useful for relief-style visualization of the magnitude.
        """

        save_path = f"{self.root_filename}/{self.root_filename}_eigenfunction3D_level{level}.png"

        # Datas
        eigval, vec = self.its_eigenpairs(level)
        L = getattr(self, "length", None)
        x = np.linspace(0.0, L, x_points)
        psi_n = self.an_eigenfunction(x, vec).real

        # Create the mesh grid
        y = np.linspace(0, 1, y_points)
        X, Y = np.meshgrid(x, y)
        
        # Replicate the wavefunction along the Y axis (dummy axis)
        Z = np.tile(psi_n, (y_points, 1))

        # Plot 3D
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")

        n = level - 1
        ax.set_title(f"Eigenfunction $\psi_{{{n}}}(x)$ — Level {n}")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$ (aux)")
        ax.set_zlabel("$\psi_n(x)$")
        fig.colorbar(surf, shrink=0.5, aspect=5)

        if save:
            plt.savefig(save_path, dpi=900, bbox_inches="tight")
        
        if show: plt.show()
        else: plt.close()

    @safe_execution
    def probability_density_is_plotted(self, t, coefficients, num_frames=None, num_slices=None, save=False, **plotopts):
        """
        Plots the probability density $|\Psi(x,t)|^2$.
        Supports static mode (t = float) or animation mode (t = tuple).
        """

        if len(coefficients) != self.num_levels:
            raise ValueError(f"Expected {self.num_levels} coefficients.")

        num_slices = num_slices or 200
        x_values = np.linspace(0, self.length, num_slices)

        # --- ANIMATION MODE (Time Interval) ---
        if isinstance(t, (tuple, list)) and len(t) == 2:
            num_frames = num_frames or 100
            t1, t2 = t
            times = np.linspace(t1, t2, num_frames)

            fig, ax = plt.subplots(figsize=plotopts.get("figsize", (8, 6)))

            # Compute the entire matrix (num_frames × num_slices) at once or in a fast loop
            print("Computing density frames…")
            y_all_frames = []
            for t_frame in times:
                dens = self.probability_density(x_values, t_frame, coefficients)
                y_all_frames.append(dens)
            
            y_all_frames = np.array(y_all_frames) 
            
            # Physical consistency: define fixed limits based on the global maximum
            global_max = np.max(y_all_frames)
            ax.set_ylim(0, global_max * 1.1) # Margin 10%

            line, = ax.plot(x_values, y_all_frames[0], color=plotopts.get("color", "blue"))
            ax.set_xlabel(plotopts.get("xlabel", "Position $x$"))
            ax.set_ylabel(plotopts.get("ylabel", "$|\Psi(x,t)|^2$"))
            
            title_text = ax.text(0.5, 1.05, "", transform=ax.transAxes, ha="center")

            def update(i):
                line.set_ydata(y_all_frames[i])
                title_text.set_text(f"Probability Density ($t={times[i]:.2f}$)")
                return line, title_text

            ani = animation.FuncAnimation(fig, update, frames=len(times), interval=100, blit=True)

            if save:
                filename = f"{self.root_filename}/{self.root_filename}_prob_density.gif"
                ani.save(filename, writer="pillow", fps=15)
                plt.close(fig)
                print(f"-> Animation saved: {filename}")
            else:
                plt.show()

        # --- STATIC MODE (Single Time) ---
        else:
            density_values = self.probability_density(x_values, t, coefficients)

            plt.figure(figsize=plotopts.get("figsize", (8, 6)))
            plt.plot(x_values, density_values, color=plotopts.get("color", "blue"))
            plt.xlabel(plotopts.get("xlabel", "Position $x$"))
            plt.ylabel(plotopts.get("ylabel", "$|\Psi(x,t)|^2$"))
            plt.title(plotopts.get("title", f"Probability Density ($t={t}$))"))
            plt.grid(True)
            plt.ylim(bottom=0) # Density is never negative

            if save:
                filename = f"{self.root_filename}/{self.root_filename}_prob_density_t{t}.png"
                plt.savefig(filename, dpi=900)
                plt.close()
                print(f"-> Static plot saved: {filename}")
            else:
                plt.show()

    @safe_execution
    def probability_density_3d(self, t_interval, coefficients, num_frames=50, num_slices=200, animate_rotation=False, save_as_gif=True, gif_filename=None, **plotopts):
        """
        Generates a 3D plot of the time evolution of the probability density.
        Axes: X (Position), Y (Time), Z (Density).
        """

        if gif_filename is None:
            gif_filename = f"{self.root_filename}/{self.root_filename}_probability_density_3d.gif"

        t1, t2 = t_interval
        t_values = np.linspace(t1, t2, num_frames)
        x_values = np.linspace(0, self.length, num_slices)

        # Meshgrid (Time x Position)
        T, X = np.meshgrid(t_values, x_values, indexing='ij')

        # Fill the density matrix
        print("Computing 3D surface...")
        density = np.empty_like(T, dtype=float)
        
        for i, t_val in enumerate(t_values):
            # Spatial vectorization for each time step
            density[i, :] = self.probability_density(x_values, t_val, coefficients)

        # Setup 3D Figure
        figsize = plotopts.get("figsize", (10, 8))
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        cmap = plotopts.get("cmap", "viridis")
        surf = ax.plot_surface(X, T, density, cmap=cmap, edgecolor='none')

        ax.set_xlabel(plotopts.get("xlabel", "Position $x$"))
        ax.set_ylabel(plotopts.get("ylabel", "Time $t$"))
        ax.set_zlabel(plotopts.get("zlabel", "$|\Psi(x,t)|^2$"))
        ax.set_title(plotopts.get("title", "Time Evolution of Probability Density"))
        fig.colorbar(surf, shrink=0.5, aspect=5)

        # Camera rotation animation
        if animate_rotation or save_as_gif:
            def update_view(frame):
                ax.view_init(elev=30, azim=frame)
                return [surf]

            # Full rotation (360 degrees)
            angle_frames = np.linspace(0, 360, 60) 
            ani = animation.FuncAnimation(fig, update_view, frames=angle_frames, interval=100, blit=False)

            if save_as_gif:
                ani.save(gif_filename, writer="pillow", fps=15)
                plt.close(fig)
                print(f"-> 3D GIF saved: {gif_filename}")
            else:
                plt.show()
        else:
            plt.show()

        return fig, ax, density, t_values, x_values

    @safe_execution
    def plot_wavefunction_and_density(self, t, coefficients, num_frames=None, num_slices=None, save_as_gif=False, gif_filename=None, **plotopts):
        """
        Plots, side by side:
        1. The wavefunction magnitude $|\Psi(x,t)|$
        2. The probability density $|\Psi(x,t)|^2$
        """

        if gif_filename is None:
            gif_filename = f"{self.root_filename}/{self.root_filename}_wavefunction_density.gif.gif"

        num_slices = num_slices or 200
        x_values = np.linspace(0, self.length, num_slices)

        # Helper: assumes wave_function is vectorized in x
        def get_data(t_val):
            psi = self.wave_function(x_values, t_val, coefficients)
            return np.abs(psi), np.abs(psi)**2

        # --- ANIMATION ---
        if isinstance(t, (tuple, list)) and len(t) == 2:
            num_frames = num_frames or 100
            times = np.linspace(t[0], t[1], num_frames)
            
            # Precompute for smoother results
            print("Computing comparative frames…")
            mod_all = []
            dens_all = []
            for t_val in times:
                m, d = get_data(t_val)
                mod_all.append(m)
                dens_all.append(d)
            
            mod_all = np.array(mod_all)
            dens_all = np.array(dens_all)
            
            # Fixed global limits
            max_mod = np.max(mod_all)
            max_dens = np.max(dens_all)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=plotopts.get("figsize", (12, 5)))
            
            line1, = ax1.plot(x_values, mod_all[0], label="$|\Psi|$")
            line2, = ax2.plot(x_values, dens_all[0], label="$|\Psi|^2$", color="orange")
            
            ax1.set_ylim(0, max_mod * 1.1)
            ax2.set_ylim(0, max_dens * 1.1)
            
            ax1.set_title("Wavefunction Modulus $|\Psi|$")
            ax2.set_title("Probability Density $|\Psi|^2$")
            
            for ax in (ax1, ax2):
                ax.set_xlabel("$x$")
                ax.grid(True)
            
            suptitle = fig.suptitle(f"t = {times[0]:.2f}")

            def update(i):
                line1.set_ydata(mod_all[i])
                line2.set_ydata(dens_all[i])
                suptitle.set_text(f"t = {times[i]:.2f}")
                return line1, line2, suptitle

            ani = animation.FuncAnimation(fig, update, frames=len(times), interval=100, blit=True)

            if save_as_gif:
                ani.save(gif_filename, writer="pillow", fps=15)
                plt.close(fig)
                print(f"-> GIF saved: {gif_filename}")
            else:
                plt.show()

        # --- STATIC ---
        else:
            mod_vals, dens_vals = get_data(t)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=plotopts.get("figsize", (12, 5)))
            
            ax1.plot(x_values, mod_vals, label="$|\Psi|$")
            ax2.plot(x_values, dens_vals, label="$|\Psi|^2$", color="orange")
            
            ax1.set_title(f"Modulus $|\Psi|$ ($t={t:.2f}$)")
            ax2.set_title(f"Density $|\Psi|^2$ ($t={t:.2f}$)")
            
            for ax in (ax1, ax2):
                ax.set_xlabel("$x$")
                ax.grid(True)
                ax.set_ylim(bottom=0)
            
            if save_as_gif: 
                # Save as PNG in static mode, even if save_as_gif is True
                png_name = gif_filename.replace(".gif", ".png")
                plt.savefig(png_name, dpi=900)
                plt.close(fig)
                print(f"-> Image saved: {png_name}")
            else:
                plt.show()

    @safe_execution
    def probability_density_cartoon(self, t, coefficients, num_slices=None, num_frames=None, save_as_gif=True, gif_filename=None, gif_fps=90, cmap='plasma', **plotopts):
        """
        Displays or saves a "cartoon" (2D heatmap) of the probability density.
        Useful for visualizing particle localization as a "film strip."

        Parameters
        ----------
        t : float or tuple
            - If float: generates a static plot.
            - If tuple (t0, t1): generates an animation.
        coefficients : list
            Wavefunction coefficients.
        num_slices : int, optional
            Spatial resolution (number of points along the x-axis). Default: 500.
        num_frames : int, optional
            Temporal resolution (animation only). Default: 100.
        save_as_gif : bool
            If True, saves the output to a file.
        gif_filename : str
            Output filename.
        **plotopts : dict
            Additional arguments passed to `plt.imshow`.
        """
        
        if gif_filename is None:
            gif_filename = f"{self.root_filename}/{self.root_filename}_density_cartoon.gif"
            
        resolution = num_slices or 500
        x = np.linspace(0, self.length, resolution)
        
        # Auxiliary function to generate the 2D matrix (vertical extrusion)
        def make_image(t_val):
            # Density is positive. The colormap handles the visual intensity.
            dens = self.probability_density(x, t_val, coefficients)
            # Repeat the row 50 times to create a visible "band"
            return np.tile(dens, (50, 1))

        # Extent to map pixels to real coordinates (x: 0->L, y: 0->1)
        extent = [0, self.length, 0, 1]
        
        # STATIC
        if isinstance(t, (int, float)):
            # Compute data
            img_data = make_image(t)
            
            fig, ax = plt.subplots(figsize=plotopts.get("figsize", (8, 6)))
            
            # vmin=0 ensures zero corresponds to the background/cold color
            im = ax.imshow(img_data, extent=extent, aspect="auto", cmap=cmap, 
                           vmin=0, **plotopts)
                           
            ax.set_xlabel("Position $x$")
            ax.set_yticks([]) # Remove Y axis since it is artificial
            ax.set_title(f"Density Cartoon ($t={t:.3f}$)")
            fig.colorbar(im, ax=ax, label="$|\Psi|^2$")
            
            if save_as_gif:
                # Save as PNG in static mode
                static_name = gif_filename.replace(".gif", ".png")
                fig.savefig(static_name, dpi=900, bbox_inches='tight')
                plt.close(fig)
                print(f"-> Static cartoon saved: {static_name}")
            else:
                plt.show()

        # ANIMATION
        elif isinstance(t, (tuple, list)) and len(t) == 2:
            t0, t1 = t
            frames_count = num_frames or 100
            times = np.linspace(t0, t1, frames_count)

            # A. Precomputation for global normalization
            print("Calculando máximo global para escala de cores...")
            max_density_global = 0.0
            # Quick sampling to find the maximum (check t0, midpoint, and t1)
            check_times = np.linspace(t0, t1, min(10, frames_count))
            for check_t in check_times:
                d_check = self.probability_density(x, check_t, coefficients)
                current_max = np.max(d_check)
                if current_max > max_density_global:
                    max_density_global = current_max
            
            # Add a safety margin
            vmax = max_density_global * 1.05

            # B. Plot setup
            fig, ax = plt.subplots(figsize=plotopts.get("figsize", (8, 6)))
            
            # Initial frame
            img0 = make_image(times[0])
            im = ax.imshow(img0, extent=extent, aspect="auto", cmap=cmap,
                           vmin=0, vmax=vmax, **plotopts)
            
            ax.set_xlabel("Position $x$")
            ax.set_yticks([])
            title = ax.set_title(f"Density Cartoon ($t={times[0]:.3f}$)")
            # Fixed colorbar
            fig.colorbar(im, ax=ax, label="$|\Psi|^2$")

            def update(frame_idx):
                t_curr = times[frame_idx]
                im.set_data(make_image(t_curr))
                title.set_text(f"Density Cartoon ($t={t_curr:.3f}$)")
                return (im, title)

            ani = animation.FuncAnimation(
                fig, update, frames=len(times), 
                interval=1000/gif_fps, blit=True
            )

            if save_as_gif:
                ani.save(gif_filename, writer="pillow", fps=gif_fps)
                plt.close(fig)
                print(f"-> Cartoon GIF saved: {gif_filename}")
            else:
                plt.show()

        else:
            raise ValueError("t must be a number or a tuple/list [t0, t1].")  
        
    @safe_execution
    def plot_wavefunction_snapshot(self, t: float, coefficients, num_slices: int = 1000, 
                                    save_data: bool = False, data_filename: str = None,
                                    plot_filename: str = None,
                                    **plotopts):
            """
            Plots the wavefunction at a fixed time t.
            Allows saving both the figure and the data (.txt).
            """
            import datetime 

            # Data preparation
            x_values = np.linspace(0, self.length, num_slices)
            psi_values = self.wave_function(x_values, t, coefficients)
            
            mod_psi = np.abs(psi_values)
            prob_density = mod_psi**2
            real_part = np.real(psi_values)

            # Plotting
            fig, ax = plt.subplots(figsize=plotopts.get("figsize", (10, 6)))
            
            # Magnitude (Solid black line)
            ax.plot(x_values, mod_psi, 'k-', linewidth=2, label=r'$|\Psi(x,t)|$')

            ax.set_title(f"Wavefunction Snapshot at t = {t:.4f}")
            ax.set_xlabel("Position x")
            ax.set_xlim(0, self.length)
            ax.legend(loc='upper right')
            ax.grid(True, linestyle=':', alpha=0.6)
            
            plt.tight_layout()

            # Save IMAGE 
            if save_data:
                if plot_filename is None:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    plot_filename = f"{self.root_filename}/{self.root_filename}_plot_t{t:.2f}_{timestamp}.png"
                
                fig.savefig(plot_filename, dpi=900, bbox_inches='tight')
                print(f"Plot image saved: {plot_filename}")

            # Display on screen
            plt.show()

            # Save txt data
            if save_data:
                if data_filename is None:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    data_filename = f"{self.root_filename}/{self.root_filename}_data_t{t:.2f}_{timestamp}.txt"
                
                print(f"Exporting wavefunction data to: {data_filename} ...")
                try:
                    with open(data_filename, "w", encoding='utf-8') as f:
                        f.write(f"# Wavefunction Data for t={t}, L={self.length}\n")
                        f.write(f"# Column 1: x (Position)\n")
                        f.write(f"# Column 2: Psi (Complex)\n")
                        f.write(f"# Column 3: |Psi|^2 (Density)\n")
                        f.write(f"# -----------------------------------\n")
                        for x, psi, dens in zip(x_values, psi_values, prob_density):
                            f.write(f"{x:.8e}\t{psi:.8e}\t{dens:.8e}\n")
                    print("-> Data export successful.")
                except Exception as e:
                    print(f"Error exporting data: {e}")

    @safe_execution
    def plot_wavefunction_3d(self, t_span: tuple, coefficients, num_x: int = 100, num_t: int = 100, save_data: bool = False, data_filename: str = None,
                                plot_filename: str = None, elev=30, azim=-45, **plotopts):
            """
            Plots the TIME EVOLUTION of the wavefunction magnitude in 3D.

            Axes:
            X: Position
            Y: Time (t)
            Z: Magnitude |Psi(x,t)|

            Args:
                t_span (tuple): Time interval (t_start, t_end).
                coefficients: Expansion coefficients.
                num_x (int): Spatial resolution.
                num_t (int): Temporal resolution.
            """
            import datetime
            from mpl_toolkits.mplot3d import Axes3D
            from matplotlib import cm

            t_start, t_end = t_span
            
            # Grid generation
            x = np.linspace(0, self.length, num_x)
            t = np.linspace(t_start, t_end, num_t)
            X, T = np.meshgrid(x, t)  # Build the coordinate matrix
            
            # Compute the Z matrix (|Psi|)
            Z = np.zeros_like(X)
            
            print(f"Computing time evolution from t={t_start} to t={t_end}...")
            
            # Time loop (matrix rows)
            for i, t_val in enumerate(t):
                psi_vals = self.wave_function(x, t_val, coefficients)
                Z[i, :] = np.abs(psi_vals)  # Store magnitude values
                
            # Plot setup
            fig = plt.figure(figsize=plotopts.get("figsize", (12, 9)))
            ax = fig.add_subplot(111, projection='3d')
            
            # Clean layout for publication
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.grid(True, linestyle=':', color='gray', alpha=0.3)
            
            # Surface rendering
            surf = ax.plot_surface(X, T, Z, cmap='viridis', edgecolor='none', alpha=0.9, antialiased=True)
            
            # Add contour projections on the base plane for better magnitude interpretation
            ax.contourf(X, T, Z, zdir='z', offset=0, cmap='viridis', alpha=0.3)

            # Axis and label adjustments
            ax.set_xlabel(r'Position $x$', fontsize=11, labelpad=10)
            ax.set_ylabel(r'Time $t$', fontsize=11, labelpad=10)
            ax.set_zlabel(r'$|\Psi(x,t)|$', fontsize=11, labelpad=10)
            ax.set_title(f"Time Evolution of Wavefunction Modulus\nInterval: [{t_start}, {t_end}]", fontsize=14)
            
            ax.set_xlim(0, self.length)
            ax.set_ylim(t_start, t_end)
            ax.set_zlim(0, Z.max() * 1.1) # Adjust z-limits to prevent peak clipping
            
            ax.view_init(elev=elev, azim=azim)
            ax.invert_xaxis()
            
            # Lateral colorbar
            cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
            cbar.set_label(r'$|\Psi|$ Magnitude')

            plt.tight_layout()

            # Save image
            if save_data:
                if plot_filename is None:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    plot_filename = f"{self.root_filename}/{self.root_filename}_Evol3D_{timestamp}.png"
                
                print(f"Saving Evolution 3D plot to: {plot_filename} ...")
                plt.savefig(plot_filename, dpi=900, bbox_inches='tight')

            plt.show()

            # Data export (Long Matrix: X, T, Z)
            if save_data:
                if data_filename is None:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    data_filename = f"{self.root_filename}/{self.root_filename}_EvolData_{timestamp}.txt"
                
                print(f"Exporting Evolution data to: {data_filename} ...")
                try:
                    with open(data_filename, "w", encoding='utf-8') as f:
                        f.write(f"# Wavefunction Evolution Data (Surface)\n")
                        f.write(f"# T_start={t_start}, T_end={t_end}, Nx={num_x}, Nt={num_t}, L={self.length}\n")
                        f.write(f"# Column 1: Time (t)\n")
                        f.write(f"# Column 2: Position (x)\n")
                        f.write(f"# Column 3: Modulus |Psi(x,t)|\n")
                        f.write(f"# --------------------------------------------------\n")
                        
                        for i in range(num_t):
                            for j in range(num_x):
                                # T[i,j], X[i,j], Z[i,j]
                                f.write(f"{T[i,j]:.6e}\t{X[i,j]:.6e}\t{Z[i,j]:.6e}\n")
                                
                    print(" Data export successful.")
                except Exception as e:
                    print(f"Error exporting data: {e}")
    
    @safe_execution
    def plot_probability_density_3d_static(self, t_interval: tuple, coefficients, num_t: int = 100, num_x: int = 200,  save_data: bool = False, data_filename: str = None, plot_filename: str = None, elev=30, azim=135,                                        **plotopts):
        """
        Generates a STATIC 3D figure of the time evolution of the probability density.

        Args:
            t_interval: (t_start, t_end)
            coefficients: Expansion coefficients.
            num_t: Temporal resolution.
            num_x: Spatial resolution.
        """

        import datetime
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm

        t1, t2 = t_interval
        
        # Grid generation
        x_values = np.linspace(0, self.length, num_x)
        t_values = np.linspace(t1, t2, num_t)
        
        X, T = np.meshgrid(x_values, t_values)
        Z = np.zeros_like(X) # Density matrix construction
        
        print(f"Computing density surface ({num_t}x{num_x})...")
        
        # Fill the matrix
        for i, t_val in enumerate(t_values):
            # Compute |Psi|^2 over the full spatial domain at time t
            dens_vals = self.probability_density(x_values, t_val, coefficients)
            Z[i, :] = dens_vals

        # Figure setup
        figsize = plotopts.get("figsize", (12, 9))
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Remove pane background colors
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.grid(True, linestyle=':', color='gray', alpha=0.3)

        # Surface rendering
        cmap = plotopts.get("cmap", "viridis") 
        surf = ax.plot_surface(X, T, Z, cmap=cmap, edgecolor='none', alpha=0.9, antialiased=True)
        
        # Add contour projection onto the base plane (z=0) for better interpretation
        ax.contourf(X, T, Z, zdir='z', offset=0, cmap=cmap, alpha=0.3)

        # Axis and label configuration
        ax.set_xlabel(plotopts.get("xlabel", "Position $x$"), labelpad=10)
        ax.set_ylabel(plotopts.get("ylabel", "Time $t$"), labelpad=10)
        ax.set_zlabel(plotopts.get("zlabel", r"Density $|\Psi(x,t)|^2$"), labelpad=10)
        ax.set_title(plotopts.get("title", f"Density Evolution t=[{t1}, {t2}]"), fontsize=14)
        
        ax.set_xlim(0, self.length)
        ax.set_ylim(t1, t2)
        ax.set_zlim(0, Z.max() * 1.1)

        # X-axis inversion
        ax.invert_yaxis()

        # Camera/view adjustment
        ax.view_init(elev=elev, azim=azim)

        # Colorbar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
        cbar.set_label(r'Probability Density')

        plt.tight_layout()

        # Save Image
        if save_data:
            if plot_filename is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                plot_filename = f"{self.root_filename}/{self.root_filename}_Dens3D_Static_{timestamp}.png"
            
            print(f"Saving 3D Density plot to: {plot_filename} ...")
            plt.savefig(plot_filename, dpi=900, bbox_inches='tight')

        plt.show()

        # Export Data
        if save_data:
            if data_filename is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                data_filename = f"{self.root_filename}/{self.root_filename}_Dens3D_Data_{timestamp}.txt"
            
            print(f"Exporting Density data to: {data_filename} ...")
            try:
                with open(data_filename, "w", encoding='utf-8') as f:
                    f.write(f"# 3D Probability Density Data\n")
                    f.write(f"# T_start={t1}, T_end={t2}, Nx={num_x}, Nt={num_t}, L={self.length}\n")
                    f.write(f"# Column 1: Time (t)\n")
                    f.write(f"# Column 2: Position (x)\n")
                    f.write(f"# Column 3: Density |Psi|^2\n")
                    f.write(f"# --------------------------------------------------\n")
                    
                    # mesh iteration
                    for i in range(num_t):
                        for j in range(num_x):
                            # T[i,j] -> time, X[i,j] -> position, Z[i,j] -> density
                            f.write(f"{T[i,j]:.6e}\t{X[i,j]:.6e}\t{Z[i,j]:.6e}\n")
                            
                print("Data export successful.")
            except Exception as e:
                print(f"Error exporting data: {e}")
        
        return fig, ax

    # ----- Methods for Calculate Position -----
    @safe_execution
    def expected_position(self, coefficients):
        """
            Calculation:  
            For a given instant t, it computes:  

                ⟨x⟩(t) = ∫₀ᴸ x · w(x) · |ψ(x,t)|² dx,  

            where ψ(x,t) is the wave function generated using the given coefficients.  

            Args:  
            - coefficients (list): List of coefficients (number of elements must equal num_levels).  

            Returns:  
            - exp_pos (function): A function that takes an instant t and returns the expected position.  
        """
        if len(coefficients) != self.num_levels:
            raise ValueError(
                f"The wave function is a superposition of {self.num_levels} eigenfunctions, but got {len(coefficients)} coefficients."
            )
        
        n_points = 5000
        def exp_pos(t):
            # Defines the integrand: x * w(x) * ProbabilityDensity(x, t, coefficients)`
            integrand_num = lambda x: self.probability_density(x, t, coefficients)*x*self.weight(x)
            numerator, err1 = fixed_quad(integrand_num, 0.0, self.length, n=n_points)
            
            # Defines the integrand: `∫₀ᴸ [w(x) * |ψ(x,t)|²]`
            integrand_den = lambda x: self.probability_density(x, t, coefficients)*self.weight(x)
            denominator, err1 = fixed_quad(integrand_den, 0.0, self.length, n=n_points)
            

            if denominator == 0:
                raise ZeroDivisionError("Normalization denominator is zero. Check coefficients or wavefunction.")

            return numerator/denominator 
        
        return exp_pos
    
    @safe_execution
    def expected_position_squared(self, coefficients):
        """	
            Returns a function that calculates ⟨x²⟩(t):  

                ⟨x²⟩(t) = ∫₀ᴸ [x² * w(x) * |ψ(x,t)|²] dx. 

            where ψ(x,t) is the wave function generated with the given coefficients.  

            Returns:  
            - exp_x2 (function): A function that takes an instant t and returns the expected value ⟨x²⟩(t).  
        """
        if len(coefficients) != self.num_levels:
            raise ValueError(f"The wave function is a superposition of {self.num_levels} eigenfunctions, but got {len(coefficients)} coefficients.")

        n_points = 5000
        def exp_pos_sq(t):
            # Defines the integrand: x^2 * w(x) * ProbabilityDensity(x, t, coefficients)`
            integrand_num = lambda x: self.probability_density(x, t, coefficients)*(x**2)*self.weight(x)
            numerator, err1 = fixed_quad(integrand_num, 0.0, self.length, n=n_points)
          
            # Defines the integrand: ∫₀ᴸ [w(x) * |ψ(x,t)|²]
            integrand_den = lambda x: self.probability_density(x, t, coefficients)*self.weight(x)
            denominator, err1 = fixed_quad(integrand_den, 0.0, self.length, n=n_points)
            
            return numerator/denominator
        
        return exp_pos_sq

    @safe_execution
    def position_uncertainty(self, coefficients):
        """
            Returns a function that calculates the position uncertainty, defined as:  

                σₓ(t) = sqrt(⟨x²⟩(t) - ⟨x⟩(t)²)

            for the wave packet given by the coefficients.  

            Args:  
            - coefficients (list): List of coefficients (length must be self.num_levels).  

            Returns:  
            - uncertainty (function): A function that takes t and returns the uncertainty in position.  
        """
        if len(coefficients) != self.num_levels:
            raise ValueError(f"The wave function is a superposition of {self.num_levels} eigenfunctions, but got {len(coefficients)} coefficients.")
        
        exp_pos_fn = self.expected_position(coefficients)
        exp_pos_sq_fn = self.expected_position_squared(coefficients)
        
        def uncertainty(t):
            aux1 = exp_pos_sq_fn(t)
            aux2 = exp_pos_fn(t)
            return np.sqrt(max(aux1 - aux2**2, 0.0))

        return uncertainty
    
    @safe_execution
    def position_uncertainty_relative(self, coefficients, t_max=None, num_points=300, save_filename=None):
        """
        Computes the relative position uncertainty r_x(t) = sigma_x(t) / <x>(t)
        and records the times at which it reaches its minimum values.

        If t_max is None, returns the callable function r_x(t).
        Otherwise, it evaluates r_x(t) over [0, t_max], prints and saves
        the minimum times, saves all sampled points, and plots the graph.
        
        Args:
            coefficients (List[complex]): expansion coefficients (length = self.num_levels)
            t_max (float, optional): maximum time for evaluation and plotting.
                If None, returns the r_x(t) function without plotting.
            num_points (int): number of time samples (default = 200)
            save_filename (str, optional): output filename for minimum times.
                If None, defaults to "{self.root_filename}_min_relative_uncertainty.txt"

        Returns:
            If t_max is None:
                Callable[[float], float]: the relative uncertainty function r_x(t)
            Else:
                Tuple[matplotlib.figure.Figure, float, List[float]]: 
                    (figure, minimum value, list of times at the minimum)

        Example:
            # (A) Get the function and evaluate at t = 2.5
            r_fn = system.position_uncertainty_relative(coeffs)
            print(f"r_x(2.5) = {r_fn(2.5):.4f}")

            # (B) Plot and save minima over [0, 10] with 300 samples
            fig, min_val, min_times = system.position_uncertainty_relative(
                coeffs,
                t_max=10.0,
                num_points=300,
                save_filename="min_relative_uncertainty.txt"
            )
        """

        # Validate coefficients
        if len(coefficients) != self.num_levels:
            raise ValueError(f"Expected {self.num_levels} coefficients, got {len(coefficients)}.")

        # Determine output filename if not provided
        if save_filename is None:
            save_filename = f"{self.root_filename}/{self.root_filename}_min_relative_uncertainty.txt"

        # Retrieve helper functions for ⟨x⟩(t) and σₓ(t)
        exp_pos_fn = self.expected_position(coefficients)
        sigma_fn   = self.position_uncertainty(coefficients)

        # Define the relative uncertainty function r_x(t)
        def r_x(t: float) -> float:
            mean = exp_pos_fn(t)
            sigma = sigma_fn(t)
            if mean == 0:
                return np.inf
            return sigma / mean

        # If no plotting requested, return the callable
        if t_max is None:
            return r_x

        # Sample r_x on the time grid
        t_vals = np.linspace(0, t_max, num_points)
        y_vals = np.array([r_x(t) for t in t_vals])

        # Identify the minimum value and corresponding times
        min_val = y_vals.min()
        tol = 1e-15
        min_times = [
            t for t, y in zip(t_vals, y_vals)
            if np.isclose(y, min_val, atol=tol, rtol=tol)
        ]

        # Print results
        print(f"Minimum value of dx(t) / <x>(t): {min_val:.15f}")
        print("Estimated period(s): ")
        for t in min_times:
            print(f"t = {t:.15f}")

        # Save results to file 
        with open(save_filename, 'w', encoding='utf-8') as f:
            f.write(f"# {save_filename}\n")
            f.write(f"# Program version : {self.version}\n")
            f.write(f"# Minimum value of dx(t) / <x>(t): {min_val:.15f}\n")
            f.write("# Estimated period(s): \n")
            for t in min_times:
                f.write(f"# t = {t:.15f}\n")
            f.write("# " + "="*80 + "\n")
            f.write("# (1) time, (2) r_x(t)\n")
            for tt, yy in zip(t_vals, y_vals):
                f.write(f"{tt:.6f}\t{yy:.6f}\n")

        print("Full results saved at:", save_filename)

        # Plot the relative uncertainty curve
        fig, ax = plt.subplots()
        ax.plot(t_vals, y_vals, lw=2, color='black')
        ax.set_title("Relative Position Uncertainty r_x(t)")
        ax.set_xlabel("Time t")
        ax.set_ylabel("dx(t)/<x>(t)")
        ax.grid(True)

        # Highlight the minimum points
        ax.plot(min_times, [min_val]*len(min_times), 'ro', label='Minimum point(s)')
        ax.legend(loc="upper left", bbox_to_anchor=(1.00, 1))
        plt.tight_layout()

        # Save the plot
        fig.savefig(f"{self.root_filename}/{self.root_filename}_RelativePositionUncertainty.png", bbox_inches='tight', dpi=2000)
        plt.show()

        return fig, min_val, min_times
   
    @safe_execution
    def expected_position_is_calculated(self, t_range, coefficients, num_points=None):
        """
            Receives a time interval (t_range as (t0, t1)) and a list of coefficients, and generates a file with two columns:  
            (1) Time  
            (2) Expected position  

            Behavior:  
            - If the parameter num_points is provided (must be a positive integer), it sets the number of intermediate points.  
            - Otherwise, 300 points will be used by default.  
        """
        if len(coefficients) != self.num_levels:
            raise ValueError(f"Expected {self.num_levels} coefficients, but got {len(coefficients)}.")
        # If num_points is provided and is a positive integer, use it; otherwise, use 300.
        if num_points is not None:
            if isinstance(num_points, int) and num_points > 0:
                nt = num_points
                print(f"Calculating {nt} points.")
            else:
                raise ValueError(f"Third argument must be a positive integer, got {num_points}.")
        else:
            print("Calculating 300 points by default.")
            nt = 300
        t0, t1 = t_range
        dt = (t1 - t0) / nt
        t_points = [t0 + k * dt for k in range(nt + 1)]
        ff = self.expected_position(coefficients)  # f(t) = ⟨x⟩(t)
        # Defines the file name (using the root defined in the root_filename property).
        filename = f"{self.root_filename}/{self.root_filename}_Position.txt"
        
        # Writes the header
        with open(filename, "w", encoding='utf-8') as f:
            f.write(f"# {filename}\n")
            f.write(f"# Program version : {self.version}\n")
            f.write(f"# Start: {datetime.datetime.now().strftime('%c')}\n")
            f.write("# (1) time, (2) expected position\n")
            f.write("# " + "="*80 + "\n")
        # Now, add the data (one line per time point).
        with open(filename, "a", encoding='utf-8') as f:
            for tt in t_points:
                # Equivalent `evalf` usage: converts to float with standard precision.
                f.write(f"{float(tt):.6f}\t{float(ff(tt)):.6f}\n")
        print("Data on expected position has been stored in file:", filename)
    
    @safe_execution
    def expected_position_and_uncertainty_are_calculated(self, t_range, coefficients, num_points=None):
        """
            Receives a time interval (t_range as (t0, t1)) and a list of coefficients,  
            and generates a file with four columns:  
            (1)Time  
            (2)Expected position  
            (3)Expected position – uncertainty  
            (4)Expected position + uncertainty  

            Behavior:  
            - If the parameter num_points is provided (must be a positive integer), it sets the number of intermediate points.  
            - Otherwise, 300 points will be used by default.  
        """
        if len(coefficients) != self.num_levels:
            raise ValueError(f"Expected {self.num_levels} coefficients, but got {len(coefficients)}.")
        if num_points is not None:
            if isinstance(num_points, int) and num_points > 0:
                nt = num_points
            else:
                raise ValueError(f"Third argument must be a positive integer, got {num_points}.")
        else:
            print("Calculating 300 points by default.")
            nt = 300
        t0, t1 = t_range
        dt = (t1 - t0) / nt
        t_points = [t0 + k * dt for k in range(nt + 1)]
        ff = self.expected_position(coefficients)  # f(t) = ⟨x⟩(t)
        gg = self.position_uncertainty(coefficients)  # g(t) = σₓ(t)
        # Auxiliary function to ensure the value is real: if not, returns 0.
        def ifnotreal(func, t_val):
            value = func(t_val)
            if np.isreal(value):
                return value
            else:
                return 0.0
        filename = f"{self.root_filename}/{self.root_filename}_PositionUncertainty.txt"
        with open(filename, "w", encoding='utf-8') as f:
            f.write(f"# Filename :          {filename}\n")
            f.write(f"# Program version :   {self.version}\n")
            f.write(f"# Start:              {datetime.datetime.now().strftime('%c')}\n")
            f.write("# (1) time,\t(2) expected position,\t(3) expected position - uncertainty,\t(4) expected position + uncertainty\n")
            f.write("# " + "="*80 + "\n")
        with open(filename, "a", encoding='utf-8') as f:
            for tt in t_points:
                ex_pos = ff(tt)
                uncert = ifnotreal(gg, tt)
                f.write(f"{tt:.6f}\t{ex_pos:.6f}\t{(ex_pos - uncert):.6f}\t{(ex_pos + uncert):.6f}\n")
        print("Expected position and uncertainty data stored in file:", filename)

    @safe_execution    
    def expected_position_is_plotted(self):
        """
            Reads a file containing expected position data and returns a plot.

            Behavior:
            - First, it looks for a file named <root_filename>_Position.
            - If not found, it tries <root_filename>_PositionUncertainty.
            - The data is expected to have two columns: time and expected position.
            - Finally, it plots the curve in black.
        """
        # Defines the initial filename.
        filename = f"{self.root_filename}/{self.root_filename}_Position.txt"
        if not os.path.exists(filename):
            # If it doesn't exist, try the file with uncertainty.
            filename = f"{self.root_filename}_PositionUncertainty"
            if not os.path.exists(filename):
                raise FileNotFoundError("Data file does not exist. Make sure the function 'expected_position_and_uncertainty_are_calculated()' is called before proceeding. ")
        try:
            # Reads the data, assuming the file has two columns.  
            # If the data is separated by spaces or tabs, this will work.
            data = np.loadtxt(filename)
        except Exception as e:
            raise RuntimeError(f"Error reading the file {filename}: {e}")
        
        # If the data has at least two columns:
        if data.ndim == 1:
            # If there is only one row, force it to have two columns.
            data = data.reshape(1, -1)
        if data.shape[1] < 2:
            raise ValueError("Expected data with at least 2 columns (time and expected position).")
        
        time_vals = data[:, 0]
        expected_vals = data[:, 1]
        
        plt.figure(constrained_layout=True)
        plt.plot(time_vals, expected_vals, color='black', label="⟨x⟩")
        plt.xlabel("Time")
        plt.ylabel("Expected Position")
        plt.title("Expected Position")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.root_filename}/{self.root_filename}_ExpectedPosition.png", bbox_inches='tight', dpi=2000)
        plt.show()
    
    @safe_execution
    def expected_position_and_uncertainty_are_plotted(self):
        """
            Reads a file containing expected position and uncertainty data, then returns a plot.

            The file should have 4 columns:  
            - (1) Time  
            - (2) Expected position  
            - (3) Expected position - uncertainty  
            - (4) Expected position + uncertainty  

            If the file has only 2 columns, it is assumed that uncertainty data was not generated.  
            Then, the three curves are plotted simultaneously:  
            - Uncertainty regions are shaded in gray  
            - Central curve (expected position) is plotted in black  

        """
        filename = f"{self.root_filename}/{self.root_filename}_PositionUncertainty.txt"
        if not os.path.exists(filename):
            raise FileNotFoundError("Data file does not exist.")
        try:
            data = np.loadtxt(filename)
        except Exception as e:
            raise RuntimeError(f"Error reading file {filename}: {e}")
        
        # If the read data has only 2 columns, raise an error.
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] == 2:
            raise ValueError("You must be trying to read uncertainties from a file that only has information on the expected value of the position: " + filename)
        if data.shape[1] < 4:
            raise ValueError("Data file must contain 4 columns: time, expected position, expected position - uncertainty and expected position + uncertainty.")
        
        time_vals = data[:, 0]
        expected_vals = data[:, 1]
        minus_vals = data[:, 2]
        plus_vals = data[:, 3]
        
        plt.figure(constrained_layout=True)
        plt.plot(time_vals, minus_vals, color='gray', label="<x> - σₓ")
        plt.plot(time_vals, expected_vals, color='black', label="<x>")
        plt.plot(time_vals, plus_vals, color='gray', label="<x> + σₓ")
        plt.xlabel("Time")
        plt.ylabel("Position")
        plt.title("Expected Position and Uncertainty")
        plt.grid(True)
        plt.legend(loc="upper left", bbox_to_anchor=(1.00, 1))
        plt.tight_layout()
        plt.savefig(f"{self.root_filename}/{self.root_filename}_ExpectedPositionAndUncertainty.png", bbox_inches='tight', dpi=2000)
        plt.show()
    
    @safe_execution
    def analyze_expected_position(self, tolerance=0.05, save=True, show=True):
        """
        Reads the file `<root_filename>_Position.txt` and analyzes patterns in the data:
            - Local minima and maxima (values and corresponding times)
            - Average oscillation period
            - Average amplitude
            - Mean value
            
        Saves the results to a TXT file and generates a plot with the highlighted points.

        """

        filename = f"{self.root_filename}/{self.root_filename}_Position.txt"
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} not found. Run 'expected_position_is_calculated' first.")

        # Loads the data while ignoring comment lines
        data = np.loadtxt(filename, comments="#")
        t = data[:, 0]
        x = data[:, 1]

        # --- Find local maxima and minima ---
        peaks, _ = find_peaks(x)
        troughs, _ = find_peaks(-x)

        max_times, max_vals = t[peaks], x[peaks]
        min_times, min_vals = t[troughs], x[troughs]

        # --- Estimate average period ---
        if len(peaks) > 1:
            peak_intervals = np.diff(max_times)
            period_mean = np.mean(peak_intervals)
            period_std = np.std(peak_intervals)
        else:
            period_mean, period_std = None, None

        # --- Other quantities ---
        amplitude = (np.max(x) - np.min(x)) / 2
        mean_value = np.mean(x)

        # --- Save results to a file ---
        out_file = f"{self.root_filename}/{self.root_filename}_PositionAnalysis.txt"
        with open(out_file, "w", encoding='utf-8') as f:
            f.write(f"# {out_file}\n")
            f.write(f"# Program version :   {self.version}\n")
            f.write(f"# Start:              {datetime.datetime.now().strftime('%c')}\n")
            f.write("# Analysis results of <x>(t)\n")
            f.write("# " + "="*80 + "\n")
            f.write(f"Mean Value:          {mean_value:.6f}\n")
            f.write(f"Amplitude:            {amplitude:.6f}\n")
            if period_mean is not None:
                f.write(f"Average period:    {period_mean:.6f} ± {period_std:.6f}\n")
            else:
                f.write("Average period: could not be estimated (too few peaks).\n")
            f.write("# " + "="*80 + "\n")
            f.write("# Maxima (time, value)\n")
            for tt, val in zip(max_times, max_vals):
                f.write(f"{tt:.6f}\t{val:.6f}\n")
            f.write("# " + "="*80 + "\n")
            f.write("# Minima (time, value)\n")
            for tt, val in zip(min_times, min_vals):
                f.write(f"{tt:.6f}\t{val:.6f}\n")

        print("Analysis results saved at:", out_file)

        # --- Summary printout ---
        print("="*80)
        print("Análise de <x>(t):")
        print(f"- Number of detected MAXIMA: {len(max_times)}")
        print(f"- Number of detected MINIMA: {len(min_times)}")
        if period_mean is not None:
            print(f"- Average Period: {period_mean:.4f} ± {period_std:.4f}")
        else:
            print("- Average period: could not be estimated (too few peaks).")
        print(f"- Average amplitude: {amplitude:.4f}")
        print(f"- Mean value: {mean_value:.4f}")
        print("="*80)

        # --- Graphic ---
        plt.figure(figsize=(10, 5))
        plt.plot(t, x, 'k-', label="<x>(t)")
        plt.plot(max_times, max_vals, 'ro', label="Maxima")
        plt.plot(min_times, min_vals, 'bo', label="Minima")
        plt.axhline(mean_value, color='g', linestyle='--', label="Mean Value")

        plt.xlabel("Time")
        plt.ylabel("Expected position <x>")
        plt.title("Expected position analysis")
        plt.legend(loc="upper left", bbox_to_anchor=(1.00, 1))
        plt.tight_layout()
        plt.grid(True, ls='--', lw=0.5)
        

        if save:
            plt.savefig(f"{self.root_filename}/{self.root_filename}_ExpectedPosition_Analysis.png", dpi=2000, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
    
 # ----- Methods for Calculate Momentum -----
    @safe_execution
    def expected_momentum(self, coefficients):
        """
            Returns a function that numerically calculates the normalized expected momentum ⟨p⟩(t):

                ⟨p⟩(t) = ∫ ψ*(x,t) (-iħ ∂ψ/∂x) w(x) dx / ∫ |ψ(x,t)|² w(x) dx

            Args:
                coefficients: List of coefficients of the wave function.

            Returns:
                Function that takes a time t and returns ⟨p⟩(t) numerically.
        """
        def expected_momentum_numeric(t):
            x_values = np.linspace(1e-6, self.length, 500)
            psi_values = np.array([self.wave_function(x_i, t, coefficients, normalize=False) for x_i in x_values])
            psi_conj = np.conj(psi_values)
            hbar = 1.0

            # Compute derivative using central differences.
            dpsi_dx = np.gradient(psi_values, x_values)
            p_operator_psi = -1j * hbar * dpsi_dx

            # Weight evaluated at every point.
            weight_values = np.array([self.weight(x_i) for x_i in x_values])

            # Numerator: ψ* · (p̂ ψ) · w(x)
            integrand_num = np.real(psi_conj * p_operator_psi) * weight_values
            numerator = np.trapz(integrand_num, x_values)

            # Denominator: |ψ|² · w(x)
            integrand_den = (np.abs(psi_values)**2) * weight_values
            denominator = np.trapz(integrand_den, x_values)

            if denominator == 0:
                raise ZeroDivisionError("Normalization denominator is zero. Check coefficients or wavefunction.")

            return numerator / denominator

        return expected_momentum_numeric

    @safe_execution
    def momentum_uncertainty(self, coefficients):
        """
            Returns a function that numerically calculates the momentum uncertainty Δp(t), normalized:

                Δp(t) = sqrt(⟨p²⟩ - ⟨p⟩²)

            where:
                ⟨p²⟩ = ∫ ψ* · (-ħ² d²ψ/dx²) · w(x) dx / ∫ |ψ|² · w(x) dx

            Args:
                coefficients: List of coefficients of the wave function.

            Returns:
                Function that takes time `t` and returns the momentum uncertainty.
        """
        def momentum_uncertainty_numeric(t):
            hbar = 1.0
            x_values = np.linspace(1e-6, self.length, 500)

            # ψ(x,t)
            psi_values = np.array([self.wave_function(x, t, coefficients, normalize=False) for x in x_values])
            psi_conj = np.conj(psi_values)

            # Weight at all points
            weight_values = np.array([self.weight(x) for x in x_values])

            # Compute the second derivative. d²ψ/dx²
            d2psi_dx2 = np.gradient(np.gradient(psi_values, x_values), x_values)
            p2_operator_psi = -hbar**2 * d2psi_dx2

            # Numerator de <p²>
            integrand_num = np.real(psi_conj * p2_operator_psi) * weight_values
            numerator = np.trapz(integrand_num, x_values)

            # Normalization denominator
            integrand_den = np.abs(psi_values)**2 * weight_values
            denominator = np.trapz(integrand_den, x_values)

            if denominator == 0:
                raise ZeroDivisionError("Normalization denominator is zero. Check coefficients or wavefunction.")

            exp_p2 = numerator / denominator

            # <p>²
            exp_p = self.expected_momentum(coefficients)(t)
            delta_p2 = exp_p2 - exp_p**2

            if np.isclose(delta_p2, 0, atol=1e-15):
                print("Aviso: dp2 ≈ 0. Estado possivelmente com momento bem definido.")

            return np.sqrt(max(0, delta_p2))

        return momentum_uncertainty_numeric

    @safe_execution
    def heisenberg_uncertainty(self, coefficients):
        """
            Returns a function that computes the product σₓ(t) * σₚ(t) and compares with ħ/2.

            Args:
                coefficients: List of coefficients of the wave function.

            Returns:
                Function that takes a time t and returns (σₓ·σₚ, ħ/2, boolean if inequality is satisfied).
        """
        hbar = 1.0 

        # Get uncertainty functions
        sigma_x_fn = self.position_uncertainty(coefficients)
        sigma_p_fn = self.momentum_uncertainty(coefficients)

        def uncertainty_product(t):
            sigma_x = sigma_x_fn(t)
            sigma_p = sigma_p_fn(t)
            product = sigma_x * sigma_p
            limit = hbar / 2
            return {
                "dx": sigma_x,
                "dp": sigma_p,
                "dx.dp": product,
                "hbar/2": limit,
                "valid": product >= limit
            }

        return uncertainty_product

    @safe_execution
    def momentum_uncertainty_relative(self, coefficients, t_max=None, num_points=200, save_filename=None):
        """
        Computes the relative momentum uncertainty r_p(t) = Δp(t) / ⟨p⟩(t)
        and records the times at which it reaches its minimum values.

        If t_max is None, returns the callable function r_p(t).
        Otherwise, it evaluates r_p(t) over [0, t_max], prints and saves
        the minimum times, saves all sampled points, and plots the graph.

        Args:
            coefficients (List[complex]): expansion coefficients (length = self.num_levels)
            t_max (float, optional): maximum time for evaluation and plotting.
                If None, returns the r_p(t) function without plotting.
            num_points (int): number of time samples (default = 200)
            save_filename (str, optional): output filename for results.
                If None, defaults to "{self.root_filename}_min_relative_momentum_uncertainty.txt"

        Returns:
            If t_max is None:
                Callable[[float], float]: the relative uncertainty function r_p(t)
            Else:
                Tuple[matplotlib.figure.Figure, float, List[float]]:
                    (figure, minimum value, list of times at the minimum)
        """

        # Validate coefficients
        if len(coefficients) != self.num_levels:
            raise ValueError(f"Expected {self.num_levels} coefficients, got {len(coefficients)}.")

        # Determine output filename if not provided
        if save_filename is None:
            save_filename = f"{self.root_filename}/{self.root_filename}_min_relative_momentum_uncertainty.txt"

        # Retrieve helper functions
        exp_p_fn = self.expected_momentum(coefficients)
        sigma_p_fn = self.momentum_uncertainty(coefficients)

        # Define the relative uncertainty function r_p(t)
        def r_p(t: float) -> float:
            mean_p = exp_p_fn(t)
            sigma_p = sigma_p_fn(t)
            if mean_p == 0:
                return np.inf
            return sigma_p / mean_p

        # If no plotting requested, return the callable
        if t_max is None:
            return r_p

        # Sample r_p on the time grid
        t_vals = np.linspace(0, t_max, num_points)
        y_vals = np.array([r_p(t) for t in t_vals])

        # Identify the minimum value and corresponding times
        min_val = y_vals.min()
        tol = 1e-15
        min_times = [
            t for t, y in zip(t_vals, y_vals)
            if np.isclose(y, min_val, atol=tol, rtol=tol)
        ]

        # Print results
        print(f"Minimum value of dp(t) / <p>(t): {min_val:.15f}")
        print("Estimated period (s):")
        for t in min_times:
            print(f"t = {t:.15f}")

        # Save results to file (including all sampled points)
        with open(save_filename, 'w', encoding='utf-8') as f:
            f.write(f"# {save_filename}\n")
            f.write(f"# Program version : {self.version}\n")
            f.write(f"# Minimum value of dp(t) / <p>(t): {min_val:.15f}\n")
            f.write("# Estimated period (s):\n")
            for t in min_times:
                f.write(f"# t = {t:.15f}\n")
            f.write("# " + "="*80 + "\n")
            f.write("# (1) time, (2) r_p(t)\n")
            for tt, yy in zip(t_vals, y_vals):
                f.write(f"{tt:.6f}\t{yy:.6f}\n")

        print("Full results saved in:", save_filename)

        # Plot the relative uncertainty curve
        fig, ax = plt.subplots()
        ax.plot(t_vals, y_vals, lw=2, color='black')
        ax.set_title("Relative Momentum Uncertainty r_p(t)")
        ax.set_xlabel("Time t")
        ax.set_ylabel("Δp(t)/<p>(t)")
        ax.grid(True)

        # Highlight the minimum points
        ax.plot(min_times, [min_val]*len(min_times), 'ro', label='Minimum point(s)')
        ax.legend(loc="upper left", bbox_to_anchor=(1.00, 1))
        plt.tight_layout()

        # Save the plot
        fig.savefig(f"{self.root_filename}/{self.root_filename}_RelativeMomentumUncertainty.png", bbox_inches='tight')
        plt.show()

        return fig, min_val, min_times

    @safe_execution
    def expected_momentum_and_uncertainty_are_calculated(self, t_range, coefficients, num_points=None):
        """
            Receives a time interval (t_range as (t0, t1)) and a list of coefficients,  
            and generates a file with four columns:  
            (1)Time  
            (2)Expected momentum  
            (3)Expected momentum – uncertainty  
            (4)Expected momentum + uncertainty  

            Behavior:  
            - If the parameter num_points is provided (must be a positive integer), it sets the number of intermediate points.  
            - Otherwise, 300 points will be used by default.  
        """
        if len(coefficients) != self.num_levels:
            raise ValueError(f"Expected {self.num_levels} coefficients, but got {len(coefficients)}.")
        
        # Check the number of points
        if num_points is not None:
            if isinstance(num_points, int) and num_points > 0:
                nt = num_points
            else:
                raise ValueError(f"Third argument must be a positive integer, got {num_points}.")
        else:
            print("Calculating 300 points by default.")
            nt = 300
        
        t0, t1 = t_range
        dt = (t1 - t0) / nt
        t_points = [t0 + k * dt for k in range(nt + 1)]
        
        # Functions to compute the expected value (mean) and uncertainty (standard deviation).
        ff = self.expected_momentum(coefficients)  # f(t) = ⟨p⟩(t)
        gg = self.momentum_uncertainty(coefficients)  # g(t) = σₚ(t)
        
        # Auxiliary function to guarantee the value is real; otherwise, returns 0.
        def ifnotreal(func, t_val):
            value = func(t_val)
            if np.isreal(value):
                return value
            else:
                return 0.0

        filename = f"{self.root_filename}/{self.root_filename}_MomentumUncertainty.txt"
        
        # Create and write to the output file.
        with open(filename, "w", encoding='utf-8') as f:
            f.write(f"# Filename :          {filename}\n")
            f.write(f"# Program version :   {self.version}\n")
            f.write(f"# Start:              {datetime.datetime.now().strftime('%c')}\n")
            f.write("# (1) time,\t(2) expected momentum,\t(3) expected momentum - uncertainty,\t(4) expected momentum + uncertainty\n")
            f.write("# " + "="*80 + "\n")
        
        with open(filename, "a", encoding='utf-8') as f:
            for tt in t_points:
                ex_mom = ff(tt)
                uncert = ifnotreal(gg, tt)
                f.write(f"{tt:.6f}\t{ex_mom:.6f}\t{(ex_mom - uncert):.6f}\t{(ex_mom + uncert):.6f}\n")
        
        print("Expected momentum and uncertainty data stored in file:", filename)

    @safe_execution
    def expected_momentum_and_uncertainty_are_plotted(self):
        """
            Reads a file containing expected momentum and uncertainty data, then returns a plot.

            The file should have 4 columns:  
            - (1) Time  
            - (2) Expected momentum (⟨p⟩)  
            - (3) Expected momentum (⟨p⟩) - uncertainty (σₚ)  
            - (4) Expected momentum (⟨p⟩) + uncertainty (σₚ)  

            If the file has only 2 columns, it is assumed that uncertainty data was not generated.  
            Then, the three curves are plotted simultaneously:  
            - Uncertainty regions are shaded in gray  
            - Central curve (expected momentum) is plotted in black  
        """
        filename = f"{self.root_filename}/{self.root_filename}_MomentumUncertainty.txt"
        if not os.path.exists(filename):
            raise FileNotFoundError(f"The file {filename} does not exist.")
        
        try:
            data = np.loadtxt(filename)
        except Exception as e:
            raise RuntimeError(f"Error reading file {filename}: {e}")

        # Check whether the data has 4 columns.
        if data.ndim == 1:
            data = data.reshape(1, -1)  # Adjustment to guarantee the object is a 2D array.
        if data.shape[1] != 4:
            raise ValueError("Data file must contain exactly 4 columns: time, expected momentum, expected momentum - uncertainty, and expected momentum + uncertainty.")
        
        time_vals = data[:, 0]
        expected_vals = data[:, 1]
        minus_vals = data[:, 2]
        plus_vals = data[:, 3]

        # Plotting the graph
        plt.figure(constrained_layout=True)
        plt.plot(time_vals, minus_vals, color='gray', label="<p> - dp")
        plt.plot(time_vals, expected_vals, color='black', label="<p>")
        plt.plot(time_vals, plus_vals, color='gray', label="<p> + dp")

        plt.xlabel("Time")
        plt.ylabel("Momentum")
        plt.title("Expected Momentum and Uncertainty")
        plt.grid(True)
        plt.legend(loc="upper left", bbox_to_anchor=(1.00, 1))
        plt.tight_layout()
        plt.savefig(f"{self.root_filename}/{self.root_filename}_ExpectedMomentumAndUncertainty.png", bbox_inches='tight')
        plt.show()
    
   # --- Methods for Comparing Problems ---

    @safe_execution
    def its_number_of_digits(self):
        '''
        Returns the current number of digits used by the object.
        '''
        return self.digits_used
   
    @safe_execution
    def is_described(self):
        """
        Display the problem description on the screen.
        """
        def get_func_repr(func):
            try:
                # Get the source lines of the function.
                src_lines = inspect.getsourcelines(func)[0]
                # Join the lines while removing indentation.
                src = "".join(line.strip() for line in src_lines)
                # If there's a 'return' statement, extract what follows.
                if "return" in src:
                    # Split at the first occurrence of "return" and return what follows, without extra spaces.
                    return src.split("return", 1)[1].strip()
                else:
                    return src
            except Exception as e:
                # If something fails, return to the function name.
                return func.__name__ if hasattr(func, '__name__') else str(func)
    
        separator = "-" * 80
        f_str = get_func_repr(self.f_function)
        g_str = get_func_repr(self.g_function)
        w_str = get_func_repr(self.weight)
        
        print(separator)
        print(f"Number of energy levels:       {self.num_levels}")
        print(f"Interval amplitude (Length):   {self.length} (float: {float(self.length)})")
        print(f"f function (V_spectral):       {f_str}")
        print(f"g function:                    {g_str}")
        print(f"w function (weigth):           {w_str}")
        print(f"Label:                         {self.label}")
        print(f"Number of digits to be used:   {self.num_digits}")
        
        if self.has_spectrum_been_calculated:
            if self.has_eigenvectors_been_calculated:
                print(f"The energy eigenvalues and eigenvectors have been calculated with {self.digits_used} digits by {self.calculator_name}.")
            else:
                print(f"The energy spectrum has been calculated with {self.digits_used} digits by {self.calculator_name}.")
                print("However, the energy eigenvectors have not been calculated yet.")
        else:
            print("The energy spectrum has not been calculated yet.")
        print(separator)
        print("Important: when using the spectral method, the potential must be expressed as V_spectral = 2*m*V_real.")

    @safe_execution
    def is_described_to_file(self, filename=None):
        """
            Write the problem description to a file.  
            Args:  
            - filename (str): Name of the file where the description will be written.  
        """
        if filename is None:
            filename = f"{self.root_filename}/{self.root_filename}_described.txt"
        # Helper function to obtain a user-friendly representation.
        def get_func_repr(func):
            return func.__name__ if hasattr(func, '__name__') else str(func)
            
        separator = "#" + "-" * 80
        with open(filename, "w", encoding='utf-8') as f:
            f.write(f"# Number of energy levels:            {self.num_levels}\n")
            f.write(f"# Interval amplitude (as entered):    {self.length}\n")
            f.write(f"# Interval amplitude (as float):      {float(self.length)}\n")
            f.write(f"# f function:                         {get_func_repr(self.f_function)}\n")
            f.write(f"# g function:                         {get_func_repr(self.g_function)}\n")
            f.write(f"# w (weight) function:                {get_func_repr(self.weight)}\n")
            f.write(f"# Label:                              {self.label}\n")
            f.write(f"# Number of Digits to be used:        {self.num_digits}\n")
            if self.has_spectrum_been_calculated:
                if self.has_eigenvectors_been_calculated:
                    f.write(f"# The energy eigenvalues and eigenvectors have been calculated with {self.digits_used} digits.\n")
                else:
                    f.write(f"# The energy spectrum has been calculated with {self.digits_used} digits.\n")
                    f.write("# However, the energy eigenvectors have not been calculated yet.\n")
            else:
                f.write("# The energy spectrum has not been calculated yet.\n")
            f.write(separator + "\n")
        print(f"Description has been written to file: {filename}")
    
    @safe_execution
    def compare_numerically(self, other):
        """
        Numerically compares the spectra (eigenvalues) of self (caller) and another instance (other).
        Generates a detailed TXT report.
      
            Behavior:  
            - If both problems have their spectra calculated, the generated file will contain:  
            (1) Level  
            (2) Eigenvalue of the caller problem  
            (3) Eigenvalue of the called problem  
            (4) Absolute variation (called - caller)  
            (5) Percentage variation (calculated relative to the caller's eigenvalue)  

            - The file will also include a header with program information (version, date, digit count) and problem descriptions.  
            - If spectra haven't been calculated for both problems, a warning message is displayed.
        """
        # Validation
        if not (self.has_spectrum_been_calculated and other.has_spectrum_been_calculated):
            print("Energy spectra have not been calculated for both problems.")
            return

        # Retrieve Data
        pSp = np.array(self.its_eigenvalues())  # Caller
        qSp = np.array(other.its_eigenvalues())  # Called
        
        # Limit to the smallest common size
        n_levels = min(len(pSp), len(qSp))
        pSp = pSp[:n_levels]
        qSp = qSp[:n_levels]

        # Vectorized Calculations (Faster than loops)
        diff_abs = qSp - pSp
        
        # Safe percentage calculation handling division by zero
        # usage: where(condition, value_if_true, value_if_false)
        diff_pct = np.where(
            np.isclose(pSp, 0, atol=1e-15), 
            0.0, 
            (diff_abs / np.abs(pSp)) * 100
        )

        # Write to File (Single Open Operation)
        filename = f"{self.root_filename}/{self.root_filename}_Comparison_Eigenvalues.txt"
        separator = "#" + "=" * 80 + "\n"

        print(f"Writing comparison to {filename}...")

        with open(filename, "w", encoding='utf-8') as f:
            # Header
            f.write(f"# Filename: {filename}\n")
            f.write("# Comparison of the eigenvalues of two different problems.\n")
            f.write(f"# Program version: {self.version}\n")
            f.write(f"# Start: {datetime.datetime.now().strftime('%c')}\n")
            f.write(separator)
            
            # Descriptions (Assuming is_described_to_file appends, we act carefully here. 
            f.write("# Description of the caller problem\n")
            f.close() # Close temporarily if is_described_to_file opens 'a' mode internally
            
            self.is_described_to_file(filename)
            with open(filename, "a", encoding='utf-8') as f: f.write("# Description of the called problem\n")
            other.is_described_to_file(filename)

            # Re-open for data writing
            with open(filename, "a", encoding='utf-8') as f:
                f.write(f"# Caller problem: {len(self.en_spectrum)} eigenvalues ({self.num_digits} Digits)\n")
                f.write(f"# Called problem: {len(other.en_spectrum)} eigenvalues ({other.num_digits} Digits)\n")
                f.write(separator)
                f.write(f"# Data for the lowest {n_levels} energy levels:\n")
                f.write("# (1) Level\t(2) Caller E\t(3) Called E\t(4) Abs Diff\t(5) % Diff\n")
                f.write(separator)

                # Batch writing loop
                for i in range(n_levels):
                    f.write(f"{i+1}\t{pSp[i]:.6f}\t{qSp[i]:.6f}\t{diff_abs[i]:.6f}\t{diff_pct[i]:.6f}\n")

        print("Done.")

    @safe_execution
    def compare_graphically(self, other):
        """
        Graphically compares the eigenvalues of two problems and 
        analyzes the orthogonality of the caller's eigenvectors.

            Prepares two plots:
            (1) Log10 of absolute variations (|qSp - pSp|), with:
            - Blue for positive variations (qSp - pSp > 0)
            - Red for negative variations
            - Green for zero variations

            (2) Percentage variation (qSp - pSp) / |pSp| * 100

            Before plotting, it displays the description of each problem.

            Args:  
            - other (SpectralMethod): Another instance for comparison.
        """
        # Validation
        if not (self.has_spectrum_been_calculated and hasattr(self, 'eigenvectors')):
            print("Please solve the eigenvalue and eigenvector problem first.")
            return

        # Print Descriptions
        print("Description of the caller problem:"); self.is_described(); print("\n")
        print("Description of the called problem:"); other.is_described(); print("\n")

        # Data Preparation (Vectorized)
        pSp = np.array(self.its_eigenvalues())
        qSp = np.array(other.its_eigenvalues())
        
        n_levels = min(len(pSp), len(qSp))
        pSp, qSp = pSp[:n_levels], qSp[:n_levels]
        
        diff = qSp - pSp
        
        # Calculate percentages safely
        with np.errstate(divide='ignore', invalid='ignore'):
            percent = (diff / np.abs(pSp)) * 100
        percent[np.isnan(percent)] = 0  # Fix 0/0 cases
        
        levels = np.arange(1, n_levels + 1)

        # Plotting Helper
        def plot_comparison(y_data, title, y_label, log_scale=False):
            plt.figure(figsize=(10, 5))
            
            # Boolean masks for coloring
            mask_pos = y_data > 1e-15
            mask_neg = y_data < -1e-15
            mask_null = ~mask_pos & ~mask_neg # Approximately zero

            if log_scale:
                # Transform data for log plot, handling signs
                y_plot = np.zeros_like(y_data)
                y_plot[mask_pos] = np.log10(y_data[mask_pos])
                y_plot[mask_neg] = np.log10(np.abs(y_data[mask_neg]))
                y_plot[mask_null] = 0 # Placeholder for zero
                
                if np.any(mask_pos): plt.scatter(levels[mask_pos], y_plot[mask_pos], c='blue', label="Positive (log10)")
                if np.any(mask_neg): plt.scatter(levels[mask_neg], y_plot[mask_neg], c='red', label="Negative (log10)")
                if np.any(mask_null): plt.scatter(levels[mask_null], y_plot[mask_null], c='green', label="Zero")
                plt.ylabel(f"log10(|{y_label}|)")
            else:
                if np.any(mask_pos): plt.scatter(levels[mask_pos], y_data[mask_pos], c='blue', label="Positive")
                if np.any(mask_neg): plt.scatter(levels[mask_neg], y_data[mask_neg], c='red', label="Negative")
                if np.any(mask_null): plt.scatter(levels[mask_null], y_data[mask_null], c='green', label="Zero")
                plt.ylabel(y_label)

            plt.xlabel("Energy level")
            plt.title(title)
            plt.grid(True, alpha=0.3)
            plt.legend(loc="upper left", bbox_to_anchor=(1.00, 1))
            plt.tight_layout()
            plt.show()

        # Plot 1: Absolute Variations
        print("-" * 80); print("Log10 Absolute Variation.\n")
        plot_comparison(diff, f"Log10 absolute variations (lowest {n_levels} levels)", "Variation", log_scale=True)

        # Plot 2: Percent Variations
        print("-" * 80); print("Percent variation.\n")
        plot_comparison(percent, f"Percent variations (lowest {n_levels} levels)", "Percent (%)", log_scale=False)

        # 4. Orthogonality Analysis
        print("-" * 80); print("Comparing orthogonality of eigenvectors (dot product).\n")
        
        sols = self.its_eigenpairs(save_to_file=False)
        vecs = [s[1] for s in sols] # Extract vectors
        nsols = len(vecs)
        
        # Build Deviation Matrix
        ort_matrix = np.zeros((nsols, nsols))
        for i in range(nsols):
            for j in range(i + 1, nsols):
                # We calculate upper triangle
                val = abs(self.__scalar_product(vecs[i], vecs[j]))
                ort_matrix[i, j] = val
                ort_matrix[j, i] = val # Symmetric

        # Plot Heatmap
        plt.figure(figsize=(8, 6))
        # Add epsilon to avoid log(0)
        im = plt.imshow(np.log10(ort_matrix + 1e-30), cmap="viridis", origin="lower", aspect="auto")
        plt.colorbar(im, label="log10(|<vi,vj>|)")
        plt.xlabel("Level i"); plt.ylabel("Level j")
        plt.title("Orthogonality deviation (Heatmap)")
        plt.tight_layout()
        plt.show()

        # Detailed Per-Level Plots
        # Warning: This loop creates a plot for EVERY level. Can be heavy.
        print("Generating individual orthogonality plots...")
        
        for i in range(nsols - 1):
            # Extract row i, columns from i+1 onwards
            upper_indices = np.arange(i + 1, nsols)
            errors = ort_matrix[i, i+1:]

            plt.figure(figsize=(10, 4))
            
            # Separate zero vs non-zero for plotting style
            mask_nz = errors > 1e-15
            mask_z = ~mask_nz
            
            x_vals = upper_indices + 1 # 1-based indexing for display
            
            if np.any(mask_nz):
                # Log scale for non-zeros
                y_vals = np.log10(errors[mask_nz])
                plt.plot(x_vals[mask_nz], y_vals, 'r-o', label="Non-zero deviation")
                
                # Dynamic Y-limit handling
                y_min, y_max = np.min(y_vals), np.max(y_vals)
                if y_min == y_max: y_max += 1.0; y_min -= 1.0
                plt.ylim(y_min - 0.5, y_max + 0.5)
                
            if np.any(mask_z):
                # Plot zeros at the bottom or separate visual
                # Since Y is log scale, we can't plot 0. We usually skip or plot at bottom limit.
                plt.plot(x_vals[mask_z], np.full(np.sum(mask_z), -16), 'g.', label="Zero (< 1e-15)")

            plt.xlabel("Energy level")
            plt.ylabel("log10(|<vi, vj>|)")
            plt.title(f"Orthogonality error: Level {i + 1} vs upper levels")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

    @safe_execution
    def check_solutions_numerically(self):
        """
            Numerically verifies the solutions (eigenvalues and eigenvectors) according to the following criteria:

            (1) Equation error: Calculates the maximum absolute value of the elements of M(E)v,  
            where M(E) = FirstBigMatrix() - E * SecondBigMatrix().
            (2) Normality deviation: Computes |1 - ⟨v, v⟩|.
            (3) Orthogonality deviation: For each pair (vi, vj), calculates ⟨vi, vj⟩.

            - The results are saved in text files for further analysis.
        """
        if not self.has_eigenvectors_been_calculated:
            print("Please solve the eigenvalue and eigenvector problem first.")
            return

        # Retrieve solutions
        sols = self.eigenvectors
        nsols = len(sols)
        
        # Construct the matrices ONLY ONCE outside the loop
        print("Building matrices for verification...")
        M1 = self.__FirstBigMatrix()
        M2 = self.__SecondBigMatrix()

        # ------------------ 1. Verification: Substitution in M(E)v = 0 ------------------
        filename_eq = f"{self.root_filename}/{self.root_filename}_Check_Equations.txt"
        print(f"Reporting errors in equations in file: {filename_eq}")

        header_eq = (
            f"# Maximum absolute error obtained after substitution of (v,E) in the eigenvector equation M(E)v=0\n"
            f"# Filename:           {filename_eq}\n"
            f"# Program version:    {self.version}\n"
            f"# Started in:         {datetime.datetime.now().strftime('%c')}\n"
            f"# Digits used:        {self.digits_used}\n"
            f"# " + "="*80 + "\n"
            "# (1) Energy Level\t(2) Error\n"
            f"# " + "="*80 + "\n"
        )
        
        with open(filename_eq, "w", encoding='utf-8') as f:
            f.write(header_eq)
            
            for i in range(nsols):
                E = sols[i][0]
                v = np.array(sols[i][1])
                
                # Calculates: (M1 - E*M2) * v
                term1 = M1.dot(v)
                term2 = M2.dot(v)
                residue = term1 - (E * term2)
                
                err = np.max(np.abs(residue))
                f.write(f"{i+1}\t{err:.20e}\n")

        # ------------------ 2. Verification: Normality of Eigenvectors ------------------
        filename_norm = f"{self.root_filename}/{self.root_filename}_Check_Normality.txt"
        print(f"Reporting errors in eigenvector normality in file: {filename_norm}")

        header_norm = (
            "# Checking deviation from normality of eigenvectors (dot product).\n"
            f"# Filename:           {filename_norm}\n"
            f"# Program version:    {self.version}\n"
            f"# Started in:         {datetime.datetime.now().strftime('%c')}\n"
            f"# Digits used:        {self.digits_used}\n"
            "# " + "="*80 + "\n"
            "# (1) Energy Level\t(2) Error\n"
            "# " + "="*80 + "\n"
        )
        
        with open(filename_norm, "w", encoding='utf-8') as f:
            f.write(header_norm)
            for i in range(nsols):
                v = sols[i][1]
                err = abs(1 - self.__scalar_product(v, v))
                f.write(f"{i+1}\t{err:.20e}\n")

        # ------------------ 3. Verification: Orthogonality between Eigenvectors ------------------
        filename_ortho = f"{self.root_filename}/{self.root_filename}_Check_Orthogonality.txt"
        print(f"Reporting errors in eigenvector orthogonality in file: {filename_ortho}")

        header_ortho = (
            "# Checking deviation from orthogonality of eigenvectors (dot product).\n"
            f"# Filename:           {filename_ortho}\n"
            f"# Program version:    {self.version}\n"
            f"# Started in:         {datetime.datetime.now().strftime('%c')}\n"
            f"# Digits used:        {self.digits_used}\n"
            "# " + "="*80 + "\n"
            "# Level i,\tLevel j,\tError\n"
            "# " + "="*80 + "\n"
        )
        
        with open(filename_ortho, "w", encoding='utf-8') as f:
            f.write(header_ortho)
            for i in range(nsols):
                # Upper-triangular loop (j > i) avoids recomputing symmetric pairs and the diagonal
                for j in range(i+1, nsols):
                    val = self.__scalar_product(sols[i][1], sols[j][1])
                    
                    if not np.isclose(np.imag(val), 0, atol=1e-15):
                        print(f"Warning: scalar product of levels {i+1} and {j+1} complex. Recording real part.")
                    
                    err = np.real(val)
                    f.write(f"{i+1}\t{j+1}\t{err:.20e}\n")

        print("Numerical verification Done.")

    @safe_execution
    def check_solutions_graphically(self, save: bool = True, show: bool = True, force_recalc: bool = False):
        """
            Graphically checks the quality of eigenvalues/eigenvectors:
            - Substitution errors (Mv ≈ 0)
            - Normality deviations (||v|| ≈ 1)
            - Orthogonality deviations (<vi,vj> ≈ 0)

            If the check files already exist, use their data.
            Otherwise, call check_solutions_numerically().
        """
        if not self.has_eigenvectors_been_calculated:
            print("Please solve the eigenvalue and eigenvector problem first.")
            return

        # Expected filenames
        f_eq = f"{self.root_filename}/{self.root_filename}_Check_Equations.txt"
        f_norm = f"{self.root_filename}/{self.root_filename}_Check_Normality.txt"
        f_ortho = f"{self.root_filename}/{self.root_filename}_Check_Orthogonality.txt"
        
        files_exist = os.path.exists(f_eq) and os.path.exists(f_norm) and os.path.exists(f_ortho)

        # If any file is missing or forced, compute everything before proceeding
        if not files_exist or force_recalc:
            print("Data files for graphical check missing or outdated. Running numerical check first...")
            self.check_solutions_numerically()

        # --- From this point on, it is guaranteed that the files exist. Only READ and PLOT. ---
        
        # Plot Substitution Error
        data = np.loadtxt(f_eq, comments="#")
        # Safeguard in case the file has only one line (becomes a 1D array)
        if data.ndim == 1: data = data.reshape(1, -1) 
        
        levels = data[:, 0]
        # Safe handling for log10
        errors = data[:, 1]
        log_errors = np.where(errors > 0, np.log10(errors), -30)

        self.__plot_scatter(levels, log_errors, 
                            title="Substitution error in eigenvector equation",
                            ylabel="log10(|error|)", 
                            filename_suffix="Check_Substitution",
                            color='ro', save=save, show=show)

        # Plot Normality Deviation
        data = np.loadtxt(f_norm, comments="#")
        if data.ndim == 1: data = data.reshape(1, -1)
        
        levels = data[:, 0]
        errors = data[:, 1]
        log_errors = np.where(errors > 0, np.log10(errors), -30)

        self.__plot_scatter(levels, log_errors, 
                            title="Normality deviation of eigenvectors",
                            ylabel="log10(|1 - <v,v>|)", 
                            filename_suffix="Check_Normality",
                            color='bo', save=save, show=show)

        # Plot Orthogonality (Summary 1D)
        # We need to process the orthogonality file to extract the maximum per level
        data_ortho = np.loadtxt(f_ortho, comments="#")
        if data_ortho.ndim == 1: data_ortho = data_ortho.reshape(1, -1)
        
        # Dictionary to track the maximum error per level
        max_err_per_level = {}
        nsols = len(self.eigenvectors)
        
        # Inicialize
        for i in range(1, nsols + 1):
            max_err_per_level[i] = -30.0 # Low baseline value for log scale

        for row in data_ortho:
            i, j, err = int(row[0]), int(row[1]), abs(float(row[2]))
            val_log = np.log10(err) if err > 0 else -30
            
            # Update the maximum for i and j (orthogonality is symmetric)
            if val_log > max_err_per_level[i]: max_err_per_level[i] = val_log
            if val_log > max_err_per_level[j]: max_err_per_level[j] = val_log

        # Prepare arrays for plotting
        levels_ortho = sorted(max_err_per_level.keys())
        vals_ortho = [max_err_per_level[k] for k in levels_ortho]

        self.__plot_scatter(levels_ortho, vals_ortho, 
                            title="Maximum orthogonality deviation per level",
                            ylabel="log error in orthogonality", 
                            filename_suffix="Check_Orthogonality_Summary",
                            color='ro', save=save, show=show)

        # Plot Orthogonality (2D Map)
        # Plot directly from the loaded data
        self.__plot_2d_ortho(data_ortho, save=save, show=show)

        print("Graphical check Done.")

    # --- Auxiliary plotting functions to keep the code clean ---
    @safe_execution
    def __plot_scatter(self, x, y, title, ylabel, filename_suffix, color, save, show):
        """Helper function for standard 1D plotting."""
        plt.figure(figsize=(10, 5))
        plt.plot(x, y, color, label=ylabel)
        plt.xlabel("En. level")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(loc="upper left", bbox_to_anchor=(1.00, 1))
        
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
        plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)
        
        if save:
            plt.savefig(f"{self.root_filename}/{self.root_filename}_{filename_suffix}.png", dpi=2000)
        if show:
            plt.show()
        else:
            plt.close()
    
    @safe_execution
    def __plot_2d_ortho(self, data, save, show):
        """Helper function for 2D orthogonality plots."""
        print("-" * 80)
        print("---------------- Orthogonality Summary (2D curve) ----------------")
        
        i_s = data[:, 0]
        j_s = data[:, 1]
        errs = np.abs(data[:, 2])
        log_errs = np.where(errs > 0, np.log10(errs), -30)

        plt.figure(figsize=(10, 5))
        sc = plt.scatter(i_s, j_s, c=log_errs, cmap="viridis", marker="o")
        plt.colorbar(sc, label="log10(|<vi,vj>|)")
        plt.xlabel("Level i")
        plt.ylabel("Level j")
        plt.title("Orthogonality deviation between eigenvectors")
        
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
        plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)
        
        if save:
            plt.savefig(f"{self.root_filename}/{self.root_filename}_Check_Orthogonality.png", dpi=2000)
        if show:
            plt.show()
        else:
            plt.close()
    # ---

    @safe_execution
    def check_equations(self, filename: str) -> None:
        """
            Checks whether each eigenvalue-eigenvector pair approximately satisfies the equation  
            M v = 0, where M = (F - E * S),  
            with F and S being the matrices obtained from build_matrix_d() and build_matrix_d_prime().  

            Procedure:  
            1. Checks whether the spectrum and eigenvectors have already been computed.  
            2. Opens (or creates) the file specified in filename and writes a header including:  
            - The number of digits used,  
            - The file name,  
            - The program version,  
            - The start date/time.  
            3. For each eigenvalue E and its corresponding eigenvector v, computes:  

            error = max( |(F - E * S) @ v| )  

            This value measures the error in satisfying the equation.  
            4. The results are written to the file.  

            Args:  
            - filename (str): Name of the file where results will be saved.  

            Returns:  
            - None.  
        """
        if filename is None:
            filename = f"{self.root_filename}/{self.root_filename}_Check_Equations.txt"

        if not (self.has_spectrum_been_calculated and self.has_eigenvectors_been_calculated):
                print("No eigenvectors have been computed yet. " )
                self.calculate_eigenvectors()
                self.calculate_spectrum()

        digits = self.digits_used
            
        # Defines the file header.
        header_lines = [
            f"# Checking maximum errors in satisfying eigenvalue equation. Digits used: {digits}",
            f"# Filename: {filename}",
            f"# Program version: {self.version}",
            f"# Started in: {time.strftime('%c')}",
            f"#{'='*80}",
            "# (1) En. Level\t(2) Error",
            f"#{'='*80}"
        ]
        # Opens the file in write mode (overwriting the current file, if it exists).
        with open(filename, "w", encoding='utf-8') as f:
            for line in header_lines:
                f.write(line + "\n")
            
        # Defines the matrices F and S.
        F = self.__build_matrix_d()
        S = self.__build_matrix_d_prime()
            
        # Checks for each level: it calculates Mv, that is, (F - E * S) @ v.
        with open(filename, "a", encoding='utf-8') as f:
            for i in range(self.num_levels):
                E_i = self.en_spectrum[i]
                v_i = np.array(self.eigenvectors[i])
                
                # CEnsures that v_i is a column vector; otherwise, it converts it.
                if v_i.ndim == 1:
                    v_i = v_i[:, np.newaxis]  
                
                M = F - E_i * S
                # Calculates the maximum error (maximum absolute value of the components of M @ v)
                error_val = np.max(np.abs(M @ v_i))
                
                # Tolerance for very small error
                tolerance = 1e-15
                if error_val < tolerance:
                    error_val = 0  # Ignores very small errors, in case the system is highly precise.
                
                # Writes the level (adjusted to start at 1) and the error
                f.write(f"{i+1}\t{error_val}\n")

    @safe_execution
    def relate_unsorted_and_sorted_lists(self, unsorted):
        """
            Sorts a list and relates it to the original unsorted list.  

            Example:  
            Given:  
            ```python
            unsorted = [5, 3, 2, 1, 4]
            ```  
            Returns:  
            ```python
            sorted_list = [1, 2, 3, 4, 5]
            mapping      = [4, 3, 2, 5, 1]
            ```
            The mapping indicates:  
            - The 1st element in the sorted list (1) was originally in the 4th position.  
            - The 2nd element (2) was originally in the 3rd position, and so on.  

            Args:  
            - unsorted (list): List of elements (must have distinct values).  

            Returns:  
            - tuple: (sorted_list, mapping), where mapping is 1-based.  
        """
        if len(unsorted) != len(set(unsorted)):
            raise ValueError("The list must contain distinct values.")

        # Get the sorted list and keep track of the original indices using enumerate
        sorted_list = sorted(unsorted)
        mapping = [index + 1 for value, index in sorted([(value, idx) for idx, value in enumerate(unsorted)], key=lambda x: x[0])]

        return sorted_list, mapping

    @safe_execution
    def mydensityplot(self, aux, x_label, a, b, N=200, cmap='gray_r'):
        """
            Generates a density plot ("cartoon") for the function aux over the interval [a, b].  

            Each value of aux is evaluated within the subinterval; the values are normalized and transformed into a single image row, which is replicated vertically to construct a 2D image.  

            Parameters:  
                - aux (callable): Function that receives x and returns a numeric value.  
                - x_label (str): Label for the x-axis.  
                - a (float): Lower bound of the interval.  
                - b (float): Upper bound of the interval.  
                - N (int, optional): Number of subdivisions in the interval (default: 200).  
                - cmap (str, optional): Colormap to use (default: "gray_r").  

            Returns:  
                - fig, ax: The figure and axes objects from Matplotlib.  
        """
        # Computes the subdivision step and the "vertical height" for the rectangles.
        delta = (b - a) / N
        Vertical = (b - a) * 0.618033988
        
        # Generates the points in the interval: from `a` to `b` with `N+1` points.
        points = np.linspace(a, b, N + 1)
        # Evaluates the function `aux` at each point.
        func_vals = [aux(pt) for pt in points]
        
        # Computes the maximum and minimum values.
        maxfunc = max(func_vals)
        minfunc = min(func_vals)
        deltafunc = maxfunc - minfunc
        if deltafunc == 0:
            deltafunc = 1  # Prevents division by zero if `aux` is constant.

        # Normalizes values according to the formula:  
        # `renormfunc[i] = -(minfunc - func_vals[i]) / deltafunc`
        renormfunc = [-(minfunc - val) / deltafunc for val in func_vals]
        
        # To generate a "cartoon"-style density plot, creates a 2D image by replicating the `renormfunc` line vertically (e.g., 50 lines).
        height = 50
        density_image = np.tile(np.array(renormfunc), (height, 1))
        
        # create figure
        fig, ax = plt.subplots()
        # Matplotlib colormap
        cmap_func = plt.get_cmap(cmap)
        
        # Displays the image using imshow.
        im = ax.imshow(density_image, extent=[a, b, 0, Vertical], aspect='auto', origin='lower', cmap=cmap_func)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Vertical")
        ax.set_title("Density Plot Cartoon")
        plt.colorbar(im, ax=ax)
        plt.show()
        
        return fig, ax
    
# ----- Method for Compute Optimal L ----- 
    
    @contextmanager
    def _temporary_state(self, **kwargs):
        """
            Context manager to temporarily modify instance attributes.

            - Validates critical parameters (`length`, `num_levels` must be positive).
            - Applies temporary values from `kwargs` and saves originals.
            - Clears caches; reinitializes if `length` is changed.
            - Logs changes if `debug_mode` is enabled.
            - Restores original values when exiting the context, even on errors.
        """

        old_values = {}
        context_id = id(self)  # debug
        
        try:
            # Validation
            if 'length' in kwargs and kwargs['length'] <= 0:
                raise ValueError(f"length must be positive, got {kwargs['length']}")
            if 'num_levels' in kwargs and kwargs['num_levels'] <= 0:
                raise ValueError(f"num_levels must be positive, got {kwargs['num_levels']}")
            
            # Save current values
            for key, value in kwargs.items():
                old_values[key] = getattr(self, key)
                setattr(self, key, value)
            
            # DEBUG: Log attribute changes
            if hasattr(self, 'debug_mode') and self.debug_mode:
                print(f"[DEBUG] Context {context_id}: entering with {kwargs}")
                if 'length' in kwargs:
                    print(f"       L changed from {old_values.get('length')} to {kwargs['length']}")
            
            # ALWAYS clear all caches
            self.clear_caches()
            
            # Force recomputation of critical integrals
            if 'length' in kwargs:
                # Reinitialize caches with fresh dictionaries
                self._weight_cache = {}
                self._f_integral_cache = {}
                self._g_integral_cache = {}
            
            yield self
            
        except Exception as e:
            # Log the error before restoring state
            print(f"[ERROR] In _temporary_state context {context_id}: {e}")
            raise
            
        finally:
            # Restore attribute values
            for key, value in old_values.items():
                setattr(self, key, value)
            
            # DEBUG: Log exit state
            if hasattr(self, 'debug_mode') and self.debug_mode:
                print(f"[DEBUG] Context {context_id}: exiting, restored {list(old_values.keys())}")

    @safe_execution
    def _optimization_energy(self, L: float, num_levels: int = None) -> float:
        """
        Raw version for the optimizer.
        Avoids rounding to prevent artificial discontinuities in the cost function.

        """
        if num_levels is None:
            num_levels = self.num_levels
            
        with self._temporary_state(length=L, num_levels=num_levels):
            D = self.__build_matrix_d()
            D_prime = self.__build_matrix_d_prime()
            
            try:
                # Compute only the ground-state eigenvalue (index 0)
                # Optimize with respect to the 5th excited state, since it enlarges the box, enhancing the global accuracy of the spectrum
                target_idx = 4 
                autovalores = eigh(
                    D, 
                    D_prime, 
                    eigvals_only=True, 
                    subset_by_index=[target_idx, target_idx],
                    check_finite=True
                    )
               
                result = float(autovalores[0])
                
                # Validação simples 
                if abs(result) < np.finfo(float).eps * 100:
                    return 0.0
                return result

            except np.linalg.LinAlgError:
                return float('inf')
            except ValueError:
                return float('inf')
            
    @safe_execution
    def optimize_length(self, num_levels_opt: int = None, L_bounds: tuple = (0.1, 50.0), verbose: bool = False) -> float:
        """
        Original method, now using improved find_optimal_L_for_N.
        """
        self.clear_caches()
        
        if num_levels_opt is None:
            num_levels_opt = self.num_levels
        
        if verbose:
            print("\n" + "="*50)
            print(f"L OPTIMIZATION")
            print(f"Number of basis functions: N = {num_levels_opt}")
            print(f"Limits: L ∈ [{L_bounds[0]}, {L_bounds[1]}]")
            print("="*50)
        
        L_opt = self.find_optimal_L_for_N(
            N=num_levels_opt,
            L_bounds=L_bounds,
            verbose=verbose
        )
        
        if verbose:
            print(f"\nResult: L_opt = {L_opt:.15f}")
            print("="*50)
        
        return L_opt
    
    @safe_execution
    def find_optimal_L_for_N(self, N: int, L_bounds: tuple = (0.1, 50.0), verbose: bool = False) -> float:
        """
        Intelligent Hybrid Optimization:
        Automatically distinguishes between potential wells ("Strings") and plateaus ("Oscillators")
        by analyzing the energy landscape on the initial grid.
        """
        L_min, L_max = L_bounds
        
        # Local cache
        energy_cache = {}

        def energy_wrapper(val_L):
            if val_L <= 0: return float('inf')
            key = round(val_L, 10) 
            if key in energy_cache: return energy_cache[key]
            E = self._optimization_energy(val_L, N)
            energy_cache[key] = E
            return E

        if verbose:
            print(f"\n[OPTIMIZATION] N={N} | Range=[{L_min:.2f}, {L_max:.2f}]")

        # Initial sampling (grid sweep)
        L_vals = self._create_smart_sampling(L_min, L_max, N)
        E_vals = np.array([energy_wrapper(l) for l in L_vals])
        
        valid_mask = np.isfinite(E_vals)
        if not np.any(valid_mask): return (L_min + L_max) / 2.0
        
        L_valid = L_vals[valid_mask]
        E_valid = E_vals[valid_mask]

        # Noise analysis and tolerance: Take the best 10% of points to estimate the "floor"
        idx_sorted = np.argsort(E_valid)
        n_best = max(5, len(E_valid) // 10)
        best_energies = E_valid[idx_sorted[:n_best]]
        
        min_E_grid = best_energies[0]
        noise_floor = np.std(best_energies) if n_best > 1 else 0.0
        
        # Hybrid tolerance: For high energies, relative error dominates. For low/exact energies, numerical noise dominates.
        rel_tol = 1e-12 
        tolerance = max(noise_floor * 2.0, abs(min_E_grid) * rel_tol, 1e-13)

        if verbose:
            print(f"  > Grid Min: {min_E_grid:.14f}")
            print(f"  > Tolerance: {tolerance:.2e}")

        # Topological classification: Which L values in the grid produce energy "as good as" the minimum?
        candidates_mask = (E_valid <= min_E_grid + tolerance)
        L_candidates = L_valid[candidates_mask]
        E_candidates = E_valid[candidates_mask]
        
        # Measure the spread of candidates (Max L - Min L)
        spread = np.max(L_candidates) - np.min(L_candidates)
        is_plateau = spread > (L_max - L_min) * 0.05 # If it spans more than 5% of the range, it is a plateau

        # Selection of the starting point for refinement
        if is_plateau:
            # Plateau: choose the SMALLEST L that satisfies the tolerance (left edge of the plateau)
            target_idx = np.argmin(L_candidates)
            L_target = L_candidates[target_idx]
            E_target = E_candidates[target_idx]
            strategy = "PLATEAU (Left Edge Optimization)"
            
            # Bounds focused on the left edge, allowing space to check if energy can decrease further
            search_bounds = (max(L_min, L_target * 0.5), min(L_max, L_target * 1.5))
            
        else:
            # Well / V-shape: small spread, so the minimum is localized
            # Choose the L with the LOWEST absolute energy, ignoring smaller L values
            target_idx = np.argmin(E_candidates) 
            L_target = L_candidates[target_idx]
            E_target = E_candidates[target_idx]
            strategy = "WELL (Global Min Optimization)"
            
            # Bounds strictly centered around the detected minimum
            search_bounds = (max(L_min, L_target * 0.8), min(L_max, L_target * 1.2))

        if verbose:
            print(f"  > Strategy: {strategy}")
            print(f"  > Target Start: L={L_target:.4f}")

        # Final refinement (minimize_scalar): Continuous optimizer applied only within the identified region of interest
        try:
            res = minimize_scalar(
                energy_wrapper, 
                bounds=search_bounds, 
                method='bounded', 
                options={'xatol': 1e-5}
            )
            L_opt = res.x
            E_opt = res.fun
        except:
            L_opt = L_target
            E_opt = E_target

       # Final validation (safety check): if refinement worsens the energy, revert to the best grid value
        if E_opt > min_E_grid + tolerance * 10:
             if verbose: print("  ! Refinement unstable, reverting to grid best.")
             return L_valid[idx_sorted[0]] # Return global min

        if verbose:
            print(f"  > Final Result: L={L_opt:.6f} (E={E_opt:.14f})")

        return L_opt
    
    def _create_smart_sampling(self, L_min, L_max, N):
        """ Helper extracted for cleanup. """
        # Base sampling points
        points = np.linspace(L_min, L_max, 100)
        
        # If the range is large, include logarithmically spaced points to span multiple orders of magnitude
        if L_max / L_min > 10:
            log_points = np.logspace(np.log10(L_min), np.log10(L_max), 50)
            points = np.concatenate([points, log_points])
            
        # Add higher sampling density at small L values (typically critical for RSM)
        small_points = np.linspace(L_min, L_min + (L_max-L_min)*0.2, 30)
        points = np.concatenate([points, small_points])
        
        return np.unique(np.sort(points))
    
    @safe_execution
    def compute_optimal_L_curve(self, N_values: list, L_bounds: tuple = (0.1, 50.0)):
        """
        Computes the L_opt(N) curve as shown in Figure 2 of the paper PEDRAN RMS.

        Args:
            N_values: List of N values
            L_bounds: Bounds for L

        Returns:
            Dictionary with {N: L_opt, ...}

        """
        results = {}
        
        print("Computing the L_opt(N) curve…")
        for N in N_values:
            L_opt = self.find_optimal_L_for_N(N, L_bounds, verbose=False)
            results[N] = L_opt
            print(f"  N={N:3d} -> L_opt = {L_opt:.6f}")
        
        return results
    
    @safe_execution
    def estimate_error(self, L: float, N: int, delta_N: int = 5) -> float:
        """
        Estimates the error as in the paper PEDRAN RMS: δ = |E_N − E_{N+ΔN}| / |E_N|

        Args:
            L: Box length
            N: Current number of basis functions
            delta_N: Increment used for error estimation

        Returns:
            Estimated relative error
        """
        E_N = self._optimization_energy(L, N)
        E_N_delta = self._optimization_energy(L, N + delta_N)
        
        return abs(E_N - E_N_delta) / abs(E_N)
    
    def clear_caches(self):
        """
        Fully and safely clears all caches.
        """
        # Initialize empty dictionaries for each cache
        self._weight_cache = {}
        self._f_integral_cache = {}
        self._g_integral_cache = {}
        
        # Clear the energy cache if present
        if hasattr(self, '_energy_cache'):
            self._energy_cache = {}
        
        # Clear any additional caches that may exist
        for attr_name in dir(self):
            if attr_name.endswith('_cache') and isinstance(getattr(self, attr_name), dict):
                getattr(self, attr_name).clear()
        
        return self

# ----- Method for Unified Compute -----

    @safe_execution
    def run_full_analysis(self, coeficientes, tempo: float = 100.0, pontos: int = 200):
        """
        Executes a complete analysis routine of the quantum system,
        calling in sequence the main methods already implemented.

        Parameters
        ----------
        coeficientes : list[complex]
            Expansion coefficients of the wave function.
        tempo : float, optional
            Maximum time for time-dependent calculations (default=100).
        pontos : int, optional
            Number of sampling points for time-dependent functions (default=200).
        """

        def section(title: str):
            print("=" * 80)
            print(f"\n>>>>>>> {title}\n")

        # Settings
        section("Function Settings")
        print(f"Time set: {tempo}")
        print(f"Points:   {pontos}\n")

        # System description
        section("DESCRIBED SYSTEM")
        self.is_described()

        # Solve system
        section("SYSTEM SOLVE...")
        self.is_solved()
        self.calculate_spectrum()
        self.calculate_eigenvectors()
        self.its_eigenpairs()

        # Expected position
        section("EXPECTED POSITION")
        self.expected_position_is_calculated((0.0, tempo), coeficientes, num_points=pontos)
        self.expected_position_is_plotted()

        # Uncertainty and expected position
        section("UNCERTAINTY AND EXPECTED POSITION")
        self.expected_position_and_uncertainty_are_calculated((0.0, tempo), coeficientes, num_points=pontos)
        self.expected_position_and_uncertainty_are_plotted()

        # Uncertainty and expected momentum
        section("UNCERTAINTY AND EXPECTED MOMENTUM")
        self.expected_momentum_and_uncertainty_are_calculated((0.0, tempo), coeficientes, pontos)
        self.expected_momentum_and_uncertainty_are_plotted()

        # Wavefunction and probability density
        section("WAVEFUNCTION AND DENSITY PROBABILITY")
        self.plot_eigenfunctions(num_levels=3, save=True)
        self.probability_density_is_plotted(0, coeficientes, num_frames=50, save=True)
        self.probability_density_is_plotted((0, tempo), coeficientes, num_frames=50, save=True)
        self.plot_wavefunction_and_density(
            t=0,
            coefficients=coeficientes,
            num_slices=300,
            color_psi="blue",
            linestyle_psi="-",
            color_rho="red",
            alpha_rho=0.5,
            title="|psi| e |psi|² em t=0",
            xlabel="x",
        )
        self.probability_density_3d((0, 20), coeficientes, num_frames=100, num_slices=200,
                          cmap='plasma', title="Probability Density 3d", xlabel="x", ylabel="time", zlabel="|ψ(x,t)|²")
        
        # Relative uncertainties
        section("UNCERTAINTY RELATIVE")
        self.position_uncertainty_relative(coeficientes, t_max=tempo, num_points=pontos)
        self.momentum_uncertainty_relative(coeficientes, t_max=tempo, num_points=pontos)

        # Probability density cartoon
        section("PROBABILITY DENSITY (CARTOON)")
        self.probability_density_cartoon(
            t=(0, tempo), coefficients=coeficientes, num_slices=20, cmap="plasma", alpha=0.8
        )

        # Numeric momentum functions
        section("UNCERTAINTY AND EXPECTED MOMENTUM (NUMERIC)")
        self.expected_momentum_and_uncertainty_are_calculated((0, tempo), coeficientes, pontos)
        self.expected_momentum_and_uncertainty_are_plotted()

        section("NUMERIC FUNCTIONS: <x> e dx")
        expected_position_func = self.expected_position(coeficientes)
        position_uncertainty_func = self.position_uncertainty(coeficientes)
        for i in range(0, int(tempo) + 1, 5):
            print(f"t={i}: <x> = {expected_position_func(i):.15} | dx = {position_uncertainty_func(i):.15}")

        section("NUMERIC FUNCTIONS: <p> e dp")
        expected_momentum_func = self.expected_momentum(coeficientes)
        momentum_uncertainty_func = self.momentum_uncertainty(coeficientes)
        for i in range(0, int(tempo) + 1, 5):
            print(f"t={i}: <p> = {expected_momentum_func(i):.15} | dp = {momentum_uncertainty_func(i):.15}")

        # Norm of wave function
        section("NORM OF WAVE FUNCTION")
        n = self.norm_of_wave_function(coeficientes)
        for t in np.linspace(0, tempo, 100):
            print(f"Norm at t={t:.1f}: {n(t):.15}")

        # Heisenberg uncertainty
        section("HEISENBERG UNCERTAINTY")
        heis_fn = self.heisenberg_uncertainty(coeficientes)
        for i in range(0, int(tempo) + 1, 5):
            result = heis_fn(t=i)
            print(f"t={i} | dx.dp = {result['dx.dp']:.15} | Heisenberg satisfy? {result['valid']}")

        # Analysis
        section("ANALYZE EXPECTED POSITION")
        self.analyze_expected_position()

        # Check solutions
        section("CHECK SOLUTIONS NUMERICALLY AND GRAPHICALLY")
        self.check_solutions_numerically()
        self.check_solutions_graphically()

        section("DONE!")
        """
        Executes a complete analysis routine of the quantum system,
        calling in sequence the main methods already implemented.

        Parameters
        ----------
        coeficientes : list[complex]
            Expansion coefficients of the wave function.
        tempo : float, optional
            Maximum time for time-dependent calculations (default=100).
        pontos : int, optional
            Number of sampling points for time-dependent functions (default=200).
        """

        def section(title: str):
            print("=" * 80)
            print(f"\n>>>>>>> {title}\n")

        # Settings
        section("Function Settings")
        print(f"Time set: {tempo}")
        print(f"Points:   {pontos}\n")

        # System description
        section("DESCRIBED SYSTEM")
        self.is_described()

        # Solve system
        section("SYSTEM SOLVE...")
        self.is_solved()
        self.calculate_spectrum()
        self.calculate_eigenvectors()
        self.its_eigenpairs()

        # Expected position
        section("EXPECTED POSITION")
        self.expected_position_is_calculated((0.0, tempo), coeficientes, num_points=pontos)
        self.expected_position_is_plotted()

        # Uncertainty and expected position
        section("UNCERTAINTY AND EXPECTED POSITION")
        self.expected_position_and_uncertainty_are_calculated((0.0, tempo), coeficientes, num_points=pontos)
        self.expected_position_and_uncertainty_are_plotted()

        # Uncertainty and expected momentum
        section("UNCERTAINTY AND EXPECTED MOMENTUM")
        self.expected_momentum_and_uncertainty_are_calculated((0.0, tempo), coeficientes, pontos)
        self.expected_momentum_and_uncertainty_are_plotted()

        # Wavefunction and probability density
        section("WAVEFUNCTION AND DENSITY PROBABILITY")
        self.plot_eigenfunctions(num_levels=3, save=True)
        self.probability_density_is_plotted(0, coeficientes, num_frames=50, save=True)
        self.probability_density_is_plotted((0, tempo), coeficientes, num_frames=50, save=True)
        self.plot_wavefunction_and_density(
            t=0,
            coefficients=coeficientes,
            num_slices=300,
            color_psi="blue",
            linestyle_psi="-",
            color_rho="red",
            alpha_rho=0.5,
            title="|psi| e |psi|² em t=0",
            xlabel="x",
        )
        self.probability_density_3d((0, 20), coeficientes, num_frames=100, num_slices=200,
                          cmap='plasma', title="Probability Density 3d", xlabel="x", ylabel="time", zlabel="|ψ(x,t)|²")
        
        # Relative uncertainties
        section("UNCERTAINTY RELATIVE")
        self.position_uncertainty_relative(coeficientes, t_max=tempo, num_points=pontos)
        self.momentum_uncertainty_relative(coeficientes, t_max=tempo, num_points=pontos)

        # Probability density cartoon
        section("PROBABILITY DENSITY (CARTOON)")
        self.probability_density_cartoon(
            t=(0, tempo), coefficients=coeficientes, num_slices=20, cmap="plasma", alpha=0.8
        )

        # Numeric momentum functions
        section("UNCERTAINTY AND EXPECTED MOMENTUM (NUMERIC)")
        self.expected_momentum_and_uncertainty_are_calculated((0, tempo), coeficientes, pontos)
        self.expected_momentum_and_uncertainty_are_plotted()

        section("NUMERIC FUNCTIONS: <x> e dx")
        expected_position_func = self.expected_position(coeficientes)
        position_uncertainty_func = self.position_uncertainty(coeficientes)
        for i in range(0, int(tempo) + 1, 5):
            print(f"t={i}: <x> = {expected_position_func(i):.15} | dx = {position_uncertainty_func(i):.15}")

        section("NUMERIC FUNCTIONS: <p> e dp")
        expected_momentum_func = self.expected_momentum(coeficientes)
        momentum_uncertainty_func = self.momentum_uncertainty(coeficientes)
        for i in range(0, int(tempo) + 1, 5):
            print(f"t={i}: <p> = {expected_momentum_func(i):.15} | dp = {momentum_uncertainty_func(i):.15}")

        # Norm of wave function
        section("NORM OF WAVE FUNCTION")
        n = self.norm_of_wave_function(coeficientes)
        for t in np.linspace(0, tempo, 100):
            print(f"Norm at t={t:.1f}: {n(t):.15}")

        # Heisenberg uncertainty
        section("HEISENBERG UNCERTAINTY")
        heis_fn = self.heisenberg_uncertainty(coeficientes)
        for i in range(0, int(tempo) + 1, 5):
            result = heis_fn(t=i)
            print(f"t={i} | dx.dp = {result['dx.dp']:.15} | Heisenberg satisfy? {result['valid']}")

        # Analysis
        section("ANALYZE EXPECTED POSITION")
        self.analyze_expected_position()

        # Check solutions
        section("CHECK SOLUTIONS NUMERICALLY AND GRAPHICALLY")
        self.check_solutions_numerically()
        self.check_solutions_graphically()

        section("DONE!")