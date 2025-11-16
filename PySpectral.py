#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Library-name: PySpectral Quantum Solver
Description: Library developed in Python with spectral methods for solving eigenvalue problems and wave functions in quantum wells.

Author:         Vagner Jandre Monteiro  
Contact:        <vagner.jandre@iprj.uerj.br>
Create date:    2025-03-24  
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
      • math
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

import numpy as np
import sympy as sp
import pandas as pd
import datetime
import os
import numbers
import time
import math
import inspect
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D  

from matplotlib.animation import FuncAnimation
from scipy.linalg import eig, eigh
from scipy.integrate import quad, fixed_quad

from scipy.sparse.linalg import eigsh  
from scipy.optimize import minimize_scalar
from matplotlib.ticker import MaxNLocator, MultipleLocator
from scipy.signal import find_peaks

from matplotlib import animation


# =========================================================================== 
# Class
# =========================================================================== 
class SpectralMethod:
    
    def __init__(self, num_levels, length, f_function=None, g_function=None, num_digits=15, weight=None, root_filename="output", label="project", optimizer_L='n'):
        
        """
        Initializes the spectral problem.

        Args:
            num_levels (int): Number of levels (eigenfunctions) in the problem.
            length (float): Length of interval L, positive.
            f_function (callable): Function f.
            g_function (callable): Function g.
            num_digits (int, optional): Calculation precision (default: 15).
            weight (callable, optional): Weight function w(x). If None, w(x) is assumed to be 1.
        """

        self.num_levels = num_levels
    
        self.f_function = f_function if f_function is not None else lambda x: 0  # f(x) = 0 if not provided
        self.g_function = g_function if g_function is not None else lambda x: 1  # g(x) = 1 if not provided
        self.weight = weight if weight is not None else lambda x: 1  # w(x) = 1 if not provided
    
        # Attributes that will be assigned after the calculation
        self.num_digits = num_digits
        self.digits_used = num_digits
        
        self.has_spectrum_been_calculated = False
        self.has_eigenvectors_been_calculated = False
        self.en_spectrum = None    # Ordered list of eigenvalues
        self.eigenvectors = None  # Each element of eigenvectors will be a pair [E, eigenvector].     
        self.version = "Spectral_2.0"
        
        # For file writing, we define a root (if none exists, we use "Output").
        self.root_filename = root_filename  
        self.label = label            
        self.calculator_name = "Python"  # To identify the system used
        
        # Dictionary to cache the elements of the weight matrix.
        self._weight_cache = {}

        # Selects whether the length attribute will be user-defined or optimized.        
        self.length = length
        if optimizer_L.upper() == 'Y':
                L_min = 0.1
                L_max = 10 * length
                tol = 10**(-self.num_digits)
                self.length = self.optimize_length(L_min, L_max, tol)
 
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
        integrand = lambda x: np.sin((m * np.pi * x) / L) * np.sin((n * np.pi * x) / L) * func(x)
        result, err1 = fixed_quad(integrand, 0.0, L, n=10000)
        
        
        return (2 / L) * result
    
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
    
    # --- Methods for Weight Matrix ---
    def __weight_matrix_offdiagonal(self, m, n):
        """
            Returns the (m, n) element of the off-diagonal weight matrix.
        """
        return self.__CIntegral(self.weight, m, n)

    def __weight_matrix_diagonal(self, m):
        """
            Returns the (m, m) diagonal element of the weight matrix.
        """
        return self.__CIntegral(self.weight, m, m)

    def __weight_function(self, n, m):
        """
            Returns the (n, m) element of the weight matrix,  
            using memorization to avoid repeated recalculations.
        """
        key = (n, m)
        if key in self._weight_cache:
            return self._weight_cache[key]
        
        # If n == m, use the diagonal case; otherwise, use the off-diagonal case.
        if n == m:
            value = self.__weight_matrix_diagonal(m)
        else:
            value = self.__weight_matrix_offdiagonal(m, n)
        self._weight_cache[key] = value
        return value
    
    def __scalar_product(self, u, v):
        """
            Computes the dot product of vectors u and v,  
            considering the weight:  
            Σ₍i,j₎ uᵢ * WeightFunction(i,j) * vⱼ.
            The vectors are expected as lists (or arrays), and indexing is 1-based.  
        """
        if len(u) != len(v):
            raise ValueError("Vectors have different dimensions.")
        
        dim = len(u)
        total = 0
        for i in range(1, dim+1):
            for j in range(1, dim+1):
                total += u[i-1] * self.__weight_function(i, j) * v[j-1]
        return total
    
    # --- Construction of matrices D and D' (Equation (32) from [Pedran2008]). ---
    
    # --- First
    def __FirstBigMatrixProc(self, n, m):
        """
            Defines the element (n,m) of matrix D, according to Eq.(32) from [Pedran2008].  
            For the diagonal case: returns n²·(π)²/L² + C(n,m);  
            For the off-diagonal case: returns only C(n,m).  

            Note: Since Maple uses 1-based indexing, we expect n and m to be positive integers here.
        """
        aux = self.__C(n, m)
        if n == m:
            return ((n**2 * (np.pi)**2) / (self.length**2)) + aux
        else:
            return aux

    def __FirstBigMatrix(self):
        """
            Builds the matrix D (of order N x N) according to Eq.(32) from [Pedran2008],  
            with elements defined by FirstBigMatrixProc(n,m). The matrix is symmetric.  
            Returns a NumPy array.
        """
        N = self.num_levels
        M = np.zeros((N, N))
        for i in range(N):
            for j in range(i, N):
                # Convert to 1-based indexes:
                value = self.__FirstBigMatrixProc(i+1, j+1)
                M[i, j] = value
                M[j, i] = value
        return M

    def __SecondBigMatrix(self):
        """
            Builds the matrix D' (of order N x N) according to Eq.(32), using the elements __C2.  
            Returns a NumPy array (symmetric matrix).
        """
        N = self.num_levels
        M = np.zeros((N, N))
        for i in range(N):
            for j in range(i, N):
                value = self.__C2(i+1, j+1)
                M[i, j] = value
                M[j, i] = value
        return M 
    
    # --- Second
    def __build_matrix_d(self):
        """
            Builds the matrix D defined in Eq.(32) from [Pedran2008].  
            For diagonal elements, uses: n²*(π)²/L² + C(n,n).  
            For off-diagonal elements, uses: C(n,m),  
            where C(n,m) is obtained from the integrals associated with function f.
        """
        matrix_d = np.zeros((self.num_levels, self.num_levels))
        for n in range(self.num_levels):
            for m in range(self.num_levels):
                # The calculation of C(n,m) is implemented in an auxiliary method (here, integrated into the __CIntegral function)
                aux = self.__CIntegral(self.f_function, m+1, n+1)  # Using m+1 and n+1 because of the index in Maple (starting at 1)
                if n == m:
                    matrix_d[n, m] = (((n+1)**2 * (np.pi)**2)/(self.length**2)) + aux
                else:
                    matrix_d[n, m] = aux
        return matrix_d

    def __build_matrix_d_prime(self):
        """
            Builds the matrix D′ defined in Eq.(32) from [Pedran2008],  
            based on the integrals associated with function g.
        """
        matrix_d_prime = np.zeros((self.num_levels, self.num_levels))
        for n in range(self.num_levels):
            for m in range(self.num_levels):
                matrix_d_prime[n, m] = self.__CIntegral(self.g_function, m+1, n+1)
        return matrix_d_prime

# --- Methods for Spectrum Calculation and Eigenvectors ---

    def its_eigenvalues(self, *args):
        """
            Returns the calculated eigenvalue spectrum.  
            
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

        # If no arguments are passed, returns the full spectrum
        if len(args) == 0:
            return self.en_spectrum
        
        # If only one argument was passed
        elif len(args) == 1:
            arg = args[0]
            if isinstance(arg, int):
                if arg <= 0:
                    raise ValueError("Index must be a positive integer.")
                return self.en_spectrum[arg - 1]
            elif isinstance(arg, slice):
                # Converts slice indexes from 1-based to 0-based:
                start = arg.start - 1 if arg.start is not None else None
                stop = arg.stop - 1 if arg.stop is not None else None
                step = arg.step
                return self.en_spectrum[start:stop:step]
            elif isinstance(arg, range):
                # Converts the indices of the range object (1-based) to 0-based.
                return [self.en_spectrum[i - 1] for i in arg]
            elif isinstance(arg, list):
                # If a list of integers was passed: convert each index
                return [self.en_spectrum[i - 1] for i in arg]
            else:
                raise ValueError(f"Unsupported argument type: {type(arg)}. Use int, slice, range, or list of ints.")
        
        else:
            # If multiple arguments are passed, it must be a sequence of integers
            indices = []
            for a in args:
                if isinstance(a, int):
                    indices.append(a)
                else:
                    raise ValueError(f"Unsupported argument type: {type(arg)}. Use int, slice, range, or list of ints.")
            return [self.en_spectrum[i - 1] for i in indices]

    def its_eigenvalues_silently(self, *args):
        """
            Silently retrieve eigenvalues. Typically used internally.
            Args:
            - args: Can be empty or a single positive integer.
            Returns:
            - List of eigenvalues or a specific eigenvalue.
        """
        if self.has_spectrum_been_calculated:
            if len(args) == 1:
                arg = args[0]
                if isinstance(arg, int) and arg > 0:
                    # Return the n-th eigenvalue
                    return self.en_spectrum[arg - 1]  # Adjust for zero-based indexing
            # Otherwise, return the whole spectrum
            return self.en_spectrum
        else:
            raise RuntimeError("Spectrum hasn't been calculated yet. Please use a method to calculate it first.")
        
    def calculate_spectrum(self, save_to_file: bool = True):
        """
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
        print("Calculating energy eigenvalues (single attempt).")
        print("_" * screen_width)

        A = self.__build_matrix_d()
        B = self.__build_matrix_d_prime()
        
        evs = eig(A, B, right=False)

        # Filters the eigenvalues, keeping only the real part if the imaginary part is negligible.
        res = [u.real if np.isclose(u.imag, 0, atol=1e-12) else u for u in evs]
        
        #FILTER: remove inf, NaN and non-finite values
        res = [r for r in res if np.isfinite(r)]

        nroots = len(res)
        
        if nroots != self.num_levels:
            print(f"Warning: only {nroots} eigenvalues have been found, expected {self.num_levels}.")

        number_complex = sum(1 for r in res if isinstance(r, complex))
        if number_complex > 0:
            print(f"Warning: {number_complex} eigenvalues have non-negligible imaginary parts.")

        print("Finished eigenvalue calculation.")
        print("_" * screen_width)

        self.has_spectrum_been_calculated = True
        self.en_spectrum = sorted(res)
        
        # --- Save to File TXT ---
        if save_to_file:
            filename = f"{self.root_filename}_Spectrum.txt"
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

    def calculate_spectrum_silently(self):
        """
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

        while True:
            # Construction of matrices A and B
            A = self.__build_matrix_d()
            B = self.__build_matrix_d_prime()

            # Eigenvalue calculation
            evs = eig(A, B, right=False)
            
            # Filters the eigenvalues, keeping only the real part if the imaginary part is negligible.
            res = [u.real if np.isclose(u.imag, 0, atol=1e-12) else u for u in evs]
            
    
            #FILTER: remove inf, NaN and non-finite values
            res = [r for r in res if np.isfinite(r)]

            nroots = len(res)
            # Checks if the number of eigenvalues found is correct.
            if nroots != self.num_levels or any(isinstance(r, complex) for r in res):
                # Increases the precision and tries again.
                digits = int(digits * (1 + digits_percent_increase))
                continue
            
            break

        # Stores the precision used, the spectrum, and marks that the calculation has been done.
        self.digits_used = digits
        self.has_spectrum_been_calculated = True
        self.en_spectrum = sorted(res)

        return self.en_spectrum

    def calculate_eigenvectors(self, save_to_file: bool = True):
        """
        Computes the eigenvectors for the generalized eigenvalue problem A*v = λ*B*v.
        After obtaining eigenvalues and eigenvectors (via scipy.linalg.eig), the results are
        sorted according to the eigenvalues, and each eigenvector is normalized so that
        the first nonzero element is positive.

        The results are stored in self.eigenvectors as a list of pairs [E, eigenvector],
        and the flag self.has_eigenvectors_been_calculated is set to True.
        
        If save_to_file=True (Default), saves results in "<root_filename>_Eigenvectors.txt".

        """
        if not self.has_spectrum_been_calculated:
            print("The energy spectrum hasn't been calculated yet.")
            self.calculate_spectrum()
        
        print("_" * 80)
        print("Calculating energy eigenvectors now ...")
        
        # Calculate eigenvalues and eigenvectors of the generalized eigenvalue problem
        A = self.__build_matrix_d()
        B = self.__build_matrix_d_prime()
        
        w, V = eig(A, B, right=True)

        # Convert small imaginary parts to zero
        w = np.real_if_close(w, tol=1e-12)
        V = np.real_if_close(V, tol=1e-12)

        # Sort eigenvalues and corresponding eigenvectors
        sorted_idx = np.argsort(w)
        w_sorted = w[sorted_idx]
        V_sorted = V[:, sorted_idx]

        def normalize_eigenvector(vec, tol=1e-12):
            """Normalize the eigenvector to have unit norm and first nonzero element positive."""
            norm = self.__scalar_product(vec, vec)
            if norm < tol:
                return vec
            vec = vec / np.sqrt(norm) 
            first_nonzero_idx = np.argmax(np.abs(vec) > tol)
            if vec[first_nonzero_idx] < 0:
                vec = -vec
            return vec

        # Create normalized eigenvector list
        eigenvectors = []
        for i in range(V_sorted.shape[1]):
            vec = V_sorted[:, i]
            vec_normalized = normalize_eigenvector(vec)
            eigenvectors.append([w_sorted[i], vec_normalized])  # keeps as ndarray

        # Store the eigenvectors and mark the calculation as complete
        self.eigenvectors = eigenvectors
        self.has_eigenvectors_been_calculated = True
        
        # --- Save to File ---
        if save_to_file:
            filename = f"{self.root_filename}_Eigenvectors.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"# {filename}\n")
                f.write(f"# Program version : {self.version}\n")
                f.write(f"# Date: {datetime.datetime.now().strftime('%c')}\n")
                f.write("# (1) Eigenvalue, (2) Eigenvector (full)\n")
                f.write("# " + "="*80 + "\n")
                for eigval, eigvec in eigenvectors:
                    eigvec = np.array(eigvec).flatten()
                    eigvec_str = "[" + ", ".join(f"{comp:.12e}" for comp in eigvec) + "]"
                    f.write(f"{eigval:.12e}\t{eigvec_str}\n")

            print(f"Eigenvectors saved to: {filename}")
            
        print("_" * 80)
        print("Done!")
        print("=" * 80)

        return eigenvectors
      
    def is_solved(self, number_of_digits=None):
        """
            Solves the eigenvalue and eigenvector problem.  

            - If a value for number_of_digits (a positive integer) is provided, it updates the number of digits (precision) to be used in calculations.  

            Procedure:  
                1. Optional Updates self.num_digits with the provided value.  
                2. Computes and times the spectrum (eigenvalues) by calling self.calculate_spectrum().  
                3. Computes and times the eigenvectors by calling self.calculate_eigenvectors().  
                4. Displays a final message indicating that the problem has been solved, including the number of digits used.  
        """
        if number_of_digits is not None:
            if isinstance(number_of_digits, int) and number_of_digits > 0:
                self.num_digits = number_of_digits
            else:
                raise ValueError("Number of digits must be a positive integer.")

        # Check if spectrum and eigenvectors have already been calculated.
        if not self.has_spectrum_been_calculated:
            print("Calculating eigenvalues ...")
            t0 = time.time()
            self.calculate_spectrum()
            elapsed = time.time() - t0
            print(f"Time elapsed for eigenvalue calculation: {elapsed:.6f} seconds.")
        else:
            print("Eigenvalues already calculated. Skipping eigenvalue calculation.")

        if not self.has_eigenvectors_been_calculated:
            print("Calculating eigenvectors ...")
            t0 = time.time()
            self.calculate_eigenvectors()
            elapsed = time.time() - t0
            print(f"Time elapsed for eigenvector calculation: {elapsed:.6f} seconds.")
        else:
            print("Eigenvectors already calculated. Skipping eigenvector calculation.")

        # Define the solver name.
        self.calculator_name = "Python"
        
        print("=" * 80)
        print(f"The eigenvalue/eigenvector problem has been completely solved with {self.digits_used} digits used.")
    
    def its_eigenvectors(self, *args):
        """
        Returns the computed eigenvectors (or a subset) in 1-based indexing.

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
        # ensure eigenvectors are computed
        if not self.has_eigenvectors_been_calculated:
            self.calculate_eigenvectors()

        get_vector = lambda item: item[1]  # extracts only the vector v from [λ, v]

        # no args: return all
        if len(args) == 0:
            return [get_vector(ev) for ev in self.eigenvectors]

        elif len(args) == 1:
            # int -> single vector
            arg = args[0]
            if isinstance(arg, int):
                if arg <= 0:
                    raise ValueError("Index must be a positive integer.")
                return get_vector(self.eigenvectors[arg - 1])
            
            # slice -> subset
            elif isinstance(arg, slice):
                start = arg.start - 1 if arg.start is not None else None
                stop = arg.stop - 1 if arg.stop is not None else None
                step = arg.step
                return [get_vector(ev) for ev in self.eigenvectors[start:stop:step]]
            
            # range or list -> multiple
            elif isinstance(arg, range):
                return [get_vector(self.eigenvectors[i - 1]) for i in arg]
            elif isinstance(arg, list):
                return [get_vector(self.eigenvectors[i - 1]) for i in arg]
            else:
                raise ValueError(f"Unsupported argument type: {type(arg)}. Use int, slice, range, or list of ints.")
        else:
            # multiple ints
            indices = []
            for a in args:
                if isinstance(a, int):
                    if a <= 0:
                        raise ValueError("Indices must be positive integers.")
                    indices.append(a)
                else:
                    raise ValueError(f"Unsupported argument type in multiple args: {type(a)}.")
            return [get_vector(self.eigenvectors[i - 1]) for i in indices]

    def its_eigenpairs(self, *args, save_to_file: bool = True):
        """
        Returns eigenvalue/eigenvector pairs (λ, v) in 1-based indexing.
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

        # Pair Selection
        if len(args) == 0:
            result = self.eigenvectors
        elif len(args) == 1:
            arg = args[0]
            if isinstance(arg, int):
                if arg <= 0:
                    raise ValueError("Index must be a positive integer.")
                result = self.eigenvectors[arg - 1]
            elif isinstance(arg, slice):
                start = arg.start - 1 if arg.start is not None else None
                stop = arg.stop - 1 if arg.stop is not None else None
                step = arg.step
                result = self.eigenvectors[start:stop:step]
            elif isinstance(arg, range):
                result = [self.eigenvectors[i - 1] for i in arg]
            elif isinstance(arg, list):
                result = [self.eigenvectors[i - 1] for i in arg]
            else:
                raise ValueError(f"Unsupported argument type: {type(arg)}. Use int, slice, range, or list of ints.")
        else:
            indices = []
            for a in args:
                if isinstance(a, int):
                    if a <= 0:
                        raise ValueError("Indices must be positive integers.")
                    indices.append(a)
                else:
                    raise ValueError(f"Unsupported argument type in multiple args: {type(a)}.")
            result = [self.eigenvectors[i - 1] for i in indices]

        # --- Save to File ---
        if save_to_file:
            filename = f"{self.root_filename}_Eigenpairs.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"# {filename}\n")
                f.write(f"# Program version : {self.version}\n")
                f.write("# (1) Eigenvalue, (2) Eigenvector (full)\n")
                f.write("# " + "="*80 + "\n")

                # Ensures we always have a list of pairs
                if isinstance(result, list) and all(isinstance(r, (list, tuple)) and len(r) == 2 for r in result):
                    pairs = result
                else:
                    pairs = [result]

                for eigval, eigvec in pairs:
                    eigvec = np.array(eigvec).flatten()
                    # Serializes the vector as a single string
                    eigvec_str = "[" + ", ".join(f"{comp:.12e}" for comp in eigvec) + "]"
                    line = f"{eigval:.12e}\t{eigvec_str}"
                    f.write(line + "\n")

            print(f"Eigenpairs saved in: {filename}")

        return result

    def its_eigenfunction(self, x, i, tolerance=1e-15):
        """
            Returns the i-th normalized eigenfunction.  

                - If x is numeric (i.e., an instance of `numbers.Number`), it uses NumPy.  
                - Otherwise, it treats x as symbolic and constructs the expression using SymPy.  

                The eigenfunction is defined as:  
                        ψ_i(x) = sqrt(2/L) * Σₘ₌₁^N [aₘ * sin(m π x / L)]

                where:  
                - L = self.length  
                - N = number of eigenvector coefficients (eigenstates)  
                - aₘ are the coefficients of the corresponding eigenvector  

                Args:  
                - x: Numeric value of x or string (e.g., "x").  
                - i: Level (1,2,3...).  
                - tolerance: Coefficients smaller than this value will be ignored.  

                Returns:  
                - If x is numeric (x=0.5): A numeric value.  
                - If x is a string ("x"): A SymPy expression.  

        """
        try:
            L = self.length
            N = self.num_levels 
        except:
            print("The attributes: length and levels were not found. Define them and try again.")
        try:
            coeffs = self.eigenvectors[i - 1][1]
        except:
            print("Eigenvectors not found. Compute them and try again.")
        
        if isinstance(x, (numbers.Number, np.ndarray)):
            # Ensure that x is converted to a 1D array.
            x_arr = np.atleast_1d(x).astype(float)
            
            # Convert the coefficients to a NumPy array:
            a = np.array(coeffs, dtype=float)
            m = np.arange(1, N + 1, dtype=float)
            
            # Filter insignificant coefficients:
            mask = np.abs(a) > tolerance
            if np.any(mask):
                a = a[mask]
                m = m[mask]
            
            # Compute sin(m * π * x / L) such that m is treated as a column:
            sin_term = np.sin(m[:, np.newaxis] * np.pi * x_arr / L)
            # Sum over the m axis:
            result = np.sqrt(2 / L) * np.sum(a[:, np.newaxis] * sin_term, axis=0)
            
            # If the input was a scalar, return a scalar:
            if result.size == 1:
                return result.item()
            else:
                return result
        else:
            # Symbolic branch: convert x to a Sympy object.
            x_sym = sp.sympify(x)
            # If self.length is a NumPy array, extract the scalar value:
            if isinstance(L, np.ndarray):
                L = L.item()
            L_sym = sp.Float(L)
            
            coeffs_sym = [sp.Float(c) for c in coeffs]
            psi_expr = sp.sqrt(2 / L_sym) * sp.Add(*[
                coeffs_sym[m - 1] * sp.sin(m * sp.pi * x_sym / L_sym)
                for m in range(1, N + 1)
                if abs(coeffs_sym[m - 1]) > tolerance
            ])
            return psi_expr

    def an_eigenfunction(self, x, basis_coeffs):
            """
            Individual eigenfunction.
            Evaluate the eigenfunction ψ(x) from the coefficients in the sine basis.

            The eigenfunction is expanded as:

                ψ(x) = sqrt(2 / L) * Σ_{m=1}^N coeffs[m-1] * sin(m * π * x / L)

            Parameters
                x : float or array_like
                    Position or positions at which to evaluate ψ(x).
                coeffs : array_like, shape (N_basis,)
                    Basis coefficients (components of the eigenvector).
                    Coefficients c₁ … c_N are computed in `CalculateEigenVectors`.
                L : float
                    Length of the domain.

            Returns

                float or ndarray
                    Value(s) of ψ(x). If x is a scalar, returns a scalar.
                    If `x` is an array, returns an array.
        """

            # Recover L
            L = getattr(self, "L_optimal", None) or getattr(self, "length", None)
            if L is None:
                raise AttributeError("I did not find 'L_optimal' nor 'length' in the instance.")

            # Convert coefficients and build index m = [1, 2, …, N]
            c = np.asarray(basis_coeffs).flatten()
            m = np.arange(1, c.size + 1)

            # Convert x to an array (supports float or array input).
            x_arr = np.atleast_1d(x)

            # φₘ(x) = sin(m·π·x/L) for each m and each x
            # resulting in φ shape : (N_basis, len(x_arr))
            phi = np.sin(np.pi * m[:, None] * x_arr[None, :] / L)

            # Linear combination + factor √(2/L)
            psi_vals = np.sqrt(2.0 / L) * (c[:, None] * phi).sum(axis=0)

            # If the input is a scalar, a scalar is returned.
            return psi_vals[0] if np.isscalar(x) else psi_vals

    def wave_function(self, x, t, coefficients, tolerance=1e-15, normalize=True, save_to_file=False):
        """
        Constructs the wave function ψ(x, t) as a linear combination of eigenfunctions
        for given x and time t, using the list of coefficients.

        Normalization is performed with respect to the weighted norm:
            ∫ |ψ(x, t)|² * w(x) dx

        Returns:
            - Symbolic expression if x and/or t are symbolic.
            - Numerical value if x and t are numerical.
        
        Example:
            x = np.linspace(0, 1, 200)
            t = np.linspace(0, 1, 50)
            psi_vals = p.wave_function(x, t, coeficientes)
            
        """

        if not self.has_eigenvectors_been_calculated:
            raise RuntimeError("Eigenvectors haven't been calculated yet.")
                
        # Validate number of coefficients
        if len(coefficients) != self.num_levels:
            raise ValueError(f"Expected {self.num_levels} coefficients, got {len(coefficients)}.")

        # --- numeric case ---
        if (isinstance(x, (numbers.Number, np.generic, list, np.ndarray)) and
            (isinstance(t, (numbers.Number, np.generic, list, np.ndarray)))):

            # Ensures arrays
            x_vals = np.atleast_1d(x)
            t_vals = np.atleast_1d(t)

            results = np.zeros((len(t_vals), len(x_vals)), dtype=complex)

            for j, tj in enumerate(t_vals):
                psi_total = 0
                for i in range(self.num_levels):
                    if abs(coefficients[i]) > tolerance:
                        eigenvalue = self.eigenvectors[i][0]
                        psi_n_x = self.its_eigenfunction(x_vals, i + 1, tolerance)
                        psi_total += coefficients[i] * psi_n_x * np.exp(1j * eigenvalue * tj)

                if normalize:
                    integrand = lambda x_: np.abs(self.wave_function(x_, tj, coefficients, normalize=False))**2 * self.weight(x_)
                    norm_squared = fixed_quad(integrand, 0, self.length, n=1000)[0]
                    psi_total = psi_total / np.sqrt(norm_squared)

                results[j, :] = psi_total

            # --- save to a TXT file ---
            if save_to_file:
                filename = f"{self.root_filename}_WaveFunction.txt"
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(f"# {filename}\n")
                    f.write(f"# Program version : {self.version}\n")
                    f.write(f"# Date: {datetime.datetime.now().strftime('%c')}\n")
                    f.write("# Columns: x\t t\t Psi(x,t)\n")
                    f.write("# " + "="*60 + "\n")

                    for j, tj in enumerate(t_vals):
                        for xi, psi in zip(x_vals, results[j, :]):
                            f.write(f"{xi:.6f}\t{tj:.6f}\t{psi:.12e}\n")
                            
                print(f"Wave function saved to: {filename}")

            # If it is a scalar, it returns a scalar; if it is an array, it returns a matrix.
            if np.ndim(x) == 0 and np.ndim(t) == 0:
                return results[0,0]
            elif np.ndim(t) == 0:
                return results[0,:]
            elif np.ndim(x) == 0:
                return results[:,0]
            else:
                return results

        # --- symbolic case ---
        else:
            x_sym = sp.sympify(x)
            t_sym = sp.sympify(t)
            psi_expr = 0
            for i in range(self.num_levels):
                if abs(coefficients[i]) > tolerance:
                    eigenvalue = self.eigenvectors[i][0]
                    psi_n_x = self.its_eigenfunction(x_sym, i + 1, tolerance)
                    psi_expr += coefficients[i] * psi_n_x * sp.exp(sp.I * eigenvalue * t_sym)
            
            # Default symbolic normalization (excluding the weight).
            norm = sp.sqrt(sum(sp.Abs(c)**2 for c in coefficients))
            if norm == 0:
                return 0
            return psi_expr / norm
    
    def probability_density(self, x, t, coefficients, simplify_expr=False, save_to_file=False):
        """
        Returns the probability density |ψ(x,t)|² for position x, time t,
        and the coefficients of the linear combination.

        Args:
            x : float, array-like ou simbólico
                Posição.
            t : float, array-like ou simbólico
                Tempo.
            coefficients (list): Coeficientes (tamanho igual a self.num_levels).
            simplify_expr (bool): Se True, aplica sp.simplify à expressão simbólica.
            save_to_file (bool): Se True, salva os resultados em arquivo TXT.

        Returns:
            - Se x e/ou t forem simbólicos, retorna expressão SymPy.
            - Caso contrário, retorna valores numéricos (float, vetor ou matriz).
        
        Example:
            x = np.linspace(0, 1, 200)
            t = np.linspace(0, 1, 50)
            rho_vals = p.probability_density(x, t, coeficientes)
        """

        # Validates the number of coefficients.
        if len(coefficients) != self.num_levels:
            raise ValueError(f"Expected {self.num_levels} coefficients, got {len(coefficients)}.")

        psi = self.wave_function(x, t, coefficients)

        # --- Symbolic case ---
        if isinstance(psi, sp.Basic):
            result = sp.Abs(psi)**2
            result = sp.simplify(result) if simplify_expr else result
            return result

        # --- Numerical case ---
        else:
            result = np.abs(psi)**2

            # --- save to a TXT file ---
            if save_to_file:
                filename = f"{self.root_filename}_ProbabilityDensity.txt"
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(f"# {filename}\n")
                    f.write(f"# Program version : {self.version}\n")
                    f.write(f"# Date: {datetime.datetime.now().strftime('%c')}\n")
                    f.write("# Columns: x\t t\t |Psi(x,t)|²\n")
                    f.write("# " + "="*60 + "\n")

                    x_vals = np.atleast_1d(x)
                    t_vals = np.atleast_1d(t)

                    # If both are arrays, it generates a complete table.
                    if x_vals.ndim == 1 and t_vals.ndim == 1:
                        for j, tj in enumerate(t_vals):
                            row = result[j, :] if result.ndim == 2 else result
                            for xi, rho in zip(x_vals, row):
                                f.write(f"{xi:.6f}\t{tj:.6f}\t{rho:.12e}\n")

                    # If x is an array and t is a scalar.
                    elif x_vals.ndim == 1 and t_vals.size == 1:
                        for xi, rho in zip(x_vals, result):
                            f.write(f"{xi:.6f}\t{t_vals[0]:.6f}\t{rho:.12e}\n")

                    # If x is a scalar and t is an array.
                    elif t_vals.ndim == 1 and x_vals.size == 1:
                        for tj, rho in zip(t_vals, result):
                            f.write(f"{x_vals[0]:.6f}\t{tj:.6f}\t{rho:.12e}\n")

                    # If both are scalars.
                    else:
                        f.write(f"{x_vals[0]:.6f}\t{t_vals[0]:.6f}\t{result:.12e}\n")
                
                print(f"Probability density saved to: {filename}")
            
            return result
 
    def normalize_eigenvector(self, u):
        """
            Normalizes an eigenvector (list or array) according to the definition of the dot product.  

            Args:  
            - u (list or array-like): Vector containing the eigenvector components.  

            Returns:  
            - numpy.array: Normalized vector.  
        """
        u = np.asarray(u)
        
        if u.ndim != 1:
            raise ValueError("The eigenvector must be a 1D array or list.")

        # Compute the norm (inner product).
        norm_factor = np.sqrt(self.__scalar_product(u,u))
        # Check whether the norm is close to zero (in which case normalization is not meaningful).
        
        if np.isclose(norm_factor, 0, atol=1e-15):
            return u  # Returns the original vector if the norm is zero (or nearly zero).
        # Normalize the vector.
        return u / norm_factor

    def norm_of_wave_function(self, coefficients):
        """
            Returns a function norm_func(t) that calculates the norm of the wave function,  
            which is the integral (over x from 0 to L) of the term:  

            ProbabilityDensity(x, t, coefficients) * w(x)

            where w(x) is the defined weight function.  

            Args:  
            - coefficients (list): Coefficients of the linear combination (as in `wave_function`).  

            Returns:  
            - A function norm_func(t) that returns the numerical value of the integral.  

        """
        # Check if the coefficients are valid
        if not isinstance(coefficients, (list, np.ndarray)):
            raise ValueError("Coefficients must be a list or array.")
        
        # Validate that we have the right number of coefficients
        if len(coefficients) != self.num_levels:
            raise ValueError(f"Expected {self.num_levels} coefficients, got {len(coefficients)}.")

        # Define the function norm_func(t)
        def norm_func(t):
            # Performs integration with respect to x.
            integral, err = fixed_quad(lambda x: self.probability_density(x, t, coefficients) * self.weight(x), 0.0, self.length, n=10000)
          
            return integral
        
        return norm_func
    
    # --- Plot Methods for WaveFunction and Probability Density ---
    
    def plot_wavefunctions(self, num_levels: int = None, levels: list[int] = None, x_points: int = 800, show: bool = True, save: bool = True):
        """
            Plot the eigenfunctions ψₙ(x) for the selected energy levels.
            Each ψₙ(x) is constructed using all components of its eigenvector.

            Parameters
                num_levels : int, optional
                    If set, plots the first `num_levels` levels (n = 0 … num_levels–1).
                levels : list of int, optional
                    Specific 1-based indices of levels to plot (e.g., [1, 3, 5]).
                    If provided, `num_levels` is ignored.
                x_points : int
                    Number of x points used to evaluate ψₙ(x).
                save_path : str or None
                    If provided, saves the figure to this file.
                show : bool
                    If True, displays the plot; otherwise, only saves and closes it.
        """
        save_path = f"{self.root_filename}_wavefunctions.png"
        
        # Decide which 1-based indices to use
        if levels is not None:
            idx = levels
        elif num_levels is not None:
            if num_levels < 1:
                raise ValueError("`num_levels` deve ser >= 1.")
            idx = list(range(1, num_levels + 1))
        else:
            raise ValueError("Forneça `num_levels` ou `levels`.")

        # Extract pairs (λₙ, vₙ) via its_eigenpairs
        pairs = self.its_eigenpairs(*idx)
        # Normalize to a list.
        if not isinstance(pairs, list):
            pairs = [pairs]

        # Domain in x.
        L = getattr(self, "length", None) or getattr(self, "L_optimal", None)
        if L is None:
            raise AttributeError("Define 'self.length' before plotting.")
        x = np.linspace(0.0, L, x_points)

        # Plot each energy level
        plt.figure(figsize=(8, 5))
        for level, (eigval, vec) in zip(idx, pairs):
            psi_n = self.an_eigenfunction(x, vec)
            n = level - 1
            plt.plot(x, psi_n.real, label=f"ψₙ (n={n})")

        # Apply final decorations (labels, titles, legends, etc.).
        plotted = [i - 1 for i in idx]
        plt.title(f"Eigenfunctions ψₙ(x) — Levels {plotted}")
        plt.xlabel("x")
        plt.ylabel("ψₙ(x)")
        plt.legend(loc="upper left", bbox_to_anchor=(1.00, 1))
        plt.grid(True)
        plt.tight_layout()

        # Save or display the figure.
        if save:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_wavefunction_3d(self, level: int, x_points: int = 800, y_points: int = 50, show: bool = True, save: bool = True):
        """
        Plots the eigenfunction ψₙ(x) as a 3D surface for a single level.

        Axes:
            - x: position
            - y: auxiliary axis (only to give “width” to the surface)
            - z: value of ψₙ(x)

        """

        save_path = f"{self.root_filename}_wavefunction3D_level{level}.png"

        pair = self.its_eigenpairs(level)
        eigval, vec = pair

        L = getattr(self, "length", None) or getattr(self, "L_optimal", None)
        if L is None:
            raise AttributeError("Define self.length before plotting.")
        x = np.linspace(0.0, L, x_points)

        psi_n = self.an_eigenfunction(x, vec).real

        # Create a 2D mesh: the Y-axis is only auxiliary.
        y = np.linspace(0, 1, y_points)
        X, Y = np.meshgrid(x, y)
        Z = np.tile(psi_n, (y_points, 1))  # Repeats ψ(x) along the Y-axis.

        # Plot the surface.
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")

        n = level - 1
        ax.set_title(f"Eigenfunction ψₙ(x) — Level {n}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("ψₙ(x)")
        fig.colorbar(surf, shrink=0.5, aspect=5)

        if save:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close()
    
    def probability_density_is_plotted(self, t, coefficients, num_frames=None, num_slices=None,
                                   save=False, **plotopts):
        """
        Plot the probability density |ψ(x,t)|² as a function of position.
        
        Possible Calls
            - p.probability_density_is_plotted(t, coefficients)
                t is a single time instant (numeric): generates a static plot.

            - p.probability_density_is_plotted((t1, t2), coefficients)
                t is a time interval: generates an animation using num_frames (default: 10).

            - p.probability_density_is_plotted((t1, t2), coefficients, num_frames)
                Generates an animation with the specified number of frames.

        Parameters
            t : float or tuple/list of float
                Single instant (numeric) or interval [t1, t2] representing the time range.
            coefficients : list of complex
                List of N complex coefficients (normalization is automatic).
            num_frames : int, optional
                Number of frames for the animation (default: 100).
            num_slices : int, optional
                Number of points along the x-axis (default: 200).
            save : bool, optional
                If True, saves the animation as a GIF or PNG.
            **plotopts : dict, optional
                Additional plotting options such as "color", "title", "xlabel", "ylabel", "figsize".
        """

        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        # Validates the number of coefficients.
        if len(coefficients) != self.num_levels:
            raise ValueError(f"Expected {self.num_levels} coefficients, got {len(coefficients)}.")

        # Animated case (time interval).
        if isinstance(t, (tuple, list)) and len(t) == 2:
            if num_frames is None:
                num_frames = 100
            t1, t2 = t
            times = np.linspace(t1, t2, num_frames)
            if num_slices is None:
                num_slices = 200
            x_values = np.linspace(0, self.length, num_slices)

            fig, ax = plt.subplots(figsize=plotopts.get("figsize", (8, 6)))

            # Precomputes all frames.
            y_all_frames = [self.probability_density(x_values, t_frame, coefficients) for t_frame in times]
            global_min = np.min([np.min(y) for y in y_all_frames])
            global_max = np.max([np.max(y) for y in y_all_frames])
            ax.set_ylim(global_min, global_max)

            line, = ax.plot(x_values, y_all_frames[0], color=plotopts.get("color", "blue"))
            ax.set_xlabel(plotopts.get("xlabel", "Position (x)"))
            ax.set_ylabel(plotopts.get("ylabel", "|ψ(x,t)|²"))

            def update(i):
                line.set_ydata(y_all_frames[i])
                ax.set_title(f"Probability Density |ψ(x,t)|² (t={times[i]:.2f})")
                return line,

            ani = animation.FuncAnimation(fig, update, frames=len(times), interval=200, blit=True)

            if save:
                filename = f"{self.root_filename}_prob_density.gif"
                ani.save(filename, writer="pillow", fps=5)
                plt.close(fig)
                print(f"Animation saved at: '{filename}'")
            else:
                plt.show()

        # Static case (single time)
        else:
            if num_slices is None:
                num_slices = 500
            x_values = np.linspace(0, self.length, num_slices)
            density_values = self.probability_density(x_values, t, coefficients)

            plt.figure(figsize=plotopts.get("figsize", (8, 6)))
            plt.plot(x_values, density_values, color=plotopts.get("color", "blue"))
            plt.xlabel(plotopts.get("xlabel", "Position (x)"))
            plt.ylabel(plotopts.get("ylabel", "|ψ(x,t)|²"))
            plt.title(plotopts.get("title", f"Probability Density |ψ(x,t)|² (t={t})"))
            plt.grid(True)

            if save:
                filename = f"{self.root_filename}_prob_density_t{t}.png"
                plt.savefig(filename, dpi=150)
                plt.close()
                print(f"Static plot saved at: '{filename}'")
            else:
                plt.show()
     
    def probability_density_3d(self, t_interval, coefficients, num_frames=50, num_slices=200,
                            animate_rotation=False, save_as_gif=True, gif_filename=None, **plotopts):
        """
        Plot the probability density |ψ(x,t)|² in a 3D graph.

        Axes:
            - x-axis: Position (0 -> self.length)
            - y-axis: Time (t1 -> t2)
            - z-axis: Probability density

        Parameters
        ----------
        t_interval : tuple(float, float)
            Intervalo de tempo (t1, t2).
        coefficients : list[complex]
            Coeficientes da função de onda.
        num_frames : int
            Número de pontos no tempo.
        num_slices : int
            Número de pontos no espaço.
        animate_rotation : bool
            Se True, anima a rotação 3D.
        save_as_gif : bool
            Se True, salva a animação como GIF.
        gif_filename : str
            Nome do arquivo GIF (default: "<root>_probability_density_3d.gif").
        **plotopts : dict
            Opções extras de plotagem (cmap, figsize, labels, etc.).
        """

        if gif_filename is None:
            gif_filename = f"{self.root_filename}_probability_density_3d.gif"

        # Time interval.
        t1, t2 = t_interval
        t_values = np.linspace(t1, t2, num_frames)
        x_values = np.linspace(0, self.length, num_slices)

        # Mesh (time × position).
        T, X = np.meshgrid(t_values, x_values, indexing='ij')

        # Probability density.
        density = np.empty_like(T, dtype=float)
        for i, t_val in enumerate(t_values):
            density[i, :] = self.probability_density(x_values, t_val, coefficients)

        # 3D figure.
        figsize = plotopts.get("figsize", (10, 8))
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        cmap = plotopts.get("cmap", "viridis")
        surf = ax.plot_surface(X, T, density, cmap=cmap, edgecolor='none')

        # Labels
        ax.set_xlabel(plotopts.get("xlabel", "Position (x)"))
        ax.set_ylabel(plotopts.get("ylabel", "Time (t)"))
        ax.set_zlabel(plotopts.get("zlabel", "|ψ(x,t)|²"))
        ax.set_title(plotopts.get("title", "Probability Density (3D)"))
        fig.colorbar(surf, shrink=0.5, aspect=5)

        # Animation
        if animate_rotation or save_as_gif:
            def update(frame):
                ax.view_init(elev=30, azim=frame)
                return [surf]

            angle_frames = np.linspace(0, 360, num_frames)
            ani = animation.FuncAnimation(fig, update, frames=angle_frames, interval=200, blit=False)

            if save_as_gif:
                ani.save(gif_filename, writer="pillow", fps=30)
                plt.close(fig)
                print(f"3D GIF saved at: '{gif_filename}'")
            else:
                plt.show()
        else:
            plt.show()

        return fig, ax, density, t_values, x_values
    
    def plot_wavefunction_and_density(self, t, coefficients, num_frames=None, num_slices=None, 
                                    save_as_gif=False, gif_filename="wavefunction_density.gif", **plotopts):
        """
        Plots |ψ(x,t)| (modulus) and |ψ(x,t)|² (probability density) as static or animated plots.

        Parameters:
            - t: Time instant (float) or interval (tuple [t1, t2]) for animation.
            - coefficients: List of coefficients for the wave function.
            - num_frames (optional): Number of frames for animation.
            - num_slices (optional): Number of x-points (default 200).
            - save_as_gif (bool): Save animation or static frame as GIF.
            - gif_filename (str): Filename to save.
            - **plotopts: Additional plot options like title, xlabel, etc.
        """
        if num_slices is None:
            num_slices = 200
        x_values = np.linspace(0, self.length, num_slices)

        def get_data(t_val):
            psi = np.array([self.wave_function(x, t_val, coefficients) for x in x_values])
            return np.abs(psi), np.abs(psi)**2

        if isinstance(t, (tuple, list)) and len(t) == 2:
            # Animation mode
            if num_frames is None:
                num_frames = 100
            times = np.linspace(t[0], t[1], num_frames)
            mod_all = []
            dens_all = []
            for t_val in times:
                mod, dens = get_data(t_val)
                mod_all.append(mod)
                dens_all.append(dens)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=plotopts.get("figsize", (12, 5)))
            line1, = ax1.plot(x_values, mod_all[0], label="|ψ(x,t)|")
            line2, = ax2.plot(x_values, dens_all[0], label="|ψ(x,t)|²", color="orange")
            ax1.set_title("Wavefunction Modulus |ψ(x,t)|")
            ax2.set_title("Probability Density |ψ(x,t)|²")
            for ax in (ax1, ax2):
                ax.set_xlabel("x")
                ax.grid(True)
            ax1.set_ylim(0, 1.1 * np.max(mod_all))
            ax2.set_ylim(0, 1.1 * np.max(dens_all))

            def update(i):
                line1.set_ydata(mod_all[i])
                line2.set_ydata(dens_all[i])
                fig.suptitle(f"t = {times[i]:.2f}")
                return line1, line2

            ani = animation.FuncAnimation(fig, update, frames=len(times), interval=500, blit=True)

            if save_as_gif:
                ani.save(gif_filename, writer="pillow", fps=5)
                plt.close(fig)
                print(f"GIF saved as: {gif_filename}")
            else:
                plt.show()

        else:
            # Static mode
            mod_vals, dens_vals = get_data(t)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=plotopts.get("figsize", (12, 5)))
            ax1.plot(x_values, mod_vals, label="|ψ(x,t)|")
            ax2.plot(x_values, dens_vals, label="|ψ(x,t)|²", color="orange")
            ax1.set_title("Wavefunction Modulus |ψ(x,t)|")
            ax2.set_title("Probability Density |ψ(x,t)|²")
            for ax in (ax1, ax2):
                ax.set_xlabel("x")
                ax.grid(True)
            fig.suptitle(f"t = {t:.2f}")

            if save_as_gif:
                plt.savefig(gif_filename, dpi=150)
                plt.close()
                print(f"Image saved as: {gif_filename}")
            else:
                plt.show()

    def probability_density_cartoon(self,  t, coefficients, num_slices=None, save_as_gif=True, gif_fps=30, cmap='plama',                                  **plotopts):
        """
            Display or save as GIF a "cartoon" of the probability density.

            - If t is a float → generates a static plot at time t.
            - If t is a tuple (t0, t1) → generates an animation from t0 to t1.

            Parameters
            ----------
            num_slices : int, optional
                Number of x samples to use (default: 500).
            save_as_gif : bool, optional
                If True, saves the animation to `gif_filename`.
            gif_filename : str, optional
                Name of the output GIF file.
            gif_fps : int, optional
                Frames per second of the GIF.
            **plotopts : dict, optional
                Additional options passed to plt.imshow (e.g., cmap="viridis").
        """
       
        gif_filename = f"{self.root_filename}_density_cartoon.gif"
        
        resolution = 500
        x = np.linspace(0, self.length, resolution)

        def make_image(t_val):
            # Use negative density values to emphasize regions of higher probability.
            dens = -self.probability_density(x, t_val, coefficients)
            # Construct a 2D image by replicating the density values along the vertical axis.
            return np.tile(dens, (50, 1))

        # extent and style parameters
        extent = [0, self.length, 0, 1]
        im_kwargs = dict(extent=extent, aspect="auto", cmap=cmap,  **plotopts)

        if isinstance(t, (int, float)):
            # STATIC PLOT
            fig, ax = plt.subplots()
            img = make_image(t)
            im = ax.imshow(img, **im_kwargs)
            ax.set_xlabel("x")
            ax.set_yticks([])
            ax.set_title(f"Probability Density at t={t:.3f}")
            fig.colorbar(im, ax=ax)
            if save_as_gif:
                # save single frame as GIF
                fig.savefig(gif_filename, format="gif")
                plt.close(fig)
                print(f"Saved static density as GIF in file: {gif_filename}")
            else:
                plt.show()

        elif isinstance(t, (tuple, list)) and len(t) == 2:
            # ANIMATION
            t0, t1 = t
            if num_slices is None or num_slices <= 0:
                num_slices = 10
            times = np.linspace(t0, t1, num_slices)

            fig, ax = plt.subplots()
            img0 = make_image(times[0])
            im = ax.imshow(img0, **im_kwargs)
            ax.set_xlabel("x")
            ax.set_yticks([])
            fig.colorbar(im, ax=ax)

            def update(frame_t):
                im.set_data(make_image(frame_t))
                ax.set_title(f"t = {frame_t:.3f}")
                return (im,)

            ani = FuncAnimation(
                fig,
                update,
                frames=times,
                interval=500,
                blit=True,
                repeat=True
            )

            if save_as_gif:
                ani.save(gif_filename, writer="pillow", fps=gif_fps)
                plt.close(fig)
                print(f"Animation saved as GIF in file: {gif_filename}")
            else:
                plt.show()

        else:
            raise ValueError("t must be a number or a tuple/list [t0, t1].")
        
    # --- Methods for Calculate Position ---
    
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
        
        n_points = 10000
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

        n_points = 10000
        def exp_pos_sq(t):
            # Defines the integrand: x^2 * w(x) * ProbabilityDensity(x, t, coefficients)`
            integrand_num = lambda x: self.probability_density(x, t, coefficients)*(x**2)*self.weight(x)
            numerator, err1 = fixed_quad(integrand_num, 0.0, self.length, n=n_points)
          
            # Defines the integrand: ∫₀ᴸ [w(x) * |ψ(x,t)|²]
            integrand_den = lambda x: self.probability_density(x, t, coefficients)*self.weight(x)
            denominator, err1 = fixed_quad(integrand_den, 0.0, self.length, n=n_points)
            
            return numerator/denominator
        
        return exp_pos_sq

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
            save_filename = f"{self.root_filename}_min_relative_uncertainty.txt"

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

        # Save results to file (now includes all sampled points)
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
        fig.savefig(f"{self.root_filename}_RelativePositionUncertainty.png", bbox_inches='tight')
        plt.show()

        return fig, min_val, min_times
   
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
        filename = f"{self.root_filename}_Position.txt"
        
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
            filename = f"{self.root_filename}_PositionUncertainty.txt"
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
        filename = f"{self.root_filename}_Position.txt"
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
        plt.savefig(f"{self.root_filename}_ExpectedPosition.png", bbox_inches='tight')
        plt.show()

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
        filename = f"{self.root_filename}_PositionUncertainty.txt"
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
        plt.savefig(f"{self.root_filename}_ExpectedPositionAndUncertainty.png", bbox_inches='tight')
        plt.show()
 
    def analyze_expected_position(self, tolerance=0.05, save=True, show=True):
        """
        Reads the file `<root_filename>_Position.txt` and analyzes patterns in the data:
            - Local minima and maxima (values and corresponding times)
            - Average oscillation period
            - Average amplitude
            - Mean value
            
        Saves the results to a TXT file and generates a plot with the highlighted points.

        """

        filename = f"{self.root_filename}_Position.txt"
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
        out_file = f"{self.root_filename}_PositionAnalysis.txt"
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

        # --- Gráfico ---
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
            plt.savefig(f"{self.root_filename}_ExpectedPosition_Analysis.png", dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
 
 # --- Methods for Calculate Momentum ---
 
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
            save_filename = f"{self.root_filename}_min_relative_momentum_uncertainty.txt"

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
        fig.savefig(f"{self.root_filename}_RelativeMomentumUncertainty.png", bbox_inches='tight')
        plt.show()

        return fig, min_val, min_times

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

        filename = f"{self.root_filename}_MomentumUncertainty.txt"
        
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
        filename = f"{self.root_filename}_MomentumUncertainty.txt"
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
        plt.savefig(f"{self.root_filename}_ExpectedMomentumAndUncertainty.png", bbox_inches='tight')
        plt.show()
        
   # --- Methods for Comparing Problems ---
   
    def its_number_of_digits(self):
        '''
            Retorna o número de dígitos atualmente utilizados pelo objeto.
        '''
        return self.digits_used
   
    def is_described(self):
        """
            # Display the problem description on the screen.
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

    def is_described_to_file(self, filename):
        """
            Write the problem description to a file.  

            Args:  
            - filename (str): Name of the file where the description will be written.  
        """
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
    
    def compare_numerically(self, other):
        """
            Numerically compares the spectra (eigenvalues) of self (caller) and another instance (other).  

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
        # Checks if both problems have already had their spectra calculated.
        if not (self.has_spectrum_been_calculated and other.has_spectrum_been_calculated):
            print("Energy spectra have not been calculated for both problems.")
            return

        # Retrieves the spectra (lists of eigenvalues).
        pSp = self.its_eigenvalues()   # Spectrum of the caller problem.
        qSp = other.its_eigenvalues()   # Spectrum of the called problem.
        NumberOfLevels = min(len(pSp), len(qSp))
        
        # Retrieves the number of digits (assuming the class stores this information).
        pDig = self.num_digits
        qDig = other.num_digits
        
        # Defines the output filename.
        filename = f"{self.root_filename}_Comparison_Eigenvalues.txt"
        separator = "#" + "=" * 80 + "\n"
        
        # Opens (or creates) the file and writes the header.
        with open(filename, "w", encoding='utf-8') as f:
            f.write(f"# Filename: {filename}\n")
            f.write("# Comparison of the eigenvalues of two different problems.\n")
            f.write(f"# Program version: {self.version}\n")
            f.write(f"# Start: {datetime.datetime.now().strftime('%c')}\n")
            f.write(separator)
            f.write("# Description of the caller problem\n")
        
        # Writes the description of the **caller** in the file  
        # (assuming `is_described_to_file` writes to the file).
        self.is_described_to_file(filename)
        
        # Adds the description of the **called** problem.
        with open(filename, "a") as f:
            f.write("# Description of the called problem\n")
        other.is_described_to_file(filename)
        
        # Adds information about the number of eigenvalues and digits used.
        with open(filename, "a") as f:
            f.write(f"# Caller problem has {len(pSp)} eigenvalues, calculated with {pDig} Digits\n")
            f.write(f"# Called problem has {len(qSp)} eigenvalues, calculated with {qDig} Digits\n")
            f.write(separator)
            f.write(f"# Absolute and percent variations of the lowest {NumberOfLevels} energy levels:\n")
            f.write("# Columns: (1) level, (2) caller eigenvalue, (3) called eigenvalue, (4) absolute variation (called - caller), (5) percent variation\n")
            f.write(separator)
        
        # For each level (up to the minimum number of levels between the two problems),  
        # calculates the absolute variation and percentage variation.
        for i in range(NumberOfLevels):
            err1 = qSp[i] - pSp[i]
            # Avoids division by zero if `pSp[i]` is very close to zero.
            if math.isclose(pSp[i], 0, abs_tol=1e-15):
                err2 = 0.0
            else:
                err2 = (err1 / abs(pSp[i])) * 100
            with open(filename, "a") as f:
                f.write(f"{i+1}\t{pSp[i]:.6f}\t{qSp[i]:.6f}\t{err1:.6f}\t{err2:.6f}\n")
        
        print(f"Comparison of spectra has been written in the file {filename}")
        print("Done")
    
    def compare_graphically(self, other):
        """
            Graphically compares the eigenvalues of the two problems.

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
        # Checks if both spectra have been calculated.
        if not (self.has_spectrum_been_calculated and hasattr(self, 'eigenvectors')):
            print("Please solve the eigenvalue and eigenvector problem first.")
            return
        
        # Displays the descriptions of the problems.
        print("Description of the caller problem:")
        self.is_described()
        print("\n")
        print("Description of the called problem:")
        other.is_described()
        print("\n")
        
        # Retrieves the number of spectra.
        pSp = self.its_eigenvalues()
        qSp = other.its_eigenvalues()      
        
        NumberOfLevels = min(len(pSp), len(qSp))
        
        # Initializes lists for absolute and percentage variations.
        pontosPos = []         # diffs > 0: Positive variations
        pontosNeg = []         # diffs < 0: Negative variations
        pontosNull = []        # diffs == 0: Zero variations
        pontosPercentPos = []
        pontosPercentNeg = []
        pontosPercentNull = []
        
        for i in range(NumberOfLevels):
            diff = qSp[i] - pSp[i]
            if math.isclose(diff, 0, abs_tol=1e-15):
                pontosNull.append((i+1, 0))
                pontosPercentNull.append((i+1, 0))
            elif diff > 0:
                pontosPos.append((i+1, math.log10(diff)))
                percent = (diff / abs(pSp[i])) * 100 if not np.isclose(pSp[i],0, atol=1e-15) else 0
                pontosPercentPos.append((i+1, percent))
            else:
                pontosNeg.append((i+1, math.log10(-diff)))
                percent = (-diff / abs(pSp[i])) * 100 if not np.isclose(pSp[i],0, atol=1e-15) else 0
                pontosPercentNeg.append((i+1, percent))
                
        # First plot: log10 of absolute variations.
        print("-" * 80)
        print("Log10 Absolute Variation.")
        print("\n")
        plt.figure(figsize=(10, 5))
        if pontosPos:
            pts = np.array(pontosPos)
            plt.scatter(pts[:, 0], pts[:, 1], color='blue', label="Positive differences (log10)")
        if pontosNeg:
            pts = np.array(pontosNeg)
            plt.scatter(pts[:, 0], pts[:, 1], color='red', label="Negative differences (log10)")
        if pontosNull:
            pts = np.array(pontosNull)
            plt.scatter(pts[:, 0], pts[:, 1], color='green', label="Zero differences")
            
        plt.xlabel("Energy level")
        plt.ylabel("log10(|variation|)")
        plt.title(f"Log10 of absolute variations (called - caller) for the lowest {NumberOfLevels} energy levels")
        plt.grid(True)
        plt.legend(loc="upper left", bbox_to_anchor=(1.00, 1))
        plt.tight_layout()
        plt.show()

        # Second plot: percentage variations.
        print("-" * 80)
        print("Percent variation.")
        print("\n")
        
        plt.figure(figsize=(10, 5))
        if pontosPercentPos:
            pts = np.array(pontosPercentPos)
            plt.scatter(pts[:, 0], pts[:, 1], color='blue', label="Positive percent variation")
        if pontosPercentNeg:
            pts = np.array(pontosPercentNeg)
            plt.scatter(pts[:, 0], pts[:, 1], color='red', label="Negative percent variation")
        if pontosPercentNull:
            pts = np.array(pontosPercentNull)
            plt.scatter(pts[:, 0], pts[:, 1], color='green', label="Zero variation")
        plt.xlabel("Energy level")
        plt.ylabel("Percent variation (%)")
        plt.title(f"Percent variations [(called - caller)/caller] for the lowest {NumberOfLevels} energy levels")
        plt.grid(True)
        plt.legend(loc="upper left", bbox_to_anchor=(1.00, 1))
        plt.show()

        print("-" * 80)
        print("Comparing orthogonality of eigenvectors (dot product).")
        print("\n")
        
        # Assuming 'sols' is the list of solutions and 'nsols' is its size.
        sols = self.its_eigenpairs(save_to_file=False)
        nsols = len(sols)       
        
       # Creates deviation matrix.
        ort_matrix = np.zeros((nsols, nsols))
        for i in range(nsols):
            for j in range(i+1, nsols):
                err = abs(self.__scalar_product(sols[i][1], sols[j][1]))
                ort_matrix[i, j] = err

        # Plot heatmap
        plt.figure(figsize=(10, 5))
        im = plt.imshow(np.log10(ort_matrix + 1e-30), cmap="viridis", origin="lower", aspect="auto")

        plt.colorbar(im, label="log10(|<vi,vj>|)")
        plt.xlabel("Level i")
        plt.ylabel("Level j")
        plt.title("Orthogonality deviation between eigenvectors")

        plt.tight_layout()
        plt.show()
        
        # Plot Any Comparations
        for i in range(nsols - 1):
            print(f"\nComparing Level {i + 1} to upper levels.")
            pontosOrt = []      # For nonzero deviations.
            pontosOrtZero = []  # For zero deviations.
            for j in range(i + 1, nsols):
                # Calculates the orthogonality error
                err = abs(self.__scalar_product(sols[i][1], sols[j][1]))
                err_scaled = err 
                if np.isclose(err_scaled, 0, atol=1e-15):
                    pontosOrtZero.append((j + 1, 0))
                else:
                    pontosOrt.append((j + 1, math.log10(err_scaled)))
            
            plt.figure(figsize=(10, 5))
            if pontosOrt:
                pts_ort = np.array(pontosOrt)
                plt.plot(pts_ort[:, 0], pts_ort[:, 1], 'r-o', label="Non-zero deviation")
                tick_interval = max(1, nsols // 10)
                plt.xticks(np.arange(i + 2, nsols + 1, tick_interval), rotation=45)
                # Obtains the real values (undoing the log) to determine a suitable linthresh.
                real_vals = [10 ** y for y in pts_ort[:, 1] if y != 0]
                if real_vals:
                    linthresh = max(1e-8, np.min(real_vals))
                    plt.yscale('symlog', linthresh=linthresh)
                    y_min, y_max = np.min(pts_ort[:, 1]), np.max(pts_ort[:, 1])
                    plt.ylim(y_min - 0.1, y_max + 0.1)
                else:
                    plt.yscale('symlog')
            if pontosOrtZero:
                pts_ort0 = np.array(pontosOrtZero)
                plt.plot(pts_ort0[:, 0], pts_ort0[:, 1], 'g-o', label="Zero deviation")
            
            plt.xlabel("Energy level")
            plt.ylabel("log10(|<vi, vj>|)")
            plt.title(f"Orthogonality error: Level {i + 1} compared to upper levels")
            
            ax = plt.gca()
            # Defines minor ticks for higher resolution and formats the y-axis in scientific notation.
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e'))
            plt.legend()
            plt.tight_layout()
            plt.xlim(i + 2, nsols + 0.5)
            plt.show()

    def check_solutions_numerically(self):
        """
            Numerically verifies the solutions (eigenvalues and eigenvectors) according to the following criteria:

            (1) Equation error: Calculates the maximum absolute value of the elements of M(E)v,  
            where M(E) = FirstBigMatrix() - E * SecondBigMatrix().
            (2) Normality deviation: Computes |1 - ⟨v, v⟩|.
            (3) Orthogonality deviation: For each pair (vi, vj), calculates ⟨vi, vj⟩.

            - The results are saved in text files for further analysis.

            - Note:  
            It is assumed that FirstBigMatrix() and SecondBigMatrix() are already implemented,  
            and that the spectrum (sols) is obtained via its_eigenvectors(), returning a list of pairs [E, v].
        """
        if not self.has_eigenvectors_been_calculated:
            print("Please solve the eigenvalue and eigenvector problem first.")
            return

        # ------------------ 1. Verification: Substitution in M(E)v = 0 ------------------
        filename_eq = f"{self.root_filename}_Check_Equations.txt"
        print("Reporting errors in equations in file:", filename_eq)
        
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

        # Retrieves the solutions (list of `[E, eigenvector]`).
        sols = self.eigenvectors
        nsols = len(sols)
        
        # For each solution, calculates `M_E = FirstBigMatrix() - E * SecondBigMatrix()`  
        # and finds the maximum error.
        for i in range(nsols):
            E = sols[i][0]
            v = np.array(sols[i][1])
            M_E = self.__FirstBigMatrix() - E * self.__SecondBigMatrix()
            Av = M_E.dot(v)
            err = np.max(np.abs(Av))
            with open(filename_eq, "a") as f:
                f.write(f"{i+1}\t{err:.20e}\n")

        # ------------------ 2. Verification: Normality of Eigenvectors ------------------
        filename_norm = f"{self.root_filename}_Check_Normality.txt"
        print("Reporting errors in eigenvector normality in file:", filename_norm)
        
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
            # v should be interpreted as stored; assumes self.__scalar_product is defined.
            v = sols[i][1]
            err = abs(1 - self.__scalar_product(v, v))
            with open(filename_norm, "a") as f:
                f.write(f"{i+1}\t{err:.20e}\n")
        
        # ------------------ 3. Verification: Orthogonality between Eigenvectors ------------------
        filename_ortho = f"{self.root_filename}_Check_Orthogonality.txt"
        print("Reporting errors in eigenvector orthogonality in file:", filename_ortho)
        
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
        
        # For each pair (i, j) with i < j, calculates the dot product and records the error (saving only the real part).
        for i in range(nsols):
            for j in range(i+1, nsols):
                err = self.__scalar_product(sols[i][1], sols[j][1])
                if not np.isclose(np.imag(err), 0, atol=1e-15):
                    print(f"Warning: scalar product of levels {i+1} and {j+1} yielded a complex number. Recording just the real part.")
                err = np.real(err)
                with open(filename_ortho, "a") as f:
                    f.write(f"{i+1}\t{j+1}\t{err:.20e}\n")
        
        print("Done.")
        
    def check_solutions_graphically(self, save: bool = True, show: bool = True):
        """
            Graphically checks the quality of eigenvalues/eigenvectors:
            - Substitution errors (Mv ≈ 0)
            - Normality deviations (||v|| ≈ 1)
            - Orthogonality deviations (<vi,vj> ≈ 0)

            If the check files already exist, use their data.
            Otherwise, recalculate.
        """

        if not self.has_eigenvectors_been_calculated:
            print("Please solve the eigenvalue and eigenvector problem first.")
            return

        sols = self.eigenvectors
        nsols = len(sols)

        # ---------------- Substitution Error ----------------
        print("-"*80)
        print("---------------- Substitution Error ----------------")
        filename_eq = f"{self.root_filename}_Check_Equations.txt"
        errors_subst = []

        if os.path.exists(filename_eq):
            print(f"Reading substitution errors {filename_eq}")
            data = np.loadtxt(filename_eq, comments="#")
            for row in data:
                level, err = int(row[0]), float(row[1])
                log_err = np.log10(err) if err > 0 else -30
                errors_subst.append((level, log_err))
        else:
            print("Calculating substitution errors...")
            for i, (E, v) in enumerate(sols, start=1):
                M_E = self.__FirstBigMatrix() - E * self.__SecondBigMatrix()
                err = np.max(np.abs(M_E.dot(v)))
                log_err = np.log10(err) if err > 0 else -30
                errors_subst.append((i, log_err))

        plt.figure(figsize=(10, 5))
        pts = np.array(errors_subst)
        plt.plot(pts[:, 0], pts[:, 1], 'r-o', label="log10(|M(E)v|)")
        plt.xlabel("En. level")
        plt.ylabel("log10(|error|)")
        plt.title("Substitution error in eigenvector equation")
        plt.legend(loc="upper left", bbox_to_anchor=(1.00, 1))

        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
        ax.xaxis.set_minor_locator(MultipleLocator(5))
        ax.yaxis.set_minor_locator(MultipleLocator(1))
        plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)
        if save:
            plt.savefig(f"{self.root_filename}_Check_Substitution.png", dpi=300)
        if show:
            plt.show()
        else:
            plt.close()

        # ---------------- Normality Deviation ----------------
        print("-"*80)
        print("---------------- Normality Deviation ----------------")
        filename_norm = f"{self.root_filename}_Check_Normality.txt"
        errors_norm = []

        if os.path.exists(filename_norm):
            print(f"Reading deviations from normality of {filename_norm}")
            data = np.loadtxt(filename_norm, comments="#")
            for row in data:
                level, err = int(row[0]), float(row[1])
                log_err = np.log10(err) if err > 0 else -30
                errors_norm.append((level, log_err))
        else:
            print("Calculating deviations from normality...")
            for i, (_, v) in enumerate(sols, start=1):
                err = abs(1 - self.__scalar_product(v, v))
                log_err = np.log10(err) if err > 0 else -30
                errors_norm.append((i, log_err))

        plt.figure(figsize=(10, 5))
        pts = np.array(errors_norm)
        plt.plot(pts[:, 0], pts[:, 1], 'bo-', label="log10(|1 - <v,v>|)")
        plt.xlabel("En. level")
        plt.ylabel("log10(|1 - <v,v>|)")
        plt.title("Normality deviation of eigenvectors")
        plt.legend(loc="upper left", bbox_to_anchor=(1.00, 1))
       
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
        ax.xaxis.set_minor_locator(MultipleLocator(5))
        ax.yaxis.set_minor_locator(MultipleLocator(1))
        plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)
        if save:
            plt.savefig(f"{self.root_filename}_Check_Normality.png", dpi=300)
        if show:
            plt.show()
        else:
            plt.close()

        # ---------------- Orthogonality Summary (1D curve) ----------------
        print("-"*80)
        print("---------------- Creating summary plot of orthogonality deviation (max per level) ----------------")

        ortho_summary = []
        for i in range(nsols):
            errs_i = []
            for j in range(nsols):
                if i != j:
                    err = abs(self.__scalar_product(sols[i][1], sols[j][1]))
                    log_err = np.log10(err) if err > 0 else -30
                    errs_i.append(log_err)
            if errs_i:
                ortho_summary.append((i+1, max(errs_i)))  # pega o pior caso

        plt.figure(figsize=(10, 5))
        pts = np.array(ortho_summary)
        plt.plot(pts[:, 0], pts[:, 1], 'r-o', label="log10(|<vi,vj>|) max")
        plt.xlabel("Energy level")
        plt.ylabel("log error in orthogonality")
        plt.title("Maximum orthogonality deviation per level")
        plt.legend(loc="upper left", bbox_to_anchor=(1.00, 1))

        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
        ax.xaxis.set_minor_locator(MultipleLocator(5))
        ax.yaxis.set_minor_locator(MultipleLocator(1))

        plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)
        if save:
            plt.savefig(f"{self.root_filename}_Check_Orthogonality_Summary.png", dpi=300)
        if show:
            plt.show()
        else:
            plt.close()
            
        # ---------------- Orthogonality Summary (2D curve) ----------------
        print("-"*80)
        print("---------------- Orthogonality Summary (2D curve) ----------------")
        filename_ortho = f"{self.root_filename}_Check_Orthogonality.txt"
        errors_ortho = []

        if os.path.exists(filename_ortho):
            print(f"Reading orthogonality deviations from {filename_ortho}")
            data = np.loadtxt(filename_ortho, comments="#")
            for row in data:
                i, j, err = int(row[0]), int(row[1]), float(row[2])
                log_err = np.log10(abs(err)) if err != 0 else -30
                errors_ortho.append((i, j, log_err))
        else:
            print("Calculating orthogonality deviations...")
            for i in range(nsols):
                for j in range(i+1, nsols):
                    err = abs(self.__scalar_product(sols[i][1], sols[j][1]))
                    log_err = np.log10(err) if err > 0 else -30
                    errors_ortho.append((i+1, j+1, log_err))

        plt.figure(figsize=(10, 5))
        pts = np.array(errors_ortho)
        sc = plt.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], cmap="viridis", marker="o")
        plt.colorbar(sc, label="log10(|<vi,vj>|)")
        plt.xlabel("Level i")
        plt.ylabel("Level j")
        plt.title("Orthogonality deviation between eigenvectors")
       
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
        ax.xaxis.set_minor_locator(MultipleLocator(5))
        ax.yaxis.set_minor_locator(MultipleLocator(1))
        plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)
        if save:
            plt.savefig(f"{self.root_filename}_Check_Orthogonality.png", dpi=300)
        if show:
            plt.show()
        else:
            plt.close()

        print("="*80)
        print("Done.")

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
            if not (self.has_spectrum_been_calculated and self.has_eigenvectors_been_calculated):
                    #raise RuntimeError("No eigenvectors have been computed yet.")
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
    
# --- Method for Compute Optimal L ---

    def __build_D_at(self, L: float) -> np.ndarray:
        """
        Monta a matriz D(L) = T(L) + C(L) sem alterar permanentemente self.length.
        """
        old_L = self.length
        self.length = L
        D = self.__build_matrix_d()    
        self.length = old_L
        return D
 
    def __energy_ground_state(self, L: float) -> float:
        """
        Returns the smallest eigenvalue of D(L), i.e., E₀(L).
        """
        D = self.__build_D_at(L)
        # returns the eigenvalues in ascending order.
        return np.linalg.eigh(D)[0][0]
 
    def optimize_length(self, L_min = 0.1, L_max = None, xatol=1e-15):
        """
        Searches in [L_min, L_max] for the L that minimizes E₀(L).
        """
        L_max = 5*self.length
        
        res = minimize_scalar(
            lambda L: self.__energy_ground_state(L),
            bounds=(L_min, L_max),
            method='bounded',
            options={'xatol': xatol}
        )
        return res.x   

    def optimal_L(self, L=10):
        """
        Determines the optimal L for the 1D Schrödinger equation using the spectral method.

        Parameters
        ----------
        L : float
            Initial guess for the domain length.

        Returns
        -------
        L_optimal : float
            Optimal value of L.
        E0 : float
            Ground-state energy (lowest eigenvalue of the Hamiltonian).

        """

        tol = 1e-4
        V = self.f_function          # potential V(x)
        N = self.num_levels          # Number of basis functions.
        weight_function = self.weight
        L_guess = L

        def energy_for_L(L):
            """Calcula a energia fundamental para um dado L."""
            m_values = np.arange(1, N + 1)

            # Kinetic matrix (diagonal).
            # Normalization: <sin(mπx/L), sin(nπx/L)> = L/2 δ_mn
            K = np.diag((np.pi**2) * (m_values**2) / (2 * L**2))

            # Potential matrix
            V_mat = np.zeros((N, N))

            def integrand(x, m, n):
                psi_m = np.sqrt(2/L) * np.sin((m * np.pi * x) / L)
                psi_n = np.sqrt(2/L) * np.sin((n * np.pi * x) / L)
                return psi_m * V(x) * psi_n * weight_function(x)

            for i in range(N):
                for j in range(i, N):
                    m_val, n_val = i + 1, j + 1
                    try:
                        integral, _ = fixed_quad(integrand, 0, L, args=(m_val, n_val), n=10000)
                    except Exception:
                        integral = 0.0
                    V_mat[i, j] = integral
                    if i != j:
                        V_mat[j, i] = integral

            H = K + V_mat

            # Smallest eigenvalue (ground-state energy)
            try:
                E0 = eigsh(H, k=1, which='SA', return_eigenvectors=False)[0]
            except Exception:
                E0 = np.linalg.eigh(H)[0][0]

            return E0

        # Minimization with respect to L.
        result = minimize_scalar(
            lambda L: energy_for_L(L),
            bounds=(0.1, 5 * L_guess),
            method='bounded',
            tol=tol
        )

        L_optimal = result.x
        E0 = result.fun

        return L_optimal, E0
    
# --- Method for Unified Compute ---

    def run_full_analysis(self, coeficientes, tempo: float = 100.0, pontos: int = 200):
        """
        Executes a complete analysis routine of the quantum system,
        calling in sequence the main methods already implemented.

        Parameters
        ----------
        coefficients : list[complex]
            Expansion coefficients of the wave function.
        time : float
            Maximum time for time-dependent calculations (default=100).
        points : int
            Number of sampling points for time-dependent functions (default=200).

        """
        
        print("="*80)
        print("\n")
        print(">>>>>>> Function Settings")
        print(f"Time set:   {tempo}")
        print(f"Points:     {pontos}")
        print("\n")
        
        print("="*80)
        print("\n")
        print(">>>>>>> DESCRIBED SYSTEM")
        print("\n")
        self.is_described()
        print("\n")
        
        print("="*80)
        print("\n")
        print(">>>>>>> SYSTEM SOLVE...")
        print("\n")
        self.is_solved()
        print("\n")
        self.calculate_spectrum()
        print("\n")
        self.calculate_eigenvectors()
        print("\n")
        self.its_eigenpairs()
        print("\n")

        print("="*80)
        print("\n")
        print(">>>>>>> EXPECTED POSITION")
        print("\n")
        self.expected_position_is_calculated((0.0, tempo), coeficientes, num_points=pontos)
        print("\n")
        self.expected_position_is_plotted()
        print("\n")

        print("="*80)
        print("\n")
        print(">>>>>>> UNCERTAINTY AND EXPECTED POSITION")
        print("\n")
        self.expected_position_and_uncertainty_are_calculated((0.0, tempo), coeficientes, num_points=pontos)
        print("\n")
        self.expected_position_and_uncertainty_are_plotted()
        print("\n")

        print("="*80)
        print("\n")
        print(">>>>>>> UNCERTAINTY AND EXPECTED MOMENTUM")
        print("\n")
        self.expected_momentum_and_uncertainty_are_calculated((0.0, tempo), coeficientes, pontos)
        print("\n")
        self.expected_momentum_and_uncertainty_are_plotted()
        print("\n")

        print("="*80)
        print("\n")
        print(">>>>>>> WAVEFUNCTION AND DENSITY PROBABILITY")
        print("\n")
        self.plot_wavefunctions(num_levels=3, save=True)
        print("\n")
        self.probability_density_is_plotted(0, coeficientes, num_frames=50, save=True)
        print("\n")
        self.probability_density_is_plotted((0,tempo), coeficientes, num_frames=50, save=True)
        print("\n")
        self.plot_wavefunction_and_density( 
        t=0, 
        coefficients=coeficientes, 
        num_slices=300, 
        color_psi='blue', 
        linestyle_psi='-', 
        color_rho='red', 
        alpha_rho=0.5, 
        title='|psi| e |psi|² em t=0', 
        xlabel='x', 
        )
        print("\n")

        print("="*80)
        print("\n")
        print(">>>>>>> UNCERTAINTY RELATIVE")
        print("\n")
        self.position_uncertainty_relative(coeficientes, t_max=tempo, num_points=pontos)
        print("\n")
        self.momentum_uncertainty_relative(coeficientes, t_max=tempo, num_points=pontos)
        print("\n")

        print("="*80)
        print("\n")
        print(">>>>>>> PROBABILITY DENSITY (CARTOON)")
        print("\n")
        self.probability_density_cartoon(
            t=(0, tempo),
            coefficients=coeficientes,
            num_slices=20,
            cmap='plasma',
            alpha=0.8,
        )
        print("\n")
        print("="*80)
        print("\n")
        print(">>>>>>> UNCERTAINTY AND EXPECTED MOMENTUM (NUMERIC)")
        print("\n")
        self.expected_momentum_and_uncertainty_are_calculated((0, tempo), coeficientes, pontos)
        self.expected_momentum_and_uncertainty_are_plotted()

        print("="*80)
        print("\n")
        print(">>>>>>> NUMERIC FUNCTIONS: <p> e dp")
        print("\n")
        expected_momentum_func = self.expected_momentum(coeficientes)
        print("\n")
        momentum_uncertainty_func = self.momentum_uncertainty(coeficientes)
        for i in range(0, int(tempo)+1, 5):
            print(f"t={i}: <p> = {expected_momentum_func(i)} | dp = {momentum_uncertainty_func(i)}")
        print("\n")

        print("="*80)
        print("\n")
        print(">>>>>>> NORM OF WAVE FUNCTION")
        print("\n")
        n = self.norm_of_wave_function(coeficientes)
        for t in np.linspace(0, tempo, 100):
            print(f"Norm at t={t:.1f}: {n(t):.20}")
        print("\n")

        print("="*80)
        print("\n")
        print(">>>>>>> HEISENBERG UNCERTAINTY")
        print("\n")
        heis_fn = self.heisenberg_uncertainty(coeficientes)
        for i in range(0, int(tempo)+1, 5):
            result = heis_fn(t=i)
            print(f"t={i} | dx.dp = {result['dx.dp']} | Heisenberg satisfy? {result['valid']}")
        print("\n")

        print("="*80)
        print("\n")
        print(">>>>>>> ANALYZE EXPECTED POSITION")
        print("\n")
        self.analyze_expected_position()
        print("\n")

        print("="*80)
        print("\n")
        print(">>>>>>> CHECK SOLUTIONS NUMERICALLY AND GRAPHICALLY")
        print("\n")
        self.check_solutions_numerically()
        print("\n")
        self.check_solutions_graphically()
        print("\n")

        print("="*80)
        print(">>>>>>> DONE!")
         