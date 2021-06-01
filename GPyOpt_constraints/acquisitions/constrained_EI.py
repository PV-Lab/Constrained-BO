# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .base import AcquisitionBase
from ..util.general import get_quantiles
from scipy.interpolate import griddata
import numpy as np

class AcquisitionConstrainedEI(AcquisitionBase): # Aleks edit
    """
    Constrained expected improvement acquisition function # Aleks edit

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function
    :param jitter: positive value to make the acquisition more explorative.

    .. Note:: allows to compute the Improvement per unit of cost

    """

    analytical_gradient_prediction = True

    def __init__(self, model, space, constraint_function, optimizer=None, cost_withGradients=None, jitter=0.01): # Aleks edit
        self.optimizer = optimizer
        super(AcquisitionConstrainedEI, self).__init__(model, space, optimizer, cost_withGradients=cost_withGradients) # Aleks edit
        self.jitter = jitter
        self.constraint_function = constraint_function # Aleks edit

    @staticmethod
    def fromConfig(model, space, constraint_function, optimizer, cost_withGradients, config): # Aleks edit
        return AcquisitionConstrainedEI(model, space, constraint_function, optimizer, cost_withGradients, jitter=config['jitter']) # Aleks edit

    def _compute_acq(self, x): # Aleks edit
        """
        Computes the Expected Improvement per unit of cost
        """
        m, s = self.model.predict(x)
        fmin = self.model.get_fmin()
        phi, Phi, u = get_quantiles(self.jitter, fmin, m, s)

        # Grab the value of the constraint function by locating the constraint value using the x coordinates
        # without interpolating, the x-coords E[0,1] are used as indices on the 100x100 constraint mesh
        # by rounding the x-coords. An abs and -1 are introduced to fix problem of index 100 out of range when x=1.
        if x.shape[1] == 2: # if x-coords are 2D
            constr = (
            self.constraint_function[abs((x[:, 1].round(2) * 100 - 1).astype(int)), abs((x[:, 0].round(2) * 100 - 1).astype(int))]).reshape(
                x.shape[0], 1)  # reverse coordinate order
        elif x.shape[1] == 3: # if x-coords are 3D
            X = self.constraint_function[0,:,:]
            Y = self.constraint_function[1,:,:]
            Z = self.constraint_function[2,:,:]
            constrX = (X[abs((x[:, 1].round(2) * 100 - 1).astype(int)), abs((x[:, 0].round(2) * 100 - 1).astype(int))]).reshape(
                x.shape[0], 1)  # reverse coordinate order
            constrY = (Y[abs((x[:, 2].round(2) * 100 - 1).astype(int)), abs((x[:, 1].round(2) * 100 - 1).astype(int))]).reshape(
                x.shape[0], 1)  # reverse coordinate order
            constrZ = (Z[abs((x[:, 0].round(2) * 100 - 1).astype(int)), abs((x[:, 2].round(2) * 100 - 1).astype(int))]).reshape(
                x.shape[0], 1)  # reverse coordinate order
            constr = (constrX + constrY + constrZ) / 3. # take linear interpolation of 3D space

        f_acqu = s * (u * Phi + phi) * constr

        return f_acqu

    def _compute_acq_withGradients(self, x):
        """
        Computes the Expected Improvement and its derivative (has a very easy derivative!)
        """
        fmin = self.model.get_fmin()
        m, s, dmdx, dsdx = self.model.predict_withGradients(x)
        phi, Phi, u = get_quantiles(self.jitter, fmin, m, s)

        # Grab the value of the constraint function by locating the constraint value using the x coordinates
        # without interpolating, the x-coords E[0,1] are used as indices on the 100x100 constraint mesh
        # by rounding the x-coords. An abs and -1 are introduced to fix problem of index 100 out of range when x=1.
        if x.shape[1] == 2: # if x-coords are 2D
            constr = (
            self.constraint_function[abs((x[:, 1].round(2) * 100 - 1).astype(int)), abs((x[:, 0].round(2) * 100 - 1).astype(int))]).reshape(
                x.shape[0], 1)  # reverse coordinate order
        elif x.shape[1] == 3: # if x-coords are 3D
            X = self.constraint_function[0,:,:]
            Y = self.constraint_function[1,:,:]
            Z = self.constraint_function[2,:,:]
            constrX = (X[abs((x[:, 1].round(2) * 100 - 1).astype(int)), abs((x[:, 0].round(2) * 100 - 1).astype(int))]).reshape(
                x.shape[0], 1)  # reverse coordinate order
            constrY = (Y[abs((x[:, 2].round(2) * 100 - 1).astype(int)), abs((x[:, 1].round(2) * 100 - 1).astype(int))]).reshape(
                x.shape[0], 1)  # reverse coordinate order
            constrZ = (Z[abs((x[:, 0].round(2) * 100 - 1).astype(int)), abs((x[:, 2].round(2) * 100 - 1).astype(int))]).reshape(
                x.shape[0], 1)  # reverse coordinate order
            constr = (constrX + constrY + constrZ) / 3. # take linear interpolation of 3D space

        f_acqu = s * (u * Phi + phi) * constr
        df_acqu = dsdx * phi - Phi * dmdx * constr

        return f_acqu, df_acqu

