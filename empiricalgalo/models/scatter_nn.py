# Copyright (C) 2021 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

import numpy
from copy import deepcopy
from warnings import warn
import os

import tensorflow as tf
from tensorflow.keras.layers import (Input, Normalization, Dense, Add)
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import neural_structured_learning as nsl
from sklearn.metrics import r2_score

import joblib


class GaussianLossNN:
    """
    An adversarial neural network 1-dimensional regressor with a Gaussian
    loss function.

    Arguments
    ---------
    Ninputs : int
        Number of input features.
    deep_layers : list of int, optional
        Number of neurons within each deep layer.
        By default `[16, 16, 16, 16, 8]`.
    activation : str, optional
        Activation function. By default `selu`, the scaled exponential linear
        unit.
    initializer : str, optional
        Network weights initialiser, by default `LecunNormal`. Alternatively
        can be picked from Tensorflow's selection of initialisers.
    adv_multiplier : float, optinal
        Multiplier to adversarial regularization loss. By default 0.2.
    adv_step : float, optional
        Step size to find the adversarial sample. By default to 0.001.
    pgd_iters : int, optional
        Nnumber of attack iterations for Projected Gradient Descent (PGD)
        attack. Defaults to 3.
    seed : int, optional
        Random seed for setting the initial weights.
    """
    def __init__(self, Ninputs, checkpoint_dir, deep_layers=[16, 16, 16, 16, 8],
                 activation="selu", initializer="LecunNormal",
                 adv_multiplier=0.2, adv_step=0.001, pgd_iters=3, seed=None):
        # Initialise the model
        self.model, self.adv_model = self._make_model(
            Ninputs, deep_layers, activation, initializer, adv_multiplier,
            adv_step, pgd_iters, seed)

        if not os.path.isdir(checkpoint_dir):
            raise ValueError("Invalid `checkpoint_dir` `{}`"
                             .format(checkpoint_dir))
        self.checkpoint_dir = checkpoint_dir

        self._params = {"Ninputs": Ninputs, "deep_layers": deep_layers,
                        "activation": activation, "initializer": initializer,
                        "adv_multiplier": adv_multiplier, "adv_step": adv_step,
                        "pgd_iters": pgd_iters, "seed": seed}

    def _make_model(self, Ninputs, deep_layers, activation, initializer,
                   adv_multiplier, adv_step, pgd_iters, seed):
        """Make the (adversarial) model."""
        # Weights initialiser
        if initializer == "LecunNormal":
            inits = tf.keras.initializers.LecunNormal(seed)
        else:
            inits = initializer

        input_shape = (Ninputs, )

        # Linear part, directly connect to the output with no activation
        linear_input = Input(shape=input_shape, name="linear_input")
        linear = Dense(2, name="linear_layer")(linear_input)

        # Deep part, densely connected layers
        deep_input = Input(shape=input_shape, name="deep_input")
        # Normalising layer
        deep = Normalization()(deep_input)
        # Append the deep layers
        for i, layer in enumerate(deep_layers):
            deep = Dense(layer, activation=activation, kernel_initializer=inits,
                         name="deep_{}".format(i + 1))(deep)
        # Need two output nodes: mean and variance
        deep = Dense(2, name="deep_final")(deep)

        # Connect wide and deep
        final_layer = Add(name="add_linear_deep")([linear, deep])
        # The distribution at the end of the NN. Softplus transform the std to
        # ensure positivity.
        lambda_dist = lambda t: tfd.Normal(
            loc=t[..., :1], scale=1e-6 + tf.math.softplus(0.1 * t[..., 1:]))
        # Append the distribution layer
        final_layer = tfp.layers.DistributionLambda(lambda_dist)(final_layer)

        # Generate the model
        model = tf.keras.models.Model(inputs=[linear_input, deep_input],
                                      outputs=final_layer)
        adv_config = nsl.configs.make_adv_reg_config(
            multiplier=adv_multiplier, adv_step_size=adv_step,
            pgd_iterations=pgd_iters)
        adv_model = nsl.keras.AdversarialRegularization(
            model, adv_config=adv_config)

        return model, adv_model

    @staticmethod
    def _hamiltonian_loss(x, dist):
        """The Hamiltonian (negative log likelihood) loss function."""
        return -dist.log_prob(x)

    def get_callbacks(self, patience):
        """
        Get the early stopping and checkpointing callbacks. Restores the best
        weights.

        Arguments
        ---------
        patience: int
            The patience, if the loss minimum does not change over this many
            epochs terminate the training.

        Returns
        -------
        cbs: list of callbacks
            The early stopping and model checkpoint callbacks.
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, "cp.ckpt")
        return [tf.keras.callbacks.EarlyStopping(
                    patience=patience,restore_best_weights=True),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=checkpoint_path, save_weights_only=True, verbose=0)]

    def fit(self, Xtrain, ytrain, batch_size, optimizer="adamax", patience=50,
            epochs=500, validation_size=0.2):
        """
        Fit the NN with the given optimizer and save its training history and
        weights into the checkpoint folder.

        Arguments
        ---------
        Xtrain: 2-dimensional array
            Feature array.
        ytrain: 1-dimensional array
            Target array.
        batch_size: int
            The batch size.
        optimizer: keras optimizer, optional
            Optimizer to train the network. By default `adamax` with default TF
            parameters.
        patience: int, optional
            The patience, if the loss minimum does not change over this many
            epochs terminate the training. By default 50.
        epochs: int, optional
            Number of epochs to train the network, by defualt 500.
        validation_size: float, optional
            Fractional validation size.

        Returns
        -------
        None
        """
        # Compile the model
        self.adv_model.compile(optimizer=optimizer, loss=self._hamiltonian_loss)
        # Data in a format to be given to the NN
        data = {"linear_input": Xtrain,
                "deep_input": Xtrain,
                "label": ytrain}

        callbacks = self.get_callbacks(patience)

        history = self.adv_model.fit(
            x=data, batch_size=batch_size, callbacks=callbacks, verbose=0,
            epochs=epochs, validation_split=validation_size)

        # Save the history and the params for reproducibility
        joblib.dump(history.history,
                    os.path.join(self.checkpoint_dir, 'history.p'))
        joblib.dump(self._params, os.path.join(self.checkpoint_dir, 'params.p'))

    def predict(self, X, full=False):
        """
        Predict the mean or the distribution if `full`.

        Arguments
        ---------
        X: 2-dimensional array
            Feature array.
        full: bool, optional
            Whether to return the probability distribution instead. By default
            `False` and returns only the mean.

        Returns
        -------
        out: 1-dimensional array or tensor of distributions
            Predictions. If `full` returns distributions, otherwise returns the
            mean prediction.
        """
        yhat = self.model({"linear_input": X, "deep_input": X})
        if full:
            return yhat
        return numpy.asarray(yhat.mean()).reshape(-1,)

    def predict_stats(self, X):
        """
        Predict the mean and standard deviations for samples `X`.

        Arguments
        ---------
        X: 2-dimensional array
            Feature array.

        Returns
        -------
        stats: 2-dimensional array
            Array of shape (`Nsamples`, 2). The first and second columns are
            the mean and standard deviation, respectively.
        """
        yhat = self.predict(X, full=True)

        mu = numpy.asarray(yhat.mean()).reshape(-1,)
        std = numpy.asarray(yhat.stddev()).reshape(-1,)
        return numpy.vstack([mu, std]).T

    def score_R2mean(self, X, y):
        r"""
        Calculate the :math:`R^2` score of mean predictions defined as

        .. math::
            R^2 = 1 - \frac{\sum_n (\mu_n - y_n)^2}{\sum_n (\mu_n - \hat{y})^2}

        where :math:`\mu_n, y_n` are the predicted mean and true values,
        respectively, of the :math:`n`th sample. :math:`\hat{y}` is the average
        of the true values.

        Arguments
        ---------
        X: 2-dimensional array
            Feature array.
        y: 1-dimensional array
            Target array.

        Returns
        -------
        R2: float
            The R2 score.
        """
        return r2_score(self.predict(X), y)

    def score_reduced_chi2(self, X, y):
        r"""
        Calculate the reduced :math:`\chi^2` score defined as

        .. math::
            \chi^2 = \frac{1}{N - 2} \sum_{n} \frac{(\mu_n - y_n)^2}{\sigma_n^2}

        where :math:`\mu_n, \sigma_n, y_n` are the :math:`n`th predicted mean
        value, predicted uncertainty and true value, respectively. Lastly,
        :math:`N` is the number of samples.

        Values of :math:`\chi^2 \gg 1` indicates that the error variance is
        underestimated and :math:`\chi^2 < 1` indicates the error variance is
        overestimated.

        Arguments
        ---------
        X: 2-dimensional array
            Feature array.
        y: 1-dimensional array
            Target array.

        Returns
        -------
        chi2 : float
            The reduced :math:`\chi^2` value.
        """
        stats = self.predict_stats(X)
        if y.ndim > 1 and y.shape[1] > 1:
            raise TypeError("`y` must be a 1D array.")
        else:
            y = y.reshape(-1,)

        return numpy.sum((stats[:, 0] - y)**2 / stats[:, 1]**2) / (y.size- 2)

    def predict_gradient(self, X):
        """
        Predict the gradient of the predictions with respect to the input
        features.

        Arguments
        ---------
        X: 2-dimensional array
            Feature array.

        Returns
        -------
        grad: 3-dimensional array
            Array of gradients of shape (2, `Nsamples`, `Nfeatures`). The first
            axis correspond to the gradient of the mean and standard deviation,
            respectively.
        """
        X = tf.convert_to_tensor(X)
        x_input = {"linear_input": X, "deep_input": X}
        # We will need separate tapes for mu and std
        with tf.GradientTape() as t_mu:
            t_mu.watch(x_input)
            mu_pred = self.model(x_input).mean()

        with tf.GradientTape() as t_std:
            t_std.watch(x_input)
            std_pred = self.model(x_input).stddev()

        # The linear and deep input gradients are the same so might as well
        # take this.
        mu_grad = t_mu.gradient(mu_pred, x_input)["linear_input"]
        std_grad = t_std.gradient(std_pred, x_input)["linear_input"]

        return numpy.stack([numpy.asarray(mu_grad), numpy.asarray(std_grad)])

    @classmethod
    def from_checkpoint(cls, checkpoint_dir, optimizer):
        """
        Initialise from a checkpoint.

        Arguments
        ---------
        checkpoint_dir: str
            Path to the directory with the checkpoint files `params.p` and
            `cp.ckpt`.
        optimizer: keras optimizer
            Optimizer to train the network.

        Returns
        -------
        network: :py:class:`GaussianLossNN`
            The initialised model with loaded weights.
        """
        params = joblib.load(os.path.join(checkpoint_dir, "params.p"))
        network = cls(**params, checkpoint_dir=checkpoint_dir)
        network.adv_model.compile(optimizer=optimizer,
                                  loss=network._hamiltonian_loss)
        checkpoint_path = os.path.join(checkpoint_dir, "cp.ckpt")
        network.adv_model.load_weights(checkpoint_path)
        return network

    @classmethod
    def fit_directly(cls, Xtrain, ytrain, batch_size, checkpoint_dir,
                     model_kwargs={}, optimizer="adamax", patience=50,
                     epochs=500, validation_size=0.2):
        """
        Initialise the model and directly fit it.

        TODO: add docs
        """
        # Deepcopy the kwargs
        model_kwargs = deepcopy(model_kwargs)

        # Do some input checking
        if model_kwargs.pop("Ninputs", None) is not None:
            warn("`Ninputs` inferred implicitly from `Xtrain`. Ignoring the value "
                 "in `model_kwargs`.")
        if model_kwargs.pop("checkpoint_dir", None) is not None:
            warn("`checkpoint_dir` must be specified outside `model_kwargs`. "
                 "Ignoring the value in `model_kwargs`.")

        # Initiliase the model
        Ninputs = Xtrain.shape[1]
        network = cls(Ninputs, checkpoint_dir, **model_kwargs)
        # Fit it
        network.fit(Xtrain, ytrain, batch_size, optimizer, patience, epochs,
                  validation_size)
        return network


class EnsembleGaussianLossNN:
    """
    Make clear that this class is not for training (since we do that with MPI)
    """

    def __init__(self, checkpoint_dirs, optimizer=None):
        if not isinstance(checkpoint_dirs, list):
            raise TypeError("`checkpoint_dirs` must be a list.")

        if optimizer is None:
            optimizer = "hmmm"

        self.models = [GaussianLossNN.from_checkpoint(path, optimizer)
                       for path in checkpoint_dirs]

    def predict_stats(self, X, y=None, dscore=0.1, median_tol=0.01):
        """
        Predict the mean and standard deviation for features `X`.

        Arguments
        ---------
        X : n-dimensional array
            Feature array of shape (`Nsamples`, `Nfeatures`).
        y : 1-array, optional
            Target array corresponding to `X` of shape (`Nsamples`, ).
            Optional, if supplied used to reject models with outlier values
            of R^2 that did not converge.

        Returns
        -------
        out : dict with keys
            means : array
                Mean predictions of each sample from each model, shape
                is (`Nmodels`, `Nsamples`).
            stds : array
                Standard deviation of each sample from each model, shape
                is (`Nmodels`, `Nsamples`).
        """
        means = []
        stds = []

        for model in self.models:
            pars = model.predict_stats(X)

            means.append(pars["mean"])
            stds.append(pars["std"])

        if y is not None:
            scores = numpy.asarray([r2_score(mean, y) for mean in means])

            selected_models = numpy.ones(scores.size, dtype=bool)
            prev_median = None
            while True:
                median = numpy.median(scores[selected_models])
                lower = median - dscore

                selected_models[scores < lower] = False
                # Continue to the next iteration if first iteration
                if prev_median is None:
                    continue
                # Check whether hit the stopping condition
                if numpy.abs(median / prev_median - 1) < median_tol:
                    break
                else:
                    prev_median = median
            # Select the means and stds from models that survived
            means = [mean for i, mean in enumerate(means) if selected_models[i]]
            stds = [std for i, std in enumerate(stds) if selected_models[i]]
            scores = scores[selected_models]
        else:
            scores = numpy.nan


        means = numpy.vstack(means)
        stds = numpy.vstack(means)

        return {"means": means,
                "stds": stds,
                "scores": scores}

    def stacked_ensemble_stats(self, stats=None, X=None, y=None):
        are_both_none = stats is None and X is None
        are_both_given = stats is not None and X is not None
        if are_both_none or are_both_given:
            raise ValueError("Must supply either `stats` of `X`")

        if stats is None:
            stats = self.predict_stats(X)
        # The ensemble mean is the mean of the models
        mean = numpy.mean(stats["means"], axis=0)
        # The ensemble std is the sqrt of the averaged variance with correction
        std = numpy.mean(stats["stds"]**2
                         + (stats["means"] - mean)**2, axis=0)**0.5

        mean_deviation = numpy.std(stats["means"], axis=0)
        std_deviation = numpy.std(stats["stds"], axis=0)

        if y is not None:
            in1sigma = numpy.sum(numpy.abs(mean - y) < std) / mean.size
        else:
            in1sigma = numpy.nan

        return {"mean": mean,
                "std": std,
                "mean_deviation": mean_deviation,
                "std_deviation": std_deviation,
                "in1sigma": in1sigma}
