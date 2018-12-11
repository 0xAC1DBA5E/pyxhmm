#cython: boundscheck=False, wraparound=False, cdivision=True

cimport cython
from libc.math cimport log
cimport numpy as np
import numpy as np


# pi      initial state distribution
# tau     transition matrix
# epsilon emissions matrix
# iota    observations
# delta   probability of observations for each state
# alpha   forward probabilities
# beta    backward probabilities
# gamma   state probabilities
# ksi     transition probabilities
# zeta    filtered forward probabilities

cpdef np.float64_t shannon_entropy(np.float64_t[::1] prob) nogil:
    cdef np.int64_t i, I = len(prob)
    cdef np.float64_t entropy = 0.0
    for i in xrange(I):
        if prob[i] > 0.0:
            entropy -= prob[i] * log(prob[i])
    return entropy

cpdef np.float64_t logprob(np.float64_t[::1] prob) nogil:
    cdef np.int64_t t, T = len(prob)
    cdef np.float64_t out = 0.0
    for t in xrange(T):
        out += log(prob[t])
    return out

cdef void calc_delta(np.float64_t[:,::1] epsilon, np.int64_t[::1] iota, np.float64_t[:,::1] delta,
                     np.int64_t S, np.int64_t T) nogil:
    cdef np.int64_t t, i
    for t in xrange(T):
        for i in xrange(S):
            delta[t,i] = epsilon[i, iota[t]]

cdef void calc_alpha_beta(np.float64_t[::1] pi, np.float64_t[:,::1] tau, np.float64_t[:,::1] delta, 
                          np.float64_t[:,::1] alpha, np.float64_t[:,::1] beta, np.float64_t[::1] norm,
                          np.int64_t S, np.int64_t T) nogil:
    cdef np.int64_t t, i, j
    # calc alpha
    norm[0] = 0.0
    for i in xrange(S):
        alpha[0,i] = pi[i]*delta[0,i]
        norm[0] += alpha[0,i]
        beta[T-1,i] = 1.0
    for i in xrange(S):
        alpha[0,i] /= norm[0]
    for t in xrange(1, T):
        norm[t] = 0.0
        for i in xrange(S):
            alpha[t,i] = 0.0
            for j in xrange(S):
                alpha[t,i] += alpha[t-1,j]*tau[j,i]
            alpha[t,i] *= delta[t,i]
            norm[t] += alpha[t,i]
        for i in xrange(S):
            alpha[t,i] /= norm[t]
    # calc beta
    for t in reversed(xrange(T-1)):
        for i in xrange(S):
            beta[t,i] = 0.0
            for j in xrange(S):
                beta[t,i] += beta[t+1,j]*tau[i,j]*delta[t+1,j]
            beta[t,i] /= norm[t+1]
            
cdef void calc_gamma_ksi(np.float64_t[:,::1] tau, np.float64_t[:,::1] delta, np.float64_t[:,::1] alpha, np.float64_t[:,::1] beta,
                         np.float64_t[::1] norm, np.float64_t[:,::1] gamma, np.float64_t[:,:,::1] ksi,
                         np.int64_t S, np.int64_t T) nogil:
    cdef np.int64_t t, i, j
    cdef np.float64_t sigma_gamma, sigma_ksi
    for t in xrange(T):
        sigma_gamma = sigma_ksi = 0.0
        for i in xrange(S):
            gamma[t,i] = alpha[t,i]*beta[t,i]
            sigma_gamma += gamma[t,i]
            if t < T-1:
                for j in xrange(S):
                    ksi[t,i,j] = alpha[t,i]*tau[i,j]*beta[t+1,j]*delta[t+1,j]/norm[t+1]
                    sigma_ksi += ksi[t,i,j]
        for i in xrange(S):
            if sigma_gamma > 0:
                gamma[t,i] /= sigma_gamma
            for j in xrange(S):
                if t < T-1 and sigma_ksi > 0:
                    ksi[t,i,j] /= sigma_ksi
                else:
                    ksi[t,i,j] = 0.0

cdef void calc_tau_epsilon(np.float64_t[:,::1] gamma, np.float64_t[:,:,::1] ksi, np.int64_t[::1] iota,
                      np.float64_t[:,::1] tau, np.float64_t[:,::1] epsilon, np.float64_t[::1] sigma_gamma,
                      np.int64_t U, np.int64_t S, np.int64_t T) nogil:
    cdef np.int64_t t, i, j
    # reset tau
    for i in xrange(S):
        sigma_gamma[i] = 0.0
        for j in xrange(S):
            tau[i,j] = 0.0
        for j in xrange(U):
            epsilon[i, j] = 0.0
    # update epsilon
    for t in xrange(T-1):
        for i in xrange(S):
            epsilon[i, iota[t]] += gamma[t,i]
            sigma_gamma[i] += gamma[t,i]
            for j in xrange(S):
                tau[i,j] += ksi[t,i,j]
    # normalize
    for i in xrange(S):
        for j in xrange(S):
            tau[i,j] /= sigma_gamma[i]
    # update last value
    t += 1
    for i in xrange(S):
        epsilon[i, iota[t]] += gamma[t,i]
        sigma_gamma[i] += gamma[t,i]
    # normalize
    for i in xrange(S):
        for j in xrange(U):
            epsilon[i,j] /= sigma_gamma[i]

cdef void calc_zeta(np.float64_t[::1] pi, np.float64_t[:,::1] tau, np.float64_t[:,::1] alpha,
               np.float64_t[:,::1] zeta, np.int64_t S, np.int64_t T) nogil:
    cdef np.int64_t t, i, j
    for i in xrange(S):
        zeta[0,i] = pi[i]
    for t in xrange(1, T):
        for i in xrange(S):
            zeta[t,i] = 0.0
            for j in xrange(S):
                zeta[t,i] += alpha[t-1,j]*tau[j,i]
            
cpdef baum_welch(np.float64_t[::1] pi, np.float64_t[:,::1] tau, np.float64_t[:,::1] epsilon, np.int64_t[::1] iota, np.int64_t n_iter):
    cdef np.int64_t i, j, S = len(tau), T = len(iota), U = np.max(iota)+1
    # init values
    cdef np.float64_t[:,::1] alpha = np.empty(shape=(T, S), dtype=np.float64)
    cdef np.float64_t[:,::1] beta = np.empty(shape=(T, S), dtype=np.float64)
    cdef np.float64_t[:,::1] gamma = np.empty(shape=(T, S), dtype=np.float64)
    cdef np.float64_t[:,::1] delta = np.empty(shape=(T, S), dtype=np.float64)
    cdef np.float64_t[:,:,::1] ksi = np.empty(shape=(T, S, S), dtype=np.float64)
    cdef np.float64_t[:,::1] zeta = np.empty(shape=(T, S), dtype=np.float64)
    # temporaries
    cdef np.float64_t[::1] norm = np.empty(shape=T, dtype=np.float64)
    cdef np.float64_t[::1] sigma = np.empty(shape=S, dtype=np.float64)
    # calculate
    with nogil:
        for i in xrange(n_iter):
            calc_delta(epsilon, iota, delta, S, T)
            calc_alpha_beta(pi, tau,  delta, alpha, beta, norm, S, T)
            calc_gamma_ksi(tau, delta, alpha, beta, norm, gamma, ksi, S, T)
            for j in xrange(S):
                pi[j] = gamma[0,j]
            # update tau, epsilon
            calc_tau_epsilon(gamma, ksi, iota, tau, epsilon, sigma, U, S, T)
        calc_zeta(pi, tau, alpha, zeta, S, T)
    return np.asarray(alpha), np.asarray(beta), np.asarray(gamma), np.asarray(delta), np.asarray(ksi), np.asarray(zeta)
