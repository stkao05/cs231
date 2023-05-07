from builtins import range
import numpy as np


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, D = X.shape
    D, C = W.shape

    for i in range(N):
        xi = X[i]  # (D)
        yi = y[i]
        z = xi @ W  # (C)
        logits = np.zeros_like(z)
        lmax = np.exp(np.max(z))

        for j in range(C):
            logits[j] = np.exp(z[j] - lmax)

        lsum = np.sum(logits)
        logits /= lsum
        prob = logits[yi]
        loss += -np.log(prob)

    loss /= N
    loss += reg * np.sum(W**2)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, _ = X.shape
    logits = X @ W  # (N, C)

    logit_max = logits.max(axis=1, keepdims=True)  # (N, 1)
    norm_logits = logits - logit_max  # (N, C)

    counts = np.exp(norm_logits)  # (N, C)
    counts_sum = counts.sum(axis=1, keepdims=True)  # (N, 1)
    counts_sum_div = counts_sum**-1  # (N, 1)
    probs = counts * counts_sum_div  # (N, C)
    logprobs = np.log(probs)
    loss = -logprobs[range(N), y].mean()

    # back propogation
    # dlogprob = -1 / N * np.ones_like(logprob)  # (N, )
    # dprob = 1 / prob * dlogprob  # (N, )

    # dlogits = np.zeros_like(logits)  # (N, C)
    # dlogits[np.arrange(N), y] = dprob

    # dzexp = np.ones_like(zexp)

    # dZ = np.copy(Z)
    # dZ[np.arange, y] =

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
