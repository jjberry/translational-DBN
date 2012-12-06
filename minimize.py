import numpy as np

def minimize(fun, X, args, length):
    '''
    Implements Conjugate Gradient Optimization, which minimizes a continuous
    differentiable multivariate function. Starting point is given by "x", 
    and the function named in "fun" must return a function value and a vector 
    partial derivatives. The Polack-Ribiere flavor of CG is used to compute
    search directions, and a line search using quadratic and cubic polynomial
    approximations and the Wolfe-Powell stopping criteria are used together with
    the slope ratio method ofr guessing initial step sizes.

    args:
        callable fun:   returns the value of the cost function and the partial
                        derivatives
        array X:        the starting point
        args:           a tuple containing the additional arguments to pass to
                        the function
        length:         the number of line searches to perform

    Ported to python from Carl Edward Rasmussen's minimize.m for matlab
    '''
    # constants for line searches. RHO and SIG are the constants in the 
    # Wolfe-Powell conditions
    RHO = 0.01
    SIG = 0.5
    INT = 0.1  #don't reevaluate within 0.1 of the limit of the current bracket
    EXT = 3.0  #extrapolate maximum 3 times the current bracket
    MAX = 20   #max 20 function evaluations per line search
    RATIO = 100 #max allowed slope ratio

    #if len(length) == 2:
    #    red = length[1]
    #    length = length[0]
    #else:
    #    red = 1.0
    red = 1.0
    if length>0:
        S = ['Linesearch']
    else:
        S = ['Function evaluation']

    i = 0 # the run length counter
    ls_failed = 0 # no previous line search has failed
    fX = []
    f1, df1 = fun(X, *args)
    if (length<0):
        i += 1
    s = -df1              #search direction is steepest
    d1 = np.dot(-s.T, s)  #this is the slope
    z1 = red/(1.0-d1)     #initial step

    while i < np.abs(length):
        if (length>0):
            i += 1
        X0 = X.copy()     #make backup copy of current values
        f0 = f1.copy()
        df0 = df1.copy()
        X = X + z1 * s
        f2, df2 = fun(X, *args)
        if (length<0):
            i += 1
        d2 = np.dot(df2.T, s)
        f3, d3, z3 = (f1, d1, -z1)  #initialize point 3 equal to point 1 
        if length>0:
            M = MAX
        else:
            M = min(MAX, -length-i)
        success = 0    #initialize variables
        limit = -1
        while True:
            while ((f2 > f1+z1*RHO*d1) or (d2 > -SIG*d1)) and (M > 0):
                limit = z1  #tighten the bracket
                if f2 > f1:
                    z2 = z3 - (0.5*d3*z3*z3)/(d3*z3+f2-f3) #quadratic fit
                else:
                    A = 6.0*(f2-f3)/z3+3.0*(d2+d3)    #cubic fit
                    B = 3.0*(f3-f2)-z3*(d3+2.0*d2)
                    z2 = (np.sqrt(B*B-A*d2*z3*z3)-B)/A
                if np.isnan(z2) or np.isinf(z2):
                    z2 = z3/2.0 #if there was a problem, bisect
                z2 = max(min(z2, INT*z3),(1-INT)*z3) #don't accept too close limits
                z1 += z2  #update the step
                X += z2*s
                f2, df2 = fun(X, *args)
                M -= 1
                if (length<0):
                    i += 1
                d2 = np.dot(df2.T, s)
                z3 = z3-z2  #z3 is now relative to the locatoin of z2
            if (f2 > f1+z1*RHO*d1) or (d2 > -SIG*d1):
                break   #this is a failure
            elif (d2 > SIG*d1):
                success = 1
                #print "success"
                break   #success
            elif (M == 0):
                break   #failure
            A = 6.0*(f2-f3)/z3+3.0*(d2+d3) #make cubic extrapolation
            B = 3.0*(f3-f2)-z3*(d3+2.0*d2)
            z2 = -d2*z3*z3/(B+np.sqrt(B*B-A*d2*z3*z3))
            if (not np.isreal(z2)) or np.isnan(z2) or np.isinf(z2) or (z2<0):
                if limit < -0.5:
                    z2 = z1 * (EXT-1.0)
                else:
                    z2 = (limit-z1)/2.0
            elif (limit > -0.5) and (z2+z1 > limit):
                z2 = (limit-z1)/2.0
            elif (limit < -0.5) and (z2+z1 > z1*EXT):
                z2 = z1*(EXT-1.0)
            elif z2 < -z3*INT:
                z2 = -z3*INT
            elif (limit > -0.5) and (z2 < (limit-z1)*(1.0-INT)):
                z2 = (limit-z1)*(1.0-INT)
            f3, d3, z3 = (f2, d3, -z2)
            z1 += z2
            X += z2*s
            f2, df2 = fun(X, *args)
            M -= 1
            if (length<0):
                i += 1
            d2 = np.dot(df2.T, s)

        if success == 1:
            f1 = f2
            fX.append(f1)
            #Polack-Ribiere direction
            s = (np.dot(df2.T,df2)-np.dot(df1.T,df2)) / \
                    np.dot(np.dot(df1.T,df1),s) - df2
            tmp = df1
            df1 = df2
            df2 = tmp #swap derivatives
            if d2 > 0:
                s = -df1
                d2 = np.dot(-s.T,s)
            z1 = z1 * min(RATIO, d1/(d2-2.2251e-308))
            d1 = d2
            ls_failed = 0
        else:
            X = X0      # restore to point before this line search
            f1 = f0 
            df1 = df0
            if (ls_failed == 1) or (i > np.abs(length)):
                break   #giving up
            tmp = df1
            df1 = df2
            df2 = tmp
            s = -df1
            z1 = 1.0/(1.0-d1)
            ls_failed = 1

    return X, fX, i

