import numpy as np
from scipy.integrate import solve_ivp

def rk4Step(f, t, y, h):
    """
    Perform one step of the RK4 method.
    
    Parameters:
        f : callable
            Function that returns dy/dt
        t : float
            Current time
        y : array_like
            Current solution values
        h : float
            Step size
    
    Returns:
        array_like: Solution at t + h
    """
    k1 = f(t, y)
    k2 = f(t + 0.5*h, y + 0.5*h*np.array(k1))
    k3 = f(t + 0.5*h, y + 0.5*h*np.array(k2))
    k4 = f(t + h, y + h*np.array(k3))
    
    return y + (h/6.0)*(np.array(k1) + 2*np.array(k2) + 2*np.array(k3) + np.array(k4))

def solveWithRk4(odeSystem, tSpan, y0, tEval=None, **kwargs):
    """
    Solve ODE system using 4th order Runge-Kutta method.
    
    Parameters:
        odeSystem : callable
            Function of the form f(t, y) returning dy/dt
        tSpan : tuple
            The time interval (t0, tf)
        y0 : array_like
            Initial conditions
        tEval : array_like, optional
            Times at which to store the solution
            
    Returns:
        Object with fields:
            t : array_like
                Time points
            y : array_like
                Solution values at t
    """
    t0, tf = tSpan
    if tEval is None:
        # Default to 1000 points if not specified
        tEval = np.linspace(t0, tf, 1000)
    
    dt = tEval[1] - tEval[0]
    y = np.zeros((len(tEval), len(y0)))
    y[0] = y0
    
    # Integrate using RK4
    for i in range(1, len(tEval)):
        y[i] = rk4Step(odeSystem, tEval[i-1], y[i-1], dt)
    
    class Solution:
        pass
    
    sol = Solution()
    sol.t = tEval
    sol.y = y.T
    return sol

def solveStiffOde(odeSystem, tSpan, y0, tEval=None, rtol=1e-8, atol=1e-10, method='RK4'):
    """
    Solves an ODE system using either RK4 or other methods from scipy.
    
    Parameters:
        odeSystem : callable
            Function of the form f(t, y) returning dy/dt.
        tSpan : tuple
            The time interval (t0, tf).
        y0 : array_like
            Initial conditions.
        tEval : array_like, optional
            Times at which to store the computed solution.
        rtol : float
            Relative tolerance (used only for scipy methods).
        atol : float
            Absolute tolerance (used only for scipy methods).
        method : string
            Integration method ('RK4' or any scipy solver name).
    
    Returns:
        sol : Solution object
            The solution with fields t and y.
    """
    if method == 'RK4':
        return solveWithRk4(odeSystem, tSpan, y0, tEval)
    else:
        return solve_ivp(odeSystem, tSpan, y0, t_eval=tEval, method=method, rtol=rtol, atol=atol)

def solveBoundaryLayer(fastSystem, tSpan, y0, epsilon, **kwargs):
    """
    Solves the boundary layer equation (fast system) using appropriate scaling.
    
    Parameters:
        fastSystem : callable
            Function representing the fast dynamics.
        tSpan : tuple
            The time interval in the stretched time scale.
        y0 : array_like
            Initial conditions.
        epsilon : float
            Small parameter characterizing the singular perturbation.
        **kwargs : dict
            Additional arguments passed to solveStiffOde.
    
    Returns:
        sol : OdeResult
            Solution in the boundary layer.
    """
    # Scale time according to τ = t/ε
    scaledTSpan = (tSpan[0]/epsilon, tSpan[1]/epsilon)
    
    def scaledSystem(t, y):
        return fastSystem(epsilon * t, y, epsilon)
    
    return solveStiffOde(scaledSystem, scaledTSpan, y0, **kwargs)

def solveReducedProblem(slowSystem, tSpan, y0, **kwargs):
    """
    Solves the reduced problem (slow system) when epsilon → 0.
    
    Parameters:
        slowSystem : callable
            Function representing the slow dynamics.
        tSpan : tuple
            The time interval.
        y0 : array_like
            Initial conditions.
        **kwargs : dict
            Additional arguments passed to solveStiffOde.
    
    Returns:
        sol : OdeResult
            Solution of the reduced system.
    """
    return solveStiffOde(slowSystem, tSpan, y0, **kwargs)

def compositeSolution(outerSol, innerSol, matchingPoint, epsilon):
    """
    Constructs a composite solution by combining inner and outer solutions.
    
    Parameters:
        outerSol : OdeResult
            Solution of the outer (reduced) problem.
        innerSol : OdeResult
            Solution of the inner (boundary layer) problem.
        matchingPoint : float
            Point where the solutions are matched.
        epsilon : float
            Small parameter.
            
    Returns:
        t : array
            Time points.
        y : array
            Composite solution.
    """
    # Combine and sort time points
    tOuter = outerSol.t
    tInner = epsilon * innerSol.t  # Scale back to original time
    
    # Get solutions at matching point
    outerMatch = np.interp(matchingPoint, tOuter, outerSol.y[0])
    innerMatch = np.interp(matchingPoint/epsilon, innerSol.t, innerSol.y[0])
    
    # Combine solutions using additive composition
    def composite(t):
        outer = np.interp(t, tOuter, outerSol.y[0])
        inner = np.interp(t/epsilon, innerSol.t, innerSol.y[0])
        return outer + inner - outerMatch
    
    # Create combined time array
    tCombined = np.sort(np.unique(np.concatenate([tOuter, tInner])))
    yComposite = composite(tCombined)
    
    return tCombined, yComposite