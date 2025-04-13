import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import hilbert
import os

# Simple class to store solution data
class Solution:
    def __init__(self, t, y):
        self.t = t
        self.y = y

# Define the parameters from equation (33) - adjusted for numerical stability
gStar = 106.75  # effective degrees of freedom (standard model value)
tIn = 1e-3      # initial temperature (GeV)
mP = 1.22e19    # Planck mass (GeV)
lambdaVal = 0.1  # λ in the potential
phiIn = 1e-5    # initial value for ϕ_in (GeV)
beta = 0.01     # β parameter
g = 1.0         # coupling constant g
muNu = 0.1      # neutrino mass scale (eV)

# Convert muNu to GeV for consistency
muNu = muNu * 1e-9  # convert eV to GeV

# Rescale parameters to avoid numerical overflow
scalingFactor = 1e10
bScaled = 4 * lambdaVal * (phiIn/scalingFactor)**2 / (tIn**2)
cScaled = beta * g / scalingFactor
dScaled = beta * (phiIn/scalingFactor)**2 / (muNu**2)

# The small parameter multiplying the second derivative in Eq(33)
epsilon = (8 * np.pi**3 * gStar / 90) * (tIn / mP)**2

def odeSystem(tTilde, y):
    """
    ODE system corresponding to equation (33) in dimensionless variables,
    with scaling to avoid numerical issues.
    
    Parameters:
        tTilde : float
            Dimensionless temperature variable.
        y : array_like
            y[0] = ϕ̃, y[1] = dϕ̃/dT̃.
    
    Returns:
        dydT : list
            [dϕ̃/dT̃, d²ϕ̃/dT̃²]
    """
    phiTilde, dphiDt = y
    
    # To avoid division by zero, ensure tTilde is not too small
    tTilde = max(tTilde, 1e-8)
    
    # Added clipping to avoid numerical overflow
    phiClipped = np.clip(phiTilde, -1e8, 1e9)
    
    # Right-hand side of the second equation as per equation (33)
    # First term: -4λφ²ᵢₙφ³/T²ᵢₙ
    numerator = -bScaled * phiClipped**3
    
    # Second term: -(βgT̃²φ/12)(βφ²ᵢₙφ²/μ²ν - 1)
    # Only compute if phi isn't too large to avoid numerical instability
    if abs(phiClipped) < 1e6:
        try:
            secondTerm = cScaled * tTilde**2 * phiClipped * (dScaled * phiClipped**2 - 1)
            numerator -= secondTerm
        except (OverflowError, FloatingPointError):
            # Handle potential numerical errors
            pass
        
    # Handle potential division by zero or very small values
    try:
        d2phiDt2 = numerator / (epsilon * tTilde**6)
        
        # Add a safeguard against extreme values that might cause instability
        d2phiDt2 = np.clip(d2phiDt2, -1e15, 1e15)
    except (OverflowError, FloatingPointError, ZeroDivisionError):
        # If an error occurs, use a fallback value
        d2phiDt2 = 0.0
    
    return [dphiDt, d2phiDt2]

def extractEnvelope(t, y):
    """
    Extract the amplitude envelope of an oscillatory signal using Hilbert transform.
    
    Parameters:
        t : array_like
            Time points
        y : array_like
            Signal values
    
    Returns:
        t : array_like
            Original time points
        envelope : array_like
            Amplitude envelope
    """
    # Handle NaN values by replacing them with zeros
    yClean = np.nan_to_num(y)
    
    # Compute the analytic signal (using the Hilbert transform)
    analyticSignal = hilbert(yClean)
    
    # Get the amplitude envelope
    amplitudeEnvelope = np.abs(analyticSignal)
    
    return t, amplitudeEnvelope

def solveNumerically(initialConditions, tInitial=1.0, tFinal=1e-8, numPoints=5000):
    """
    Solve the full system with different initial conditions and extract both the
    oscillatory solution and its envelope.
    
    Parameters:
        initialConditions : list of list
            List of initial conditions [phi(1), dphi/dT(1)]
        tInitial : float
            Initial temperature (normalized)
        tFinal : float
            Final temperature (normalized)
        numPoints : int
            Number of evaluation points
            
    Returns:
        solutions : list of Solution
            Numerical solutions for each initial condition
        envelopes : list of tuples
            Amplitude envelopes for each solution
    """
    # Start slightly away from initial point to reduce stiffness
    tStart = tInitial * 0.999
    
    # Use logarithmic spacing with more points to better capture behavior at small T
    tEval = np.logspace(np.log10(tFinal), np.log10(tStart), numPoints)
    
    solutions = []
    envelopes = []
    
    for y0 in initialConditions:
        # Define a wrapper function with better safeguards
        def safeOdeSystem(t, y):
            # Add additional safeguards for extreme t values
            if t < 1e-8:
                t = 1e-8
            return odeSystem(t, y)
        
        # Solve using LSODA with adjusted parameters for better handling of stiffness
        sol = solve_ivp(
            safeOdeSystem, 
            (tFinal, tStart), 
            y0, 
            t_eval=tEval, 
            method='LSODA', 
            rtol=1e-8, 
            atol=1e-10,
            max_step=1e-2
        )
        
        # Convert to our Solution class format
        solution = Solution(sol.t, sol.y)
        solutions.append(solution)
        
        # Extract envelope
        envT, envY = extractEnvelope(solution.t, solution.y[0])
        envelopes.append((envT, envY))
    
    return solutions, envelopes

def getAnalyticalSolution(tValues, amplitude, epsilonVal=epsilon, lambdaValue=lambdaVal, phaseShift=0.0):
    """
    Generate the analytical solution for the simplified equation.
    
    The analytical solution has the form: φ = AT*cos(σ/T + γ)
    where σ = 2.0 * √(λ/ε) and A is determined by initial conditions.
    
    Parameters:
        tValues : array_like
            Temperature values (normalized)
        epsilonVal : float
            The small parameter value
        lambdaValue : float
            The lambda parameter in the potential
        amplitude : float
            The amplitude parameter A
        phaseShift : float
            The phase shift γ
            
    Returns:
        phiValues : array_like
            Field values at given temperatures
        envelope : array_like
            Amplitude envelope values
    """
    
    # Calculate σ parameter which controls oscillation frequency
    sigma = 2.0 * np.sqrt(lambdaValue / epsilonVal)
    
    # Calculate envelope: A*T
    envelope = amplitude * tValues
    
    # The analytical solution with oscillations: A*T*cos(σ/T + γ)
    phiValues = envelope * np.cos(sigma / tValues + phaseShift)
    
    return phiValues, envelope

def plotNumericalSolutions(solutions, envelopes, initialConditions, savePath='images/fullSolutionsAndEnvelopes.png'):
    """
    Plot numerical solutions and their envelopes.
    
    Parameters:
        solutions : list of Solution
            Numerical solutions
        envelopes : list of tuples
            Amplitude envelopes
        initialConditions : list of list
            Initial conditions used for solutions
        savePath : str
            Path to save the figure
    """
    plt.figure(figsize=(12, 8))
    
    # First plot: Full oscillatory solutions
    plt.subplot(2, 1, 1)
    for i, sol in enumerate(solutions):
        plt.plot(sol.t, sol.y[0], label=f'ϕ̃(1) = {initialConditions[i][0]}')
    
    plt.xlabel('T̃ (T/T_in)')
    plt.ylabel('ϕ̃')
    plt.title('Numerical Solutions of Equation 33')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    
    # Second plot: Amplitude envelopes
    plt.subplot(2, 1, 2)
    for i, (t, env) in enumerate(envelopes):
        plt.plot(t, env, label=f'Envelope for ϕ̃(1) = {initialConditions[i][0]}')
    
    plt.xlabel('T̃ (T/T_in)')
    plt.ylabel('Amplitude Envelope')
    plt.title('Amplitude Envelopes (Showing Damping Behavior)')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(savePath), exist_ok=True)
    plt.savefig(savePath)
    plt.show()

def compareWithAnalytical(phiInitial=0.5, tInitial=1.0, tFinal=1e-7,numPoints=5000, savePath='images/analyticalComparison.png'):
    """
    Compare the numerical solution with the analytical solution from the simplified model.
    
    Parameters:
        phiInitial : float
            Initial value of phi
        tInitial : float
            Initial temperature (normalized)
        tFinal : float
            Final temperature (normalized)
        numPoints : int
            Number of evaluation points
        savePath : str
            Path to save the figure
            
    Returns:
        numericalSolution : Solution
            Numerical solution object
        analyticalData : tuple
            Tuple containing (t_values, phi_values, envelope_values)
    """
    # Create a single initial condition as a list for solveNumerically
    initialConditions = [[phiInitial, 0.0]]  # Initial velocity = 0
    
    # Use the existing solveNumerically function for consistency
    solutions, envelopes = solveNumerically(
        initialConditions,
        tInitial=tInitial,
        tFinal=tFinal,
        numPoints=numPoints
    )
    
    # Extract the (only) solution and its envelope
    numericalSolution = solutions[0]
    numericalEnvelope = envelopes[0][1]  # Extract the y-values from the envelope tuple
    
    # Get t values for analytical solution matching the numerical solution
    tAnalytical = numericalSolution.t
    
    # Parameters for analytical solution
    amplitude = 1e8
    
    # Get analytical solution
    phiAnalytical, envelopeAnalytical = getAnalyticalSolution(
        tAnalytical, 
        amplitude,
        epsilonVal=epsilon
    )
    
    # Plot the results
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Compare full numerical solution with analytical
    plt.subplot(2, 1, 1)
    plt.plot(numericalSolution.t, numericalSolution.y[0], 'b-', alpha=0.7, linewidth=1,
             label='Full Numerical Solution')
    plt.plot(tAnalytical, phiAnalytical, 'r--', alpha=0.45, linewidth=1,
             label=f'Analytical: ϕ=AT̃cos(σ/T̃)')
    
    plt.xlabel('T̃ (T/T_in)')
    plt.ylabel('ϕ̃')
    plt.title('Comparison of Numerical and Analytical Solutions')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    
    # Plot 2: Compare envelopes
    plt.subplot(2, 1, 2)
    plt.plot(numericalSolution.t, numericalEnvelope, 'b-', label='Numerical Envelope')
    plt.plot(numericalSolution.t, -numericalEnvelope, 'b-') # Negative envelope
    plt.plot(tAnalytical, envelopeAnalytical, 'r--', label=f'Analytical Envelope: AT̃')
    plt.plot(tAnalytical, -envelopeAnalytical, 'r--')  # Negative envelope
    
    plt.xlabel('T̃ (T/T_in)')
    plt.ylabel('Envelope Amplitude')
    plt.title('Comparison of Numerical and Analytical Envelopes')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(savePath), exist_ok=True)
    plt.savefig(savePath)
    plt.show()
    
    # Also create a zoomed view to better see the oscillations
    createZoomedView(numericalSolution, tAnalytical, phiAnalytical, envelopeAnalytical,
                     savePath='images/analyticalComparison_zoomed.png')
    
    return numericalSolution, (tAnalytical, phiAnalytical, envelopeAnalytical)

def createZoomedView(numericalSolution, tAnalytical, phiAnalytical, envelopeAnalytical,
                     tZoomMin=0.05, tZoomMax=0.2, savePath='images/zoomed_comparison.png'):
    """
    Create a zoomed view of the solution to better visualize oscillations.
    
    Parameters:
        numericalSolution : Solution
            Numerical solution object
        tAnalytical : array_like
            Temperature values for analytical solution
        phiAnalytical : array_like
            Field values for analytical solution
        envelopeAnalytical : array_like
            Envelope values for analytical solution
        tZoomMin : float
            Minimum temperature for zoom
        tZoomMax : float
            Maximum temperature for zoom
        savePath : str
            Path to save the figure
    """
    plt.figure(figsize=(12, 6))
    
    # Filter data for zoom region
    maskNumerical = (numericalSolution.t >= tZoomMin) & (numericalSolution.t <= tZoomMax)
    maskAnalytical = (tAnalytical >= tZoomMin) & (tAnalytical <= tZoomMax)
    
    # Plot solutions
    plt.plot(numericalSolution.t[maskNumerical], numericalSolution.y[0][maskNumerical], 'b-', label='Full Numerical Solution')
    plt.plot(tAnalytical[maskAnalytical], phiAnalytical[maskAnalytical], 'r--',
             alpha=0.3, label='Analytical Solution')
    
    # Extract envelope for numerical solution in zoom region
    tEnv, env = extractEnvelope(
        numericalSolution.t[maskNumerical], 
        numericalSolution.y[0][maskNumerical]
    )
    
    # Plot envelopes
    plt.plot(tEnv, env, 'g-', alpha=0.5, label='Numerical Envelope')
    plt.plot(tEnv, -env, 'g-', alpha=0.5) # Negative envelope
    plt.plot(tAnalytical[maskAnalytical], envelopeAnalytical[maskAnalytical], 'm--', 
             alpha=0.5, label='Analytical Envelope')
    plt.plot(tAnalytical[maskAnalytical], -envelopeAnalytical[maskAnalytical], 'm--', alpha=0.5) # Negative envelope
    
    plt.xlabel('T̃ (T/T_in)')
    plt.ylabel('ϕ̃')
    plt.title(f'Detailed View of Oscillations (T̃ = {tZoomMin} to {tZoomMax})')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(savePath), exist_ok=True)
    plt.savefig(savePath)
    plt.show()

def analyzeParameterEffect(epsilonFactors=[0.5, 1.0, 2.0], phiInitial=0.5,
                           tInitial=1.0, tFinal=1e-8, numPoints=5000,
                           savePath='images/parameterEffectAnalysis.png'):
    """
    Analyze how changing the small parameter epsilon affects oscillation frequency
    but not the amplitude envelope.
    
    Parameters:
        epsilonFactors : list of float
            Factors to multiply epsilon by
        phiInitial : float
            Initial value of phi
        tInitial : float
            Initial temperature (normalized)
        tFinal : float
            Final temperature (normalized)
        numPoints : int
            Number of evaluation points
        savePath : str
            Path to save the figure
            
    Returns:
        solutions : list of Solution
            Solutions for different epsilon values
        envelopes : list of tuples
            Envelopes for different epsilon values
    """
    # Original epsilon value
    epsilonOriginal = epsilon
    
    # Start slightly away from initial point to reduce stiffness
    tStart = tInitial * 0.999
    
    # Generate t_eval points
    tEval = np.logspace(np.log10(tFinal), np.log10(tStart), numPoints)
    
    # Initial condition
    y0 = [phiInitial, 0.0]  # Standard initial condition
    
    solutions = []
    envelopes = []
    epsilonValues = []
    
    for factor in epsilonFactors:
        # Calculate modified epsilon
        epsilonModified = epsilonOriginal * factor
        epsilonValues.append(epsilonModified)
        
        # Define ODE system with modified epsilon
        def modifiedOdeSystem(t, y):
            t = max(t, 1e-8)
            
            phiTilde, dphiDt = y
            phiClipped = np.clip(phiTilde, -1e8, 1e9)
            
            numerator = -bScaled * phiClipped**3
            
            if abs(phiClipped) < 1e6:
                secondTerm = cScaled * t**2 * phiClipped * (dScaled * phiClipped**2 - 1)
                numerator -= secondTerm
                
            d2phiDt2 = numerator / (epsilonModified * t**6)
            d2phiDt2 = np.clip(d2phiDt2, -1e15, 1e15)
            
            return [dphiDt, d2phiDt2]
        
        # Solve the system
        sol = solve_ivp(
            modifiedOdeSystem, 
            (tFinal, tStart), 
            y0, 
            t_eval=tEval, 
            method='LSODA', 
            rtol=1e-8, 
            atol=1e-10, 
            max_step=1e-2
        )
        solution = Solution(sol.t, sol.y)
        solutions.append(solution)
        
        # Extract envelope
        envT, envY = extractEnvelope(sol.t, sol.y[0])
        envelopes.append((envT, envY))
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    
    # Plot full solutions
    plt.subplot(2, 1, 1)
    for i, sol in enumerate(solutions):
        plt.plot(sol.t, sol.y[0], 
                 label=f'ε = {epsilonValues[i]:.2e} (x{epsilonFactors[i]})')
    
    plt.xlabel('T̃ (T/T_in)')
    plt.ylabel('ϕ̃')
    plt.title('Effect of Small Parameter ε on Oscillation Frequency')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    
    # Plot envelopes
    plt.subplot(2, 1, 2)
    for i, (t, env) in enumerate(envelopes):
        plt.plot(t, env, 
                 label=f'Envelope for ε = {epsilonValues[i]:.2e} (x{epsilonFactors[i]})')
    
    plt.xlabel('T̃ (T/T_in)')
    plt.ylabel('Amplitude Envelope')
    plt.title('Effect of Small Parameter ε on Amplitude Envelope')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(savePath), exist_ok=True)
    plt.savefig(savePath)
    plt.show()
    
    return solutions, envelopes

def compareAnalyticalVariants(savePath='images/analytical_variants.png'):
    """
    Compare different variants of the analytical solution using simplified models.
    
    Parameters:
        savePath : str
            Path to save the figure
    
    Returns:
        tValues : array_like
            Temperature values
        analyticalSolutions : list of array_like
            List of analytical solutions with different sigma values
    """
    # Setup parameters
    tInitial = 1.0
    tFinal = 0.01
    
    # Create a fine grid for solutions
    tValues = np.logspace(np.log10(tFinal), np.log10(tInitial), 10000)
    
    # Use different sigma values to show frequency effects
    sigmaValues = [1.0, 2.0, 4.0]
    amplitude = 1e8
    
    # Compute analytical solutions
    analyticalSolutions = []
    
    for sigma in sigmaValues:
        # Use modified getAnalyticalSolution for direct sigma specification
        epsilonCustom = 4 * lambdaVal / sigma**2  # Reverse-engineer epsilon from sigma
        solution, _ = getAnalyticalSolution(
            tValues, 
            amplitude,
            epsilonVal=epsilonCustom,
        )
        analyticalSolutions.append(solution)
    
    # Plot results
    plt.figure(figsize=(12, 10))
    
    # First plot: Compare solutions with different σ values
    plt.subplot(2, 1, 1)
    
    for i, solution in enumerate(analyticalSolutions):
        plt.plot(tValues, solution, label=f'σ = {sigmaValues[i]}')
    
    plt.xlabel('T̃ (T/T_in)')
    plt.ylabel('ϕ̃')
    plt.title('Analytical Solutions with Different σ Values')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    
    # Second plot: Show envelope clearly
    plt.subplot(2, 1, 2)
    
    # Choose one solution to demonstrate envelope
    i = 1  # Use middle sigma value
    plt.plot(tValues, analyticalSolutions[i], 'b-', label=f'Solution (σ = {sigmaValues[i]})')
    
    # Plot envelope: T̃|φ_in|
    envelope = tValues * amplitude
    plt.plot(tValues, envelope, 'r--', label='Envelope: T̃|φ_in|')
    plt.plot(tValues, -envelope, 'r--')
    
    plt.xlabel('T̃ (T/T_in)')
    plt.ylabel('ϕ̃')
    plt.title('Amplitude Decay in Analytical Solution')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(savePath), exist_ok=True)
    plt.savefig(savePath)
    plt.show()
    
    return tValues, analyticalSolutions

def main():
    """Main function to execute the analysis workflow."""
    print("Analyzing Equation 33 - Comparison with Analytical Solution")
    
    # Ensure we have an 'images' directory to save the plots
    if not os.path.exists('images'):
        os.makedirs('images')
        print("Created 'images' directory for saving plots")
    
    # 1. Solve with different initial conditions
    print("\n1. Solving the full system with different initial conditions...")
    initialConditions = [
        [0.1, 0.0],  # ϕ̃(1) = 0.1, dϕ̃/dT̃(1) = 0
        [0.3, 0.0],  # ϕ̃(1) = 0.3, dϕ̃/dT̃(1) = 0
        [0.5, 0.0],  # ϕ̃(1) = 0.5, dϕ̃/dT̃(1) = 0
    ]
    solutions, envelopes = solveNumerically(initialConditions)
    plotNumericalSolutions(solutions, envelopes, initialConditions)
    
    # 2. Compare with analytical solution
    print("\n2. Comparing numerical solution with analytical form...")
    numericalSolution, analyticalData = compareWithAnalytical()
    

    # 3. Compare different analytical solution variants
    print("\n3. Comparing analytical solution variants...")
    tValues, analyticalVariants = compareAnalyticalVariants()

    # # 4. Analyze parameter effects (epsilon variations)
    # print("\n4. Analyzing effect of the small parameter epsilon...")
    # paramSolutions, paramEnvelopes = analyzeParameterEffect()
    
    
    print("\nAnalysis complete. Results saved as PNG files in the 'images' directory.")

if __name__ == "__main__":
    main()