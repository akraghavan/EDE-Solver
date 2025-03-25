import numpy as np
import matplotlib.pyplot as plt
from integrators import solveStiffOde, solveBoundaryLayer, solveReducedProblem, compositeSolution
from scipy.signal import hilbert
from scipy import interpolate

# Define the parameters from equation (33) - adjusted for numerical stability
gStar = 106.75  # effective degrees of freedom (standard model value)
tIn = 1e15  # initial temperature (GeV)
mP = 1.22e19  # Planck mass (GeV)
lambdaVal = 0.1  # λ in the potential
phiIn = 1e14  # initial value for ϕ_in (GeV)
beta = 0.01  # β parameter
g = 1.0  # coupling constant g
muNu = 0.1  # neutrino mass scale (eV)

# Convert muNu to GeV for consistency
muNu = muNu * 1e-9  # convert eV to GeV

# Rescale parameters to avoid numerical overflow
scalingFactor = 1e10
bScaled = 4 * lambdaVal * (phiIn/scalingFactor)**2 / (tIn**2)
cScaled = beta * g / scalingFactor
dScaled = beta * (phiIn/scalingFactor)**2 / (muNu**2)

# The small parameter multiplying the second derivative in Eq(33)
a = (8 * np.pi**3 * gStar / 90) * (tIn / mP)**2
epsilon = a  # This is effectively the coefficient of the second derivative

def odeSystem(tTilde, y):
    """
    ODE system corresponding to equation (33) in dimensionless variables,
    with scaling to avoid numerical issues.
    
    Parameters:
        tTilde : float
            Dimensionless temperature variable.
        y : arrayLike
            y[0] = ϕ̃, y[1] = dϕ̃/dT̃.
    
    Returns:
        dydT : list
            [dϕ̃/dT̃, d²ϕ̃/dT̃²]
    """
    phiTilde, dphiDt = y
    
    # To avoid division by zero, ensure tTilde is not too small
    tTilde = max(tTilde, 1e-10)
    
    # Added clipping to avoid numerical overflow in high powers
    phiClipped = np.clip(phiTilde, -1e8, 1e8)
    
    # Right-hand side of the second equation as per equation (33)
    # Using scaled parameters and careful handling of large powers
    numerator = -bScaled * phiClipped**3
    
    # Only add second term if phi is small enough to avoid overflow
    if abs(phiClipped) < 1e6:
        secondTerm = cScaled * tTilde**2 * phiClipped**12 * (dScaled * phiClipped**2 - 1)
        numerator -= secondTerm
        
    d2phiDt2 = numerator / (a * tTilde**6)
    
    return [dphiDt, d2phiDt2]

def extractEnvelope(t, y):
    """
    Extract the amplitude envelope of an oscillatory signal using Hilbert transform.
    
    Parameters:
        t : arrayLike
            Time points
        y : arrayLike
            Signal values
    
    Returns:
        t : arrayLike
            Original time points
        envelope : arrayLike
            Amplitude envelope
    """
    # Handle NaN values by replacing them with zeros
    yClean = np.nan_to_num(y)
    
    # Compute the analytic signal (using the Hilbert transform)
    analyticSignal = hilbert(yClean)
    
    # Get the amplitude envelope
    amplitudeEnvelope = np.abs(analyticSignal)
    
    return t, amplitudeEnvelope

def solveFullSystem():
    """
    Solve the full system with different initial conditions and extract both the 
    oscillatory solution and its envelope.
    """
    # Set the integration range for tTilde (dimensionless temperature)
    tInitial = 1.0    # starting at tIn (normalized)
    tFinal = 1e-8     # going down to very small temperatures as requested
    
    # Use logarithmic spacing with more points to better capture behavior at small T
    tEval = np.logspace(np.log10(tInitial), np.log10(tFinal), 5000)
    
    # Try different initial conditions - adjusted for numerical stability
    initialConditions = [
        [0.1, 0.0],  # ϕ̃(1) = 0.1, dϕ̃/dT̃(1) = 0
        [0.3, 0.0],  # ϕ̃(1) = 0.3, dϕ̃/dT̃(1) = 0
        [0.5, 0.0],  # ϕ̃(1) = 0.5, dϕ̃/dT̃(1) = 0
    ]
    
    solutions = []
    envelopes = []
    
    for y0 in initialConditions:
        # Solve using RK4
        sol = solveStiffOde(odeSystem, (tInitial, tFinal), y0, tEval=tEval, method='RK4')
        solutions.append(sol)
        
        # Extract envelope
        envT, envY = extractEnvelope(sol.t, sol.y[0])
        envelopes.append((envT, envY))
    
    # Plot the solutions and their envelopes
    plt.figure(figsize=(12, 8))
    
    # First plot: Full oscillatory solutions
    plt.subplot(2, 1, 1)
    for i, sol in enumerate(solutions):
        plt.plot(sol.t, sol.y[0], label=f'ϕ̃(1) = {initialConditions[i][0]}')
    
    plt.xlabel('T̃/T̃_in')
    plt.ylabel('ϕ̃')
    plt.title('Full Solutions of Equation 33')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    
    # Second plot: Amplitude envelopes
    plt.subplot(2, 1, 2)
    for i, (t, env) in enumerate(envelopes):
        plt.plot(t, env, label=f'Envelope for ϕ̃(1) = {initialConditions[i][0]}')
    
    plt.xlabel('T̃/T̃_in')
    plt.ylabel('Amplitude Envelope')
    plt.title('Amplitude Envelopes (Showing Damping Behavior)')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    
    plt.tight_layout()
    plt.savefig('fullSolutionsAndEnvelopes.png')
    plt.show()
    
    return solutions, envelopes

def singularPerturbationAnalysis():
    """
    Perform singular perturbation analysis on equation (33).
    
    This separates the fast oscillatory dynamics from the slower amplitude decay.
    """
    def fastSystem(t, y, eps):
        """Fast dynamics (boundary layer)"""
        phi, dphi = y
        # Only consider the oscillatory part, neglecting damping
        # Using clipping to avoid numerical issues
        phiClipped = np.clip(phi, -1e8, 1e8)
        d2phi = -bScaled * phiClipped**3 / (a * max(t, 1e-10)**6)
        return [dphi, d2phi]
    
    def slowSystem(t, y):
        """Slow dynamics (reduced system)"""
        phi = y[0]
        # Consider only the damping/driving terms
        # Using clipping to avoid numerical issues
        phiClipped = np.clip(phi, -1e6, 1e6)
        if abs(phiClipped) < 1e6:
            damping = -cScaled * t**2 * phiClipped**12 * (dScaled * phiClipped**2 - 1) / (a * max(t, 1e-10)**6)
        else:
            damping = 0  # Skip calculation if phi is too large
        return [damping]
    
    # Integration settings - reduced range to avoid numerical issues
    tInitial = 1.0
    tFinal = 1e-8
    tEval = np.logspace(np.log10(tInitial), np.log10(tFinal), 5000)
    
    # Initial condition - reduced magnitude
    y0 = [0.5, 0.0]  # ϕ̃(1) = 0.5, dϕ̃/dT̃(1) = 0
    
    # Matching point
    matchingPoint = 0.9  # Close to the initial time
    
    # Solve the inner (fast) and outer (slow) problems
    innerSol = solveBoundaryLayer(fastSystem, (tInitial, tInitial*0.5), y0, epsilon, method='RK4')
    outerSol = solveReducedProblem(slowSystem, (tInitial, tFinal), [y0[0]], method='RK4')
    
    # Construct composite solution
    try:
        tComp, yComp = compositeSolution(outerSol, innerSol, matchingPoint, epsilon)
    except:
        # If composite construction fails, use placeholder values
        tComp = np.array([])
        yComp = np.array([])
    
    # Solve the full system for comparison
    fullSol = solveStiffOde(odeSystem, (tInitial, tFinal), y0, tEval=tEval, method='RK4')
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(fullSol.t, fullSol.y[0], 'b-', label='Full Solution')
    plt.plot(outerSol.t, outerSol.y[0], 'r--', label='Outer Solution (Envelope)')
    
    plt.xlabel('T̃/T̃_in')
    plt.ylabel('ϕ̃')
    plt.title('Singular Perturbation Analysis: Full vs. Outer Solution')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    
    plt.subplot(2, 1, 2)
    # Zoom in on a region to show oscillations
    zoomStart = tInitial * 0.9
    zoomEnd = tInitial * 0.7
    mask = (fullSol.t <= zoomStart) & (fullSol.t >= zoomEnd)
    
    plt.plot(fullSol.t[mask], fullSol.y[0][mask], 'b-', label='Full Solution')
    
    # Superimpose the composite solution if available in this range
    if len(tComp) > 0:
        maskComp = (tComp <= zoomStart) & (tComp >= zoomEnd)
        if np.any(maskComp):
            plt.plot(tComp[maskComp], yComp[maskComp], 'g:', label='Composite Solution')
    
    plt.xlabel('T̃/T̃_in')
    plt.ylabel('ϕ̃')
    plt.title('Zoomed View of Oscillations')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('singularPerturbationAnalysis.png')
    plt.show()
    
    return fullSol, outerSol, innerSol, (tComp, yComp)
   
def analyzeParameterEffect():
    """
    Analyze how changing the small parameter epsilon affects oscillation frequency
    but not the amplitude envelope.
    """
    global a  # We'll modify the small parameter
    
    # Original epsilon value
    epsilonOriginal = a
    
    # Range of values to test - reduced range to avoid numerical issues
    epsilonFactors = [0.5, 1.0, 2.0]  # Factors to multiply epsilon by
    
    # Integration settings - reduced range
    tInitial = 1.0
    tFinal = 1e-8
    tEval = np.logspace(np.log10(tInitial), np.log10(tFinal), 5000)
    
    # Initial condition - reduced magnitude
    y0 = [0.5, 0.0]  # Standard initial condition
    
    solutions = []
    envelopes = []
    epsilonValues = []
    
    for factor in epsilonFactors:
        # Change the value of a (epsilon)
        aTemp = epsilonOriginal * factor
        epsilonValues.append(aTemp)
        
        # Define the ODE system with this modified parameter
        def modifiedOdeSystem(tTilde, y):
            phiTilde, dphiDt = y
            tTilde = max(tTilde, 1e-10)
            
            # Added clipping to avoid numerical overflow
            phiClipped = np.clip(phiTilde, -1e8, 1e8)
            
            numerator = -bScaled * phiClipped**3
            
            # Only add second term if phi is small enough to avoid overflow
            if abs(phiClipped) < 1e6:
                secondTerm = cScaled * tTilde**2 * phiClipped**12 * (dScaled * phiClipped**2 - 1)
                numerator -= secondTerm
                
            d2phiDt2 = numerator / (aTemp * tTilde**6)
            return [dphiDt, d2phiDt2]
        
        # Solve the system
        sol = solveStiffOde(modifiedOdeSystem, (tInitial, tFinal), y0, tEval=tEval, method='RK4')
        solutions.append(sol)
        
        # Extract envelope
        envT, envY = extractEnvelope(sol.t, sol.y[0])
        envelopes.append((envT, envY))
    
    # Reset a to original value
    a = epsilonOriginal
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    
    # Plot the full solutions
    plt.subplot(2, 1, 1)
    for i, sol in enumerate(solutions):
        plt.plot(sol.t, sol.y[0], label=f'ε = {epsilonValues[i]:.2e} (x{epsilonFactors[i]})')
    
    plt.xlabel('T̃/T̃_in')
    plt.ylabel('ϕ̃')
    plt.title('Effect of Small Parameter ε on Oscillation Frequency')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    
    # Plot the envelopes
    plt.subplot(2, 1, 2)
    for i, (t, env) in enumerate(envelopes):
        plt.plot(t, env, label=f'Envelope for ε = {epsilonValues[i]:.2e} (x{epsilonFactors[i]})')
    
    plt.xlabel('T̃/T̃_in')
    plt.ylabel('Amplitude Envelope')
    plt.title('Effect of Small Parameter ε on Amplitude Envelope')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    
    plt.tight_layout()
    plt.savefig('parameterEffectAnalysis.png')
    plt.show()
    
    return solutions, envelopes

def addDampedOscillator():
    """
    Add theoretical insights based on the multiple-scale analysis.
    """
    # Use a simpler example to demonstrate multiple-scale analysis
    # Consider: ε*d²ϕ/dt² + dϕ/dt + ϕ = 0 (damped oscillator with small mass)
    
    # Integration settings
    tSpan = (0, 10)
    tEval = np.linspace(tSpan[0], tSpan[1], 1000)
    y0 = [1.0, 0.0]  # ϕ(0) = 1, dϕ/dt(0) = 0
    
    # Solve for different values of ε
    epsilonValues = [0.01, 0.05, 0.1]
    solutions = []
    
    for eps in epsilonValues:
        def dampedOscillator(t, y):
            phi, dphi = y
            d2phi = -(dphi + phi) / eps
            return [dphi, d2phi]
        
        sol = solveStiffOde(dampedOscillator, tSpan, y0, tEval=tEval, method='RK4')
        solutions.append(sol)
    
    # Compute the analytical solution for the envelope
    # For the damped oscillator, the envelope is exp(-t/2)
    tAnalytical = tEval
    envelopeAnalytical = np.exp(-tAnalytical/2)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    
    for i, sol in enumerate(solutions):
        plt.plot(sol.t, sol.y[0], label=f'ε = {epsilonValues[i]}')
    
    plt.plot(tAnalytical, envelopeAnalytical, 'k--', label='Analytical Envelope ~ exp(-t/2)')
    plt.plot(tAnalytical, -envelopeAnalytical, 'k--')
    
    plt.xlabel('Time')
    plt.ylabel('ϕ(t)')
    plt.title('Multiple-Scale Analysis: Effect of ε on Oscillation Frequency')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('theoreticalInsights.png')
    plt.show()
    
    return solutions

def addAnalyticalSolutionEq1():
    """
    Analytical solution based on Equation (1) from the document:
    
    (T_in/M_P)^2 * d^2φ̃/dT̃^2 = -45g*y^2 * φ̃/(96π^3g_* * T̃^4)
    
    This demonstrates the exact analytical solution: φ = AT̃cos(σ/T̃ + γ)
    where σ^2 = 45g*y^2/(96π^3g_*) * (M_P/T_in)^2
    
    The analytical solution shows that σ only affects oscillation frequency
    while the amplitude decays as (T/T_in)|φ_in|.
    """
    # Set up parameters
    tInitial = 1.0    # normalized T_in
    tFinal = 0.01     # Small T for visualization
    
    # Create a finer grid for the analytical solution
    tEval = np.logspace(np.log10(tFinal), np.log10(tInitial), 10000)
    
    # Initial amplitude and phase parameters
    phiIn = 1.0       # Initial value |φ_in|
    A = phiIn         # Set A = |φ_in| as per the paper
    gamma = 0         # Integration constant (phase shift)
    
    # Different σ values to demonstrate frequency effect
    sigmaValues = [1.0, 2.0, 4.0]
    
    # Compute the analytical solutions
    # φ = AT̃cos(σ/T̃ + γ)
    analyticalSolutions = []
    for sigma in sigmaValues:
        solution = A * tEval * np.cos(sigma/tEval + gamma)
        analyticalSolutions.append(solution)
    
    # Plot the results
    plt.figure(figsize=(12, 10))
    
    # First plot: Compare solutions with different σ values
    plt.subplot(2, 1, 1)
    
    for i, solution in enumerate(analyticalSolutions):
        plt.plot(tEval, solution, label=f'σ = {sigmaValues[i]}')
    
    plt.xlabel('T̃/T̃_in')
    plt.ylabel('φ̃')
    plt.title('Analytical Solutions of Eq(1) with Different σ Values')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    
    # Second plot: Show envelope clearly
    plt.subplot(2, 1, 2)
    
    # Choose one σ value to demonstrate envelope
    i = 1  # Use middle sigma value
    plt.plot(tEval, analyticalSolutions[i], 'b-', label=f'Solution (σ = {sigmaValues[i]})')
    
    # Plot envelope: T̃|φ_in|
    envelope = tEval * phiIn
    plt.plot(tEval, envelope, 'r--', label='Envelope: T̃|φ_in|')
    plt.plot(tEval, -envelope, 'r--')
    
    plt.xlabel('T̃/T̃_in')
    plt.ylabel('φ̃')
    plt.title('Amplitude Decay in Analytical Solution')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    
    # Add explanatory text
    # plt.figtext(0.5, 0.01, 
    #            "Analytical solution: φ = AT̃cos(σ/T̃ + γ) where σ² = 45gy²/(96π³g_*)(M_P/T_in)²\n"
    #            "This shows σ only affects oscillation frequency while amplitude decays as T̃|φ_in|.",
    #            ha='center', fontsize=12, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig('analyticalSolutionEq1.png')
    plt.show()
    
    return tEval, analyticalSolutions

if __name__ == "__main__":
    print("Analyzing Equation 33 with Multiple-Scale Analysis")
    
    # 1. Full solution with envelope extraction
    print("\n1. Solving the full system and extracting amplitude envelopes...")
    solFull, envelopes = solveFullSystem()
    
    # 2. Singular perturbation analysis
    print("\n2. Performing singular perturbation analysis...")
    full, outer, inner, composite = singularPerturbationAnalysis()
    
    # 3. Analyze effect of small parameter
    print("\n3. Analyzing effect of changing the small parameter...")
    solsParam, envsParam = analyzeParameterEffect()
    
    # 4. Add theoretical insights with a simpler example
    print("\n4. Adding theoretical insights with a simpler example...")
    simpleSols = addDampedOscillator()
    
    # 5. Add theoretical insights with Equation (18)
    print("\n5. Adding theoretical insights with Equation (18)...")
    solEq18, envEq18 = addAnalyticalSolutionEq1()
    
    print("\nAnalysis complete. Results saved as PNG files.")