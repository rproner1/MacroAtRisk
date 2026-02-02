

def compute_oos_r1_score(benchmark_pred, y_true, y_pred, q):
    
    """
    Computes the R1 score of a set of quantile forecasts and a set of returns.
    """

    return (1 - mean_pinball_loss(y_true, y_pred, alpha=q)/mean_pinball_loss(y_true, benchmark_pred, alpha=q))*100

def compute_oos_r2_score(y_true, y_pred, benchmark):
    
    """
    Computes out-of-sample (OOS) R2.
    """
    
    return (1 - mean_squared_error(y_true, y_pred) / mean_squared_error(y_true, benchmark))*100

def estimate_mean_from_quantiles(preds, weights: List[float]=[0.15, 0.225, 0.25, 0.225, 0.15]):
    return preds @ np.array(weights).reshape(-1,1)


def evaluate_model(y_pred, y_true, benchmark, target, model_name, quantiles, suppress_recession_dates: bool=False):
    import matplotlib.pyplot as plt
    r2_scores = {}

    target_preds = y_pred
    mean_preds = estimate_mean_from_quantiles(target_preds)

    r2 = compute_oos_r2_score(y_true, mean_preds.flatten(), benchmark[target]['Expanding_Mean'].values.flatten())

    plt.plot(y_true.index, mean_preds, label=f'Pred - R2={r2:.0f}%')
    plt.plot(y_true.index, y_true, label='Actual')
    plt.plot(y_true.index, benchmark[target]['Expanding_Mean'].values.flatten(), label='Naive')
    if not suppress_recession_dates:
        plt.axvspan('2001-03-01', '2001-11-01', -1,1, color='grey', alpha=0.25)
        plt.axvspan('2007-12-01', '2009-06-01', -1,1, color='grey', alpha=0.25)
        plt.axvspan('2020-02-01', '2020-04-01', -1,1, color='grey', alpha=0.25)
    plt.legend()
    plt.show()

    quantile_performance = {}
    for q in quantiles:
        q_preds = y_pred[:,quantiles.index(q)].flatten()
        q_r1 = compute_oos_r1_score(benchmark[target][f'Expanding_Q{int(q*100)}'].values.flatten(), y_true, q_preds, q)
        quantile_performance[f'Quantile {q}'] = round(q_r1,1)
        plt.plot(y_true.index, q_preds, linestyle='--', label=f'Quantile {q} -- R1={q_r1:.2f}%')
        # plt.plot(benchmark, linestyle=':', label=f'Benchmark Q{int(q*100)}')
    plt.plot(y_true.index, y_true, color='black', label='Actual')
    # plt.axvspan('1969-12-01', '1970-11-01', -1,1, color='grey', alpha=0.25)
    # plt.axvspan('1973-11-01', '1975-03-01', -1,1, color='grey', alpha=0.25)
    # plt.axvspan('1980-01-01', '1980-07-01', -1,1, color='grey', alpha=0.25)
    # plt.axvspan('1981-07-01', '1982-11-01', -1,1, color='grey', alpha=0.25)
    # plt.axvspan('1990-07-01', '1991-03-01', -1,1, color='grey', alpha=0.25)
    plt.axvspan('2001-03-01', '2001-11-01', -1,1, color='grey', alpha=0.25)
    plt.axvspan('2007-12-01', '2009-06-01', -1,1, color='grey', alpha=0.25)
    plt.axvspan('2020-02-01', '2020-04-01', -1,1, color='grey', alpha=0.25)
    plt.legend()
    plt.savefig(f'{model_name}_quantile_predictions_{target}.png')
    plt.show()
    quantile_performance.update({'Mean R1': round(np.mean(list(quantile_performance.values())),2), 'R2': round(r2,2)})
    print(quantile_performance)

    coverages = compute_quantile_coverage(y_true.values, y_pred, quantiles)

    for q, c in coverages.items():
        print(f"Quantile {q:.2f}: empirical coverage = {100*c:.2f}% (target {100*q:.2f}%)")

    plot_coverage(coverages)

    return quantile_performance

def compute_quantile_coverage(y_true, y_pred, quantiles):
    """
    Computes empirical coverage for each quantile.

    Parameters
    ----------
    y_true : np.ndarray, shape (T,)
        Actual values.
    y_pred : np.ndarray, shape (T, Q)
        Predicted quantiles (each column corresponds to a quantile).
    quantiles : list or np.ndarray, shape (Q,)
        Quantile levels, e.g. [0.05, 0.25, 0.5, 0.75, 0.95]

    Returns
    -------
    dict : {quantile: coverage}
        Dictionary mapping quantiles to their empirical coverage.
    """
    coverages = {}
    for i, q in enumerate(quantiles):
        coverage = np.mean(y_true <= y_pred[:, i])
        coverages[q] = coverage
    return coverages


def plot_coverage(coverages):
    import matplotlib.pyplot as plt
    qs = np.array(list(coverages.keys()))
    cs = np.array(list(coverages.values()))
    
    plt.figure(figsize=(5, 5))
    plt.plot(qs, cs, 'o-', label='Empirical coverage')
    plt.plot([0, 1], [0, 1], 'k--', label='Ideal 45° line')
    plt.xlabel("Nominal quantile (target)")
    plt.ylabel("Empirical coverage")
    plt.title("Quantile calibration plot")
    plt.legend()
    plt.grid(True)
    plt.show()


def qskt(q, loc, scale, shape, df):
    """
    Quantile function for the skewed t-distribution.
    
    Parameters:
        q: Probability values (between 0 and 1)
        loc: Location parameter
        scale: Scale parameter
        shape: Shape parameter (skewness)
        df: Degrees of freedom
    
    Returns:
        Quantiles of the skewed t-distribution
    """
    from scipy.stats import t
    import numpy as np
    
    # Formula for Azzalini-Capitanio skewed t-distribution quantiles
    z = t.ppf(q, df)
    a = shape * z * np.sqrt((df+1)/(df+z**2))
    
    # Handle numeric issues safely
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        z2 = z * z
        inv = np.where(z2 == 0.0, np.inf, df / z2)
        ratio = (df + 1.0) / (1.0 + inv)
        ratio = np.maximum(ratio, 0.0)
        a = shape * np.sign(z) * np.sqrt(ratio)
    
    # Calculate skewed-t quantile (location-scale form)
    delta = a / np.sqrt(1 + a**2)
    return loc + scale * z * (1 + delta) / np.sqrt(1 - delta**2)

def quantiles_interpolation(qq_targ, QQ, lc0=None, sc0=None, sh0=None):
    """
    Fits a skewed t-distribution to match target quantiles.
    
    Parameters:
        qq_targ: Vector containing the target quantiles
        QQ: Vector of quantile levels (should include 0.05, 0.25, 0.50, 0.75, 0.95)
        lc0: Initial condition for location parameter
        sc0: Initial condition for scale parameter
        sh0: Initial condition for shape parameter
    
    Returns:
        lc: Fitted location parameter
        sc: Fitted scale parameter
        sh: Fitted shape parameter
        df: Fitted degrees of freedom parameter
    """

    import numpy as np
    from scipy import optimize
    from scipy.stats import norm

    # Set bounds for optimization
    LB = [-20, 1e-6, -30]  # lower bounds: location, scale, shape
    UB = [20, 50, 30]      # upper bounds
    
    # Find indices of target quantiles
    jq50 = np.argmin(np.abs(QQ - 0.50))
    jq25 = np.argmin(np.abs(QQ - 0.25))
    jq75 = np.argmin(np.abs(QQ - 0.75))
    jq05 = np.argmin(np.abs(QQ - 0.05))
    jq95 = np.argmin(np.abs(QQ - 0.95))
    
    # Set initial conditions if not provided
    if lc0 is None or sc0 is None or sh0 is None:
        iqn = norm.ppf(0.75) - norm.ppf(0.25)
        lc0 = qq_targ[jq50]
        sc0 = (qq_targ[jq75] - qq_targ[jq25]) / iqn
        sh0 = 0
    
    X0 = [lc0, sc0, sh0]
    
    # Select target quantiles
    select = [jq05, jq25, jq75, jq95]
    QQ_select = QQ[select]
    qq_targ_select = qq_targ[select]
    
    # Optimize for each possible value of degrees of freedom
    par = np.full((30, 3), np.nan)
    ssq = np.full(30, np.nan)
    
    for df in range(1, 31):
        def objective(x):
            return qq_targ_select - qskt(QQ_select, x[0], x[1], x[2], df)
        
        result = optimize.least_squares(
            objective, X0, bounds=(LB, UB), 
            method='trf', ftol=1e-6, xtol=1e-6
        )
        
        par[df-1, :] = result.x
        ssq[df-1] = np.sum(result.fun**2)
    
    # Find best fit
    best_df_idx = np.argmin(ssq)
    df = best_df_idx + 1  # df ranges from 1 to 30
    lc = par[best_df_idx, 0]
    sc = par[best_df_idx, 1]
    sh = par[best_df_idx, 2]
    
    return lc, sc, sh, df

class SkewedTDistribution:
    def __init__(self, loc, scale, shape, df):
        self.loc = loc
        self.scale = scale
        self.shape = shape
        self.df = df

    

    def pdf(self, x):
        """Probability density function"""
        
        z = (x - self.loc) / self.scale

        # Stable computation of a
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            z2 = z * z
            inv = np.where(z2 == 0.0, np.inf, self.df / z2)
            ratio = (self.df + 1.0) / (1.0 + inv)
            ratio = np.maximum(ratio, 0.0)
            a = self.shape * np.sign(z) * np.sqrt(ratio)

        # Clip a to prevent overflow
        a = np.clip(a, -1e6, 1e6)

        pdf_z = t.pdf(z, self.df)
        cdf_a = t.cdf(a, self.df + 1)
        return 2.0 / self.scale * pdf_z * cdf_a

    def cdf(self, x):
        """Cumulative distribution function"""
        
        # For more accurate results with extreme values, we integrate the PDF
        # For typical cases, we use a direct formula
        
        if np.isscalar(x):
            if x < self.loc - 10*self.scale:  # Far in left tail
                return 0.0
            if x > self.loc + 10*self.scale:  # Far in right tail
                return 1.0
                
            # Numerical integration for accuracy
            result, _ = integrate.quad(self.pdf, -np.inf, x)
            return result
        else:
            # Vectorized version
            result = np.zeros_like(x, dtype=float)
            
            # Far left and right tails
            result[x < self.loc - 10*self.scale] = 0.0
            result[x > self.loc + 10*self.scale] = 1.0

            # Middle range - integrate each point
            middle = (x >= self.loc - 10*self.scale) & (x <= self.loc + 10*self.scale)
            for i, xi in enumerate(x[middle]):
                result[middle][i], _ = integrate.quad(self.pdf, -np.inf, xi)

            return result

    def ppf(self, q):
        """Percent point function (inverse CDF)"""
        # For a scalar, use binary search
        if np.isscalar(q):
            if q <= 0:
                return -np.inf
            if q >= 1:
                return np.inf
                
            # Binary search
            left = self.loc - 10*self.scale
            right = self.loc + 10*self.scale

            for _ in range(50):  # Usually converges in < 50 iterations
                mid = (left + right) / 2
                if self.cdf(mid) < q:
                    left = mid
                else:
                    right = mid
                    
                if right - left < 1e-10:
                    break
                    
            return (left + right) / 2
        else:
            # Vectorized version
            return np.array([self.ppf(qi) for qi in q])

    def plot(self, x_range=None, fig=None, ax=None, plot_type='pdf', **kwargs):
        """
        Plot the distribution's PDF or CDF.
        
        Parameters:
            x_range: Optional range for x-values as [min, max]
            fig, ax: Optional matplotlib figure and axes
            plot_type: 'pdf' or 'cdf'
            **kwargs: Additional arguments for matplotlib plot function
        """
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
        if x_range is None:
            # Determine reasonable plotting range
            if self.shape < 0:  # Left skewed
                x_range = [self.loc - 3*self.scale, self.loc + 1.5*self.scale]
            elif self.shape > 0:  # Right skewed
                x_range = [self.loc - 1.5*self.scale, self.loc + 3*self.scale]
            else:  # Symmetric
                x_range = [self.loc - 3*self.scale, self.loc + 3*self.scale]

        x = np.linspace(x_range[0], x_range[1], 1000)
        
        if plot_type.lower() == 'pdf':
            y = self.pdf(x)
            ax.plot(x, y, **kwargs)
            ax.set_title(f'Skewed t-distribution PDF (loc={self.loc}, scale={self.scale}, shape={self.shape}, df={self.df})')
            ax.set_ylabel('Probability Density')
        elif plot_type.lower() == 'cdf':
            y = self.cdf(x)
            ax.plot(x, y, **kwargs)
            ax.set_title(f'Skewed t-distribution CDF (loc={self.loc}, scale={self.scale}, shape={self.shape}, df={self.df})')
            ax.set_ylabel('Cumulative Probability')
        
        ax.set_xlabel('x')
        ax.grid(True, alpha=0.3)
        
        return fig, ax
    
    


def create_skewed_t_distribution(loc, scale, shape, df):
    """
    Creates a skewed t-distribution with specified parameters.
    
    Parameters:
        loc (float): Location parameter
        scale (float): Scale parameter (must be positive)
        shape (float): Shape parameter (skewness)
        df (float): Degrees of freedom (must be positive)
    
    Returns:
        A dictionary with methods pdf, cdf, ppf, and plot
    """
    import numpy as np
    from scipy.stats import t
    import matplotlib.pyplot as plt
    from scipy import integrate
    
    if scale <= 0:
        raise ValueError("Scale parameter must be positive")
    if df <= 0:
        raise ValueError("Degrees of freedom must be positive")
        
    def pdf(x):
        """Probability density function"""
        z = (x - loc) / scale
        
        # Stable computation of a
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            z2 = z * z
            inv = np.where(z2 == 0.0, np.inf, df / z2)
            ratio = (df + 1.0) / (1.0 + inv)
            ratio = np.maximum(ratio, 0.0)
            a = shape * np.sign(z) * np.sqrt(ratio)
        
        # Clip a to prevent overflow
        a = np.clip(a, -1e6, 1e6)
        
        pdf_z = t.pdf(z, df)
        cdf_a = t.cdf(a, df + 1)
        return 2.0 / scale * pdf_z * cdf_a
    
    def cdf(x):
        """Cumulative distribution function"""
        # For more accurate results with extreme values, we integrate the PDF
        # For typical cases, we use a direct formula
        
        if np.isscalar(x):
            if x < loc - 10*scale:  # Far in left tail
                return 0.0
            if x > loc + 10*scale:  # Far in right tail
                return 1.0
                
            # Numerical integration for accuracy
            result, _ = integrate.quad(pdf, -np.inf, x)
            return result
        else:
            # Vectorized version
            result = np.zeros_like(x, dtype=float)
            
            # Far left and right tails
            result[x < loc - 10*scale] = 0.0
            result[x > loc + 10*scale] = 1.0
            
            # Middle range - integrate each point
            middle = (x >= loc - 10*scale) & (x <= loc + 10*scale)
            for i, xi in enumerate(x[middle]):
                result[middle][i], _ = integrate.quad(pdf, -np.inf, xi)
                
            return result
    
    def ppf(q):
        """Percent point function (inverse CDF)"""
        # For a scalar, use binary search
        if np.isscalar(q):
            if q <= 0:
                return -np.inf
            if q >= 1:
                return np.inf
                
            # Binary search
            left = loc - 10*scale
            right = loc + 10*scale
            
            for _ in range(50):  # Usually converges in < 50 iterations
                mid = (left + right) / 2
                if cdf(mid) < q:
                    left = mid
                else:
                    right = mid
                    
                if right - left < 1e-10:
                    break
                    
            return (left + right) / 2
        else:
            # Vectorized version
            return np.array([ppf(qi) for qi in q])
    
    def plot(x_range=None, fig=None, ax=None, plot_type='pdf', **kwargs):
        """
        Plot the distribution's PDF or CDF.
        
        Parameters:
            x_range: Optional range for x-values as [min, max]
            fig, ax: Optional matplotlib figure and axes
            plot_type: 'pdf' or 'cdf'
            **kwargs: Additional arguments for matplotlib plot function
        """
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
        if x_range is None:
            # Determine reasonable plotting range
            if shape < 0:  # Left skewed
                x_range = [loc - 3*scale, loc + 1.5*scale]
            elif shape > 0:  # Right skewed
                x_range = [loc - 1.5*scale, loc + 3*scale]
            else:  # Symmetric
                x_range = [loc - 3*scale, loc + 3*scale]
                
        x = np.linspace(x_range[0], x_range[1], 1000)
        
        if plot_type.lower() == 'pdf':
            y = pdf(x)
            ax.plot(x, y, **kwargs)
            ax.set_title(f'Skewed t-distribution PDF (loc={loc}, scale={scale}, shape={shape}, df={df})')
            ax.set_ylabel('Probability Density')
        elif plot_type.lower() == 'cdf':
            y = cdf(x)
            ax.plot(x, y, **kwargs)
            ax.set_title(f'Skewed t-distribution CDF (loc={loc}, scale={scale}, shape={shape}, df={df})')
            ax.set_ylabel('Cumulative Probability')
        
        ax.set_xlabel('x')
        ax.grid(True, alpha=0.3)
        
        return fig, ax
    
    # Return dictionary with all methods
    return {
        'pdf': pdf,
        'cdf': cdf,
        'ppf': ppf,
        'plot': plot,
        'params': {'loc': loc, 'scale': scale, 'shape': shape, 'df': df}
    }

