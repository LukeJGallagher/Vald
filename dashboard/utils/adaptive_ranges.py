"""
Adaptive Reference Ranges using EM Approximation
Based on: Kenny McMillan PhD research
Paper: "A comparison of methods to generate adaptive reference ranges in longitudinal monitoring"
https://bjsm.bmj.com/content/56/24/1451

Implements:
- EM (Expectation-Maximization) algorithm for baseline estimation
- Adaptive ranges that adjust to athlete's current state
- Flagging system for meaningful changes
- Adjustable confidence intervals
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class AdaptiveRangeCalculator:
    """
    Calculate adaptive reference ranges using EM approximation

    Parameters:
    -----------
    alpha : float
        Weighting factor for EM algorithm (0-1)
        Higher values = more weight to recent data
        Default: 0.3 (recommended in literature)

    confidence_level : float
        Confidence level for ranges (e.g., 0.95 for 95% CI)
        Default: 0.95

    min_observations : int
        Minimum number of observations required
        Default: 10
    """

    def __init__(self, alpha: float = 0.3, confidence_level: float = 0.95,
                 min_observations: int = 10):
        self.alpha = alpha
        self.confidence_level = confidence_level
        self.min_observations = min_observations
        self.z_score = stats.norm.ppf((1 + confidence_level) / 2)

    def calculate_em_baseline(self, values: np.array) -> Dict:
        """
        Calculate baseline using EM algorithm

        EM approximation formula:
        μ_t = α * x_t + (1 - α) * μ_(t-1)
        σ²_t = α * (x_t - μ_t)² + (1 - α) * σ²_(t-1)

        Where:
        - μ_t = estimated mean at time t
        - σ²_t = estimated variance at time t
        - x_t = observation at time t
        - α = weighting factor
        """

        if len(values) < self.min_observations:
            return None

        # Initialize with first few observations
        mu = np.mean(values[:5]) if len(values) >= 5 else np.mean(values[:3])
        sigma_sq = np.var(values[:5]) if len(values) >= 5 else np.var(values[:3])

        # Store evolution
        mu_history = [mu]
        sigma_history = [np.sqrt(sigma_sq)]

        # EM iterations
        for x_t in values:
            # Update mean
            mu_new = self.alpha * x_t + (1 - self.alpha) * mu

            # Update variance
            sigma_sq_new = self.alpha * (x_t - mu_new) ** 2 + (1 - self.alpha) * sigma_sq

            mu = mu_new
            sigma_sq = sigma_sq_new

            mu_history.append(mu)
            sigma_history.append(np.sqrt(sigma_sq))

        return {
            'final_mean': mu,
            'final_sd': np.sqrt(sigma_sq),
            'mean_history': np.array(mu_history),
            'sd_history': np.array(sigma_history)
        }

    def calculate_adaptive_range(self, values: np.array, dates: Optional[np.array] = None) -> Dict:
        """
        Calculate adaptive reference range

        Returns:
        --------
        Dict containing:
        - upper_limit: Upper bound of range
        - lower_limit: Lower bound of range
        - mean: Estimated mean
        - sd: Estimated standard deviation
        - flagged_indices: Indices of values outside range
        """

        baseline = self.calculate_em_baseline(values)

        if baseline is None:
            return None

        # Calculate ranges
        upper_limit = baseline['final_mean'] + self.z_score * baseline['final_sd']
        lower_limit = baseline['final_mean'] - self.z_score * baseline['final_sd']

        # Adaptive ranges for each observation
        upper_limits = baseline['mean_history'] + self.z_score * baseline['sd_history']
        lower_limits = baseline['mean_history'] - self.z_score * baseline['sd_history']

        # Flag outliers
        flagged_indices = np.where((values > upper_limits[:-1]) | (values < lower_limits[:-1]))[0]

        return {
            'upper_limit': upper_limit,
            'lower_limit': lower_limit,
            'mean': baseline['final_mean'],
            'sd': baseline['final_sd'],
            'upper_limits_history': upper_limits,
            'lower_limits_history': lower_limits,
            'mean_history': baseline['mean_history'],
            'sd_history': baseline['sd_history'],
            'flagged_indices': flagged_indices,
            'flagged_count': len(flagged_indices),
            'confidence_level': self.confidence_level,
            'alpha': self.alpha
        }

    def calculate_smallest_worthwhile_change(self, sd: float, cohens_d: float = 0.2) -> float:
        """
        Calculate Smallest Worthwhile Change

        SWC = Cohen's d * SD
        Default Cohen's d = 0.2 (small effect size)
        """
        return cohens_d * sd

    def calculate_typical_error(self, values: np.array) -> float:
        """
        Calculate Typical Error (TE) from test-retest data

        TE = SD(differences) / sqrt(2)
        """
        if len(values) < 2:
            return None

        differences = np.diff(values)
        te = np.std(differences, ddof=1) / np.sqrt(2)

        return te


class AdaptiveRangeDashboard:
    """Create visualizations for adaptive ranges"""

    def __init__(self, calculator: AdaptiveRangeCalculator):
        self.calculator = calculator

    def plot_adaptive_range(self, values: np.array, dates: Optional[np.array] = None,
                           metric_name: str = "Metric", athlete_name: str = "Athlete") -> go.Figure:
        """
        Create adaptive range plot similar to Kenny McMillan's Power BI dashboard

        Shows:
        - Individual data points
        - Adaptive upper/lower limits
        - EM-estimated mean
        - Flagged observations
        """

        range_data = self.calculator.calculate_adaptive_range(values, dates)

        if range_data is None:
            return None

        # Create x-axis (dates or indices)
        if dates is not None:
            x = dates
            xaxis_title = "Date"
        else:
            x = np.arange(len(values))
            xaxis_title = "Test Number"

        fig = go.Figure()

        # Upper limit
        fig.add_trace(go.Scatter(
            x=x,
            y=range_data['upper_limits_history'][:-1],
            mode='lines',
            name=f'Upper Limit ({self.calculator.confidence_level * 100:.0f}% CI)',
            line=dict(color='red', width=2, dash='dash'),
            fillcolor='rgba(255, 0, 0, 0.1)',
            fill='tonexty'
        ))

        # Lower limit
        fig.add_trace(go.Scatter(
            x=x,
            y=range_data['lower_limits_history'][:-1],
            mode='lines',
            name=f'Lower Limit ({self.calculator.confidence_level * 100:.0f}% CI)',
            line=dict(color='red', width=2, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.1)'
        ))

        # EM Mean
        fig.add_trace(go.Scatter(
            x=x,
            y=range_data['mean_history'][:-1],
            mode='lines',
            name='EM Estimated Mean',
            line=dict(color='#007167', width=3)
        ))

        # Actual values (non-flagged)
        non_flagged = [i for i in range(len(values)) if i not in range_data['flagged_indices']]
        if non_flagged:
            fig.add_trace(go.Scatter(
                x=x[non_flagged],
                y=values[non_flagged],
                mode='markers',
                name='Within Range',
                marker=dict(size=10, color='#007167', symbol='circle',
                          line=dict(width=2, color='white'))
            ))

        # Flagged values
        if len(range_data['flagged_indices']) > 0:
            flagged_x = x[range_data['flagged_indices']]
            flagged_y = values[range_data['flagged_indices']]

            fig.add_trace(go.Scatter(
                x=flagged_x,
                y=flagged_y,
                mode='markers',
                name='Outside Range (Flagged)',
                marker=dict(size=14, color='red', symbol='x',
                          line=dict(width=3, color='darkred'))
            ))

        # Layout
        fig.update_layout(
            title=dict(
                text=f'Adaptive Range Monitoring - {athlete_name}<br><sub>{metric_name} (α={self.calculator.alpha})</sub>',
                font=dict(size=20, family='Arial Black', color='#007167'),
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                title=xaxis_title,
                showgrid=True,
                gridcolor='lightgray'
            ),
            yaxis=dict(
                title=metric_name,
                showgrid=True,
                gridcolor='lightgray'
            ),
            plot_bgcolor='white',
            hovermode='closest',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(255,255,255,0.8)'
            ),
            height=500
        )

        # Add annotation with stats
        fig.add_annotation(
            xref='paper', yref='paper',
            x=0.99, y=0.01,
            text=f"Mean: {range_data['mean']:.2f} ± {range_data['sd']:.2f}<br>" +
                 f"Range: [{range_data['lower_limit']:.2f}, {range_data['upper_limit']:.2f}]<br>" +
                 f"Flagged: {range_data['flagged_count']}/{len(values)} ({range_data['flagged_count']/len(values)*100:.1f}%)",
            showarrow=False,
            font=dict(size=11, color='#333', family='Arial'),
            align='right',
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#007167',
            borderwidth=2,
            borderpad=8,
            xanchor='right',
            yanchor='bottom'
        )

        return fig

    def plot_multi_metric_dashboard(self, df: pd.DataFrame, athlete_name: str,
                                    metrics: List[str], date_column: str = 'recordedDateUtc') -> go.Figure:
        """
        Create multi-metric adaptive range dashboard
        Similar to Kenny McMillan's CMJ dashboard
        """

        n_metrics = len(metrics)
        fig = make_subplots(
            rows=n_metrics, cols=1,
            subplot_titles=metrics,
            vertical_spacing=0.08,
            row_heights=[1/n_metrics] * n_metrics
        )

        for i, metric in enumerate(metrics, 1):
            if metric not in df.columns:
                continue

            # Get data
            data = df[[date_column, metric]].dropna()
            if len(data) < self.calculator.min_observations:
                continue

            dates = pd.to_datetime(data[date_column])
            values = data[metric].values

            # Calculate adaptive range
            range_data = self.calculator.calculate_adaptive_range(values, dates)

            if range_data is None:
                continue

            # Upper limit
            fig.add_trace(go.Scatter(
                x=dates, y=range_data['upper_limits_history'][:-1],
                mode='lines', name=f'Upper ({metric})',
                line=dict(color='red', width=1, dash='dash'),
                fill='tonexty', fillcolor='rgba(255, 0, 0, 0.05)',
                showlegend=(i == 1)
            ), row=i, col=1)

            # Lower limit
            fig.add_trace(go.Scatter(
                x=dates, y=range_data['lower_limits_history'][:-1],
                mode='lines', name=f'Lower ({metric})',
                line=dict(color='red', width=1, dash='dash'),
                fill='tonexty', fillcolor='rgba(255, 0, 0, 0.05)',
                showlegend=(i == 1)
            ), row=i, col=1)

            # EM Mean
            fig.add_trace(go.Scatter(
                x=dates, y=range_data['mean_history'][:-1],
                mode='lines', name=f'Mean ({metric})',
                line=dict(color='#007167', width=2),
                showlegend=(i == 1)
            ), row=i, col=1)

            # Actual values
            non_flagged = [j for j in range(len(values)) if j not in range_data['flagged_indices']]
            if non_flagged:
                fig.add_trace(go.Scatter(
                    x=dates.iloc[non_flagged], y=values[non_flagged],
                    mode='markers', name='Within Range',
                    marker=dict(size=8, color='#007167', symbol='circle'),
                    showlegend=(i == 1)
                ), row=i, col=1)

            # Flagged
            if len(range_data['flagged_indices']) > 0:
                flagged_dates = dates.iloc[range_data['flagged_indices']]
                flagged_values = values[range_data['flagged_indices']]

                fig.add_trace(go.Scatter(
                    x=flagged_dates, y=flagged_values,
                    mode='markers', name='Flagged',
                    marker=dict(size=12, color='red', symbol='x'),
                    showlegend=(i == 1)
                ), row=i, col=1)

            # Update axes for this subplot
            fig.update_yaxes(title_text=metric, row=i, col=1)

        # Update layout
        fig.update_layout(
            title=dict(
                text=f'CMJ Adaptive Range Monitoring - {athlete_name}<br><sub>EM Approximation (α={self.calculator.alpha}, {self.calculator.confidence_level*100:.0f}% CI)</sub>',
                font=dict(size=22, family='Arial Black', color='#007167'),
                x=0.5,
                xanchor='center'
            ),
            height=300 * n_metrics,
            plot_bgcolor='white',
            hovermode='x unified',
            showlegend=True
        )

        fig.update_xaxes(title_text="Date")

        return fig


def calculate_range_comparison(values: np.array, alpha_values: List[float] = [0.2, 0.3, 0.4]) -> Dict:
    """
    Compare adaptive ranges with different alpha values

    Helps determine optimal alpha for specific metric/athlete
    """

    results = {}

    for alpha in alpha_values:
        calc = AdaptiveRangeCalculator(alpha=alpha)
        range_data = calc.calculate_adaptive_range(values)

        if range_data:
            results[alpha] = {
                'mean': range_data['mean'],
                'sd': range_data['sd'],
                'flagged_count': range_data['flagged_count'],
                'flagged_percent': range_data['flagged_count'] / len(values) * 100
            }

    return results


def get_recommended_alpha(cv_percent: float) -> float:
    """
    Get recommended alpha value based on metric CV%

    Guidelines from research:
    - Low variability metrics (CV < 5%): α = 0.2 (slower adaptation)
    - Moderate variability (CV 5-10%): α = 0.3 (balanced)
    - High variability (CV > 10%): α = 0.4 (faster adaptation)
    """

    if cv_percent < 5:
        return 0.2
    elif cv_percent < 10:
        return 0.3
    else:
        return 0.4


# ============================================================================
# CMJ-SPECIFIC DEFAULTS
# ============================================================================

CMJ_METRIC_ALPHAS = {
    'Jump Height (Flight Time) [cm]': 0.3,  # Moderate variability
    'Peak Power / BM_Trial': 0.3,
    'RSI-modified (Imp-Mom)_Trial': 0.35,  # Higher variability
    'Contraction Time [ms]': 0.25,  # Lower variability
    'Peak Force [N]': 0.3,
    'Relative Peak Force [N/kg]': 0.3
}


if __name__ == "__main__":
    # Test with sample data
    print("Adaptive Ranges Module loaded successfully!")
    print("\nBased on Kenny McMillan PhD research")
    print("Paper: 'A comparison of methods to generate adaptive reference ranges'")
    print("\nExample usage:")
    print("  calc = AdaptiveRangeCalculator(alpha=0.3)")
    print("  range_data = calc.calculate_adaptive_range(jump_heights)")
    print("  dashboard = AdaptiveRangeDashboard(calc)")
    print("  fig = dashboard.plot_adaptive_range(jump_heights, dates, 'Jump Height')")
