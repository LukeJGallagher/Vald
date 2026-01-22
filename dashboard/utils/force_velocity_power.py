"""
Force-Velocity-Power Profile Analysis
Based on: https://optimumsportsperformance.com/blog/r-tips-tricks-force-velocity-power-profile-graphs-in-r-shiny/

Implements:
- Force-Velocity Profile
- Power-Force Profile
- Power-Velocity Profile
- F-V imbalance detection
- Optimal power zone identification
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
from scipy import stats
from scipy.optimize import curve_fit


def calculate_fv_profile(jump_data: pd.DataFrame) -> Dict:
    """
    Calculate Force-Velocity profile from jump data

    Requires jump data with:
    - Jump Height (cm)
    - Body Mass (kg)
    - Load (kg) - for loaded jumps
    """

    # Extract required metrics
    if 'Jump Height (Flight Time) [cm]' in jump_data.columns:
        heights = jump_data['Jump Height (Flight Time) [cm]'].values
    else:
        return None

    # Calculate velocity from jump height
    # v = sqrt(2 * g * h)  where g = 9.81 m/s^2
    velocities = np.sqrt(2 * 9.81 * (heights / 100))  # Convert cm to m

    # Get or estimate force
    if 'Peak Force [N]' in jump_data.columns:
        forces = jump_data['Peak Force [N]'].values
    elif 'Relative Peak Force [N/kg]' in jump_data.columns and 'weight' in jump_data.columns:
        forces = jump_data['Relative Peak Force [N/kg]'] * jump_data['weight']
    else:
        # Estimate from body mass if available
        if 'weight' in jump_data.columns:
            body_mass = jump_data['weight'].values
            forces = body_mass * 9.81  # Rough estimate
        else:
            return None

    # Linear regression F = F0 - slope * V
    # where F0 = max force, V0 = max velocity
    slope, intercept, r_value, p_value, std_err = stats.linregress(velocities, forces)

    F0 = intercept  # Max theoretical force (when v = 0)
    V0 = -intercept / slope  # Max theoretical velocity (when F = 0)
    Pmax = (F0 * V0) / 4  # Max power at optimal point

    # Optimal force and velocity for max power
    F_opt = F0 / 2
    V_opt = V0 / 2

    # F-V imbalance (difference from 100% - optimal 1:1 ratio)
    FV_imbalance = (F0 / (F0 + V0) - 0.5) * 100

    return {
        'F0': F0,  # Maximal force (N)
        'V0': V0,  # Maximal velocity (m/s)
        'Pmax': Pmax,  # Maximal power (W)
        'F_opt': F_opt,  # Optimal force for Pmax
        'V_opt': V_opt,  # Optimal velocity for Pmax
        'FV_imbalance': FV_imbalance,  # % imbalance
        'slope': slope,
        'r_squared': r_value ** 2,
        'velocities': velocities,
        'forces': forces
    }


def create_fv_profile_plot(profile: Dict, athlete_name: str = "Athlete") -> go.Figure:
    """
    Create Force-Velocity profile plot
    Similar to Optimum Sports Performance example
    """

    if profile is None:
        return None

    # Create figure
    fig = go.Figure()

    # Actual data points
    fig.add_trace(go.Scatter(
        x=profile['velocities'],
        y=profile['forces'],
        mode='markers',
        name='Actual Tests',
        marker=dict(
            size=12,
            color='#1D4D3B',  # Saudi teal
            symbol='circle',
            line=dict(width=2, color='white')
        )
    ))

    # Theoretical F-V line
    v_theoretical = np.linspace(0, profile['V0'], 100)
    f_theoretical = profile['F0'] + profile['slope'] * v_theoretical

    fig.add_trace(go.Scatter(
        x=v_theoretical,
        y=f_theoretical,
        mode='lines',
        name='F-V Profile',
        line=dict(
            color='#a08e66',  # Saudi gold
            width=3,
            dash='solid'
        )
    ))

    # Optimal power point
    fig.add_trace(go.Scatter(
        x=[profile['V_opt']],
        y=[profile['F_opt']],
        mode='markers+text',
        name='Optimal Power',
        marker=dict(
            size=16,
            color='red',
            symbol='star',
            line=dict(width=2, color='white')
        ),
        text=['Pmax'],
        textposition='top center',
        textfont=dict(size=12, color='red', family='Arial Black')
    ))

    # F0 point
    fig.add_trace(go.Scatter(
        x=[0],
        y=[profile['F0']],
        mode='markers+text',
        name='F0 (Max Force)',
        marker=dict(size=12, color='#1D4D3B', symbol='diamond'),
        text=[f"F0: {profile['F0']:.0f} N"],
        textposition='top right'
    ))

    # V0 point
    fig.add_trace(go.Scatter(
        x=[profile['V0']],
        y=[0],
        mode='markers+text',
        name='V0 (Max Velocity)',
        marker=dict(size=12, color='#1D4D3B', symbol='diamond'),
        text=[f"V0: {profile['V0']:.2f} m/s"],
        textposition='top left'
    ))

    # Layout
    fig.update_layout(
        title=dict(
            text=f'Force-Velocity Profile - {athlete_name}',
            font=dict(size=20, family='Arial Black', color='#1D4D3B')
        ),
        xaxis=dict(
            title='Velocity (m/s)',
            showgrid=True,
            gridcolor='lightgray',
            range=[0, profile['V0'] * 1.1]
        ),
        yaxis=dict(
            title='Force (N)',
            showgrid=True,
            gridcolor='lightgray',
            range=[0, profile['F0'] * 1.1]
        ),
        plot_bgcolor='white',
        hovermode='closest',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor='rgba(255,255,255,0.8)'
        )
    )

    # Add annotations
    fig.add_annotation(
        x=profile['V_opt'],
        y=profile['F_opt'],
        text=f"Pmax: {profile['Pmax']:.0f} W",
        showarrow=True,
        arrowhead=2,
        arrowcolor='red',
        ax=-40,
        ay=-40,
        font=dict(size=12, color='red', family='Arial Black')
    )

    # FV Imbalance indicator
    imbalance_color = '#1D4D3B' if abs(profile['FV_imbalance']) < 10 else '#a08e66' if abs(profile['FV_imbalance']) < 20 else 'red'
    imbalance_text = "Force" if profile['FV_imbalance'] > 0 else "Velocity"

    fig.add_annotation(
        xref='paper', yref='paper',
        x=0.02, y=0.98,
        text=f"F-V Imbalance: {abs(profile['FV_imbalance']):.1f}%<br>({imbalance_text} Dominant)",
        showarrow=False,
        font=dict(size=11, color=imbalance_color, family='Arial'),
        align='left',
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor=imbalance_color,
        borderwidth=2,
        borderpad=8
    )

    return fig


def create_power_force_plot(profile: Dict, athlete_name: str = "Athlete") -> go.Figure:
    """Create Power-Force profile"""

    if profile is None:
        return None

    # Calculate power curve
    forces = np.linspace(0, profile['F0'], 100)
    velocities = (profile['F0'] - forces) / abs(profile['slope'])
    powers = forces * velocities

    fig = go.Figure()

    # Power curve
    fig.add_trace(go.Scatter(
        x=forces,
        y=powers,
        mode='lines',
        name='Power-Force',
        line=dict(color='#1D4D3B', width=3),
        fill='tozeroy',
        fillcolor='rgba(0, 113, 103, 0.1)'
    ))

    # Optimal point
    fig.add_trace(go.Scatter(
        x=[profile['F_opt']],
        y=[profile['Pmax']],
        mode='markers+text',
        name='Optimal',
        marker=dict(size=16, color='red', symbol='star'),
        text=[f"Pmax: {profile['Pmax']:.0f} W"],
        textposition='top center'
    ))

    # Actual data points
    actual_powers = profile['forces'] * profile['velocities']
    fig.add_trace(go.Scatter(
        x=profile['forces'],
        y=actual_powers,
        mode='markers',
        name='Actual Tests',
        marker=dict(size=10, color='#a08e66', symbol='circle')
    ))

    fig.update_layout(
        title=dict(
            text=f'Power-Force Profile - {athlete_name}',
            font=dict(size=20, family='Arial Black', color='#1D4D3B')
        ),
        xaxis=dict(title='Force (N)', showgrid=True),
        yaxis=dict(title='Power (W)', showgrid=True),
        plot_bgcolor='white',
        hovermode='closest'
    )

    return fig


def create_power_velocity_plot(profile: Dict, athlete_name: str = "Athlete") -> go.Figure:
    """Create Power-Velocity profile"""

    if profile is None:
        return None

    # Calculate power curve
    velocities = np.linspace(0, profile['V0'], 100)
    forces = profile['F0'] + profile['slope'] * velocities
    powers = forces * velocities

    fig = go.Figure()

    # Power curve
    fig.add_trace(go.Scatter(
        x=velocities,
        y=powers,
        mode='lines',
        name='Power-Velocity',
        line=dict(color='#1D4D3B', width=3),
        fill='tozeroy',
        fillcolor='rgba(0, 113, 103, 0.1)'
    ))

    # Optimal point
    fig.add_trace(go.Scatter(
        x=[profile['V_opt']],
        y=[profile['Pmax']],
        mode='markers+text',
        name='Optimal',
        marker=dict(size=16, color='red', symbol='star'),
        text=[f"Pmax: {profile['Pmax']:.0f} W"],
        textposition='top center'
    ))

    # Actual data points
    actual_powers = profile['forces'] * profile['velocities']
    fig.add_trace(go.Scatter(
        x=profile['velocities'],
        y=actual_powers,
        mode='markers',
        name='Actual Tests',
        marker=dict(size=10, color='#a08e66', symbol='circle')
    ))

    fig.update_layout(
        title=dict(
            text=f'Power-Velocity Profile - {athlete_name}',
            font=dict(size=20, family='Arial Black', color='#1D4D3B')
        ),
        xaxis=dict(title='Velocity (m/s)', showgrid=True),
        yaxis=dict(title='Power (W)', showgrid=True),
        plot_bgcolor='white',
        hovermode='closest'
    ))

    return fig


def create_combined_fvp_dashboard(profile: Dict, athlete_name: str = "Athlete") -> go.Figure:
    """
    Create combined Force-Velocity-Power dashboard
    3 plots in one view
    """

    if profile is None:
        return None

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Force-Velocity Profile',
            'Power-Force Profile',
            'Power-Velocity Profile',
            'Profile Summary'
        ),
        specs=[
            [{'type': 'scatter'}, {'type': 'scatter'}],
            [{'type': 'scatter'}, {'type': 'table'}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # 1. F-V Profile
    v_theoretical = np.linspace(0, profile['V0'], 100)
    f_theoretical = profile['F0'] + profile['slope'] * v_theoretical

    fig.add_trace(go.Scatter(
        x=profile['velocities'], y=profile['forces'],
        mode='markers', name='Actual',
        marker=dict(size=10, color='#1D4D3B')
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=v_theoretical, y=f_theoretical,
        mode='lines', name='F-V Line',
        line=dict(color='#a08e66', width=2)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=[profile['V_opt']], y=[profile['F_opt']],
        mode='markers', name='Pmax',
        marker=dict(size=14, color='red', symbol='star')
    ), row=1, col=1)

    # 2. Power-Force
    forces = np.linspace(0, profile['F0'], 100)
    velocities_calc = (profile['F0'] - forces) / abs(profile['slope'])
    powers = forces * velocities_calc

    fig.add_trace(go.Scatter(
        x=forces, y=powers,
        mode='lines', name='P-F',
        line=dict(color='#1D4D3B', width=2),
        fill='tozeroy', fillcolor='rgba(0, 113, 103, 0.1)'
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=[profile['F_opt']], y=[profile['Pmax']],
        mode='markers', name='Optimal',
        marker=dict(size=14, color='red', symbol='star'),
        showlegend=False
    ), row=1, col=2)

    # 3. Power-Velocity
    velocities = np.linspace(0, profile['V0'], 100)
    forces_calc = profile['F0'] + profile['slope'] * velocities
    powers = forces_calc * velocities

    fig.add_trace(go.Scatter(
        x=velocities, y=powers,
        mode='lines', name='P-V',
        line=dict(color='#1D4D3B', width=2),
        fill='tozeroy', fillcolor='rgba(0, 113, 103, 0.1)'
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=[profile['V_opt']], y=[profile['Pmax']],
        mode='markers', name='Optimal',
        marker=dict(size=14, color='red', symbol='star'),
        showlegend=False
    ), row=2, col=1)

    # 4. Summary Table
    imbalance_interpretation = "Force Dominant" if profile['FV_imbalance'] > 10 else "Velocity Dominant" if profile['FV_imbalance'] < -10 else "Balanced"

    fig.add_trace(go.Table(
        header=dict(
            values=['<b>Metric</b>', '<b>Value</b>', '<b>Interpretation</b>'],
            fill_color='#1D4D3B',
            align='left',
            font=dict(color='white', size=12, family='Arial Black')
        ),
        cells=dict(
            values=[
                ['F0 (Max Force)', 'V0 (Max Velocity)', 'Pmax (Max Power)', 'F-V Imbalance', 'R²'],
                [f'{profile["F0"]:.0f} N', f'{profile["V0"]:.2f} m/s', f'{profile["Pmax"]:.0f} W',
                 f'{abs(profile["FV_imbalance"]):.1f}%', f'{profile["r_squared"]:.3f}'],
                ['Theoretical max', 'Theoretical max', f'@ {profile["F_opt"]:.0f}N, {profile["V_opt"]:.2f}m/s',
                 imbalance_interpretation, 'Model fit']
            ],
            fill_color=['white', '#f0f2f5', 'white'],
            align='left',
            font=dict(size=11)
        )
    ), row=2, col=2)

    # Update axes
    fig.update_xaxes(title_text="Velocity (m/s)", row=1, col=1)
    fig.update_yaxes(title_text="Force (N)", row=1, col=1)

    fig.update_xaxes(title_text="Force (N)", row=1, col=2)
    fig.update_yaxes(title_text="Power (W)", row=1, col=2)

    fig.update_xaxes(title_text="Velocity (m/s)", row=2, col=1)
    fig.update_yaxes(title_text="Power (W)", row=2, col=1)

    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Complete Force-Velocity-Power Profile - {athlete_name}',
            font=dict(size=22, family='Arial Black', color='#1D4D3B'),
            x=0.5,
            xanchor='center'
        ),
        height=900,
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='#f8f9fa'
    )

    return fig


def interpret_fv_profile(profile: Dict) -> str:
    """Generate text interpretation of F-V profile"""

    if profile is None:
        return "Insufficient data for F-V profile analysis"

    interpretation = []

    # Overall profile quality
    if profile['r_squared'] > 0.8:
        interpretation.append("✅ **High-quality F-V profile** (R² > 0.8)")
    elif profile['r_squared'] > 0.6:
        interpretation.append("⚠️ **Moderate F-V profile quality** (R² 0.6-0.8)")
    else:
        interpretation.append("❌ **Low F-V profile quality** (R² < 0.6) - more data needed")

    # F-V Imbalance
    imbalance = profile['FV_imbalance']
    if abs(imbalance) < 10:
        interpretation.append(f"✅ **Well-balanced** F-V profile ({abs(imbalance):.1f}% imbalance)")
    elif abs(imbalance) < 20:
        dominant = "Force" if imbalance > 0 else "Velocity"
        interpretation.append(f"⚠️ **{dominant}-dominant** profile ({abs(imbalance):.1f}% imbalance)")

        if imbalance > 0:
            interpretation.append("   → *Recommendation:* Focus on speed/plyometric training")
        else:
            interpretation.append("   → *Recommendation:* Focus on strength/resistance training")
    else:
        dominant = "Force" if imbalance > 0 else "Velocity"
        interpretation.append(f"🔴 **Highly {dominant}-dominant** ({abs(imbalance):.1f}% imbalance)")
        interpretation.append("   → *Priority:* Address imbalance to optimize power production")

    # Maximal power
    pmax_relative = profile['Pmax'] / 1000  # Convert to kW
    interpretation.append(f"\n**Max Power:** {profile['Pmax']:.0f} W ({pmax_relative:.2f} kW)")
    interpretation.append(f"**Optimal Zone:** {profile['F_opt']:.0f} N @ {profile['V_opt']:.2f} m/s")

    return "\n".join(interpretation)


if __name__ == "__main__":
    # Test with sample data
    print("Force-Velocity-Power Profile Module loaded successfully!")
    print("\nExample usage:")
    print("  profile = calculate_fv_profile(jump_data)")
    print("  fig = create_fv_profile_plot(profile, 'Athlete Name')")
    print("  dashboard = create_combined_fvp_dashboard(profile, 'Athlete Name')")
