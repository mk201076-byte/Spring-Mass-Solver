"""
=============================================================================
  Single DOF Spring-Mass-Damper System — Interactive Solver
=============================================================================

  Equation of Motion:
      m * x''(t) + c * x'(t) + k * x(t) = F0 * sin(omega * t)

  Where:
      m      = mass (kg)
      c      = damping coefficient (N·s/m)
      k      = stiffness (N/m)
      x(t)   = displacement (m)
      F0     = forcing amplitude (N)
      omega  = forcing frequency (rad/s)

  Converted to State-Space form for ODE solvers:
      Let  y[0] = x   (displacement)
           y[1] = v   (velocity = x')

      Then:
           y[0]' = y[1]
           y[1]' = (F0*sin(omega*t) - c*y[1] - k*y[0]) / m

  Solvers available (all from scipy.integrate.solve_ivp):
      RK45   — Runge-Kutta 4/5 order (default, general purpose)
      RK23   — Runge-Kutta 2/3 order (simpler, less accurate)
      Radau  — Implicit Runge-Kutta, good for stiff problems
      BDF    — Backward Differentiation Formula, good for stiff problems
      LSODA  — Auto-switches between stiff and non-stiff methods

=============================================================================
  HOW TO RUN:
      1. Install dependencies:  pip install -r requirements.txt
      2. Run the script:        python spring_mass_solver.py
      3. Open your browser at:  http://127.0.0.1:8050
=============================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────────
import numpy as np

# ── Scientific computing ───────────────────────────────────────────────────────
from scipy.integrate import solve_ivp

# ── Web-based interactive dashboard ───────────────────────────────────────────
import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =============================================================================
#   PHYSICS — Equation of Motion
# =============================================================================

def equations_of_motion(t, y, m, c, k, F0, omega):
    """
    Defines the ODE system for a damped, forced spring-mass system.

    Parameters
    ----------
    t     : float       — current time (s)
    y     : list[float] — state vector [displacement x, velocity v]
    m     : float       — mass (kg)
    c     : float       — damping coefficient (N·s/m)
    k     : float       — stiffness (N/m)
    F0    : float       — forcing amplitude (N)
    omega : float       — forcing frequency (rad/s)

    Returns
    -------
    [dx/dt, dv/dt] : list[float]
    """
    x, v = y                                  # unpack state variables
    dxdt = v                                  # velocity IS the rate of displacement
    dvdt = (F0 * np.sin(omega * t) - c * v - k * x) / m   # Newton's 2nd Law
    return [dxdt, dvdt]


def run_solver(m, c, k, F0, omega, x0, v0, t_end, solver_name):
    """
    Runs the selected ODE solver and returns time + solution arrays.

    Parameters
    ----------
    m, c, k       : floats — system parameters
    F0, omega     : floats — forcing parameters
    x0, v0        : floats — initial displacement and velocity
    t_end         : float  — simulation end time (s)
    solver_name   : str    — one of 'RK45', 'RK23', 'Radau', 'BDF', 'LSODA'

    Returns
    -------
    t  : np.ndarray — time points
    x  : np.ndarray — displacement at each time point
    v  : np.ndarray — velocity at each time point
    nfev : int      — number of function evaluations (solver effort indicator)
    """
    t_span = (0, t_end)                        # simulation time window
    t_eval = np.linspace(0, t_end, 2000)       # 2000 output points for smooth plots

    sol = solve_ivp(
        fun      = equations_of_motion,        # our ODE function
        t_span   = t_span,
        y0       = [x0, v0],                   # initial conditions
        method   = solver_name,                # chosen solver
        t_eval   = t_eval,
        args     = (m, c, k, F0, omega),       # extra args passed to the ODE
        rtol     = 1e-8,                       # relative tolerance
        atol     = 1e-10,                      # absolute tolerance
        dense_output = False
    )

    return sol.t, sol.y[0], sol.y[1], sol.nfev


# =============================================================================
#   DERIVED QUANTITIES — useful system properties
# =============================================================================

def compute_system_properties(m, c, k):
    """Compute natural frequency, damping ratio, and damped frequency."""
    omega_n   = np.sqrt(k / m)                # natural frequency (rad/s)
    cc        = 2 * np.sqrt(m * k)            # critical damping coefficient
    zeta      = c / cc                        # damping ratio (dimensionless)
    omega_d   = omega_n * np.sqrt(max(1 - zeta**2, 0))  # damped natural frequency
    return omega_n, zeta, omega_d


# =============================================================================
#   COLOR SCHEME — one distinct color per solver
# =============================================================================

SOLVER_COLORS = {
    "RK45"  : "#00C8FF",   # electric blue
    "RK23"  : "#FF6B6B",   # coral red
    "Radau" : "#A8FF78",   # lime green
    "BDF"   : "#FFD93D",   # amber yellow
    "LSODA" : "#C77DFF",   # violet
}

ALL_SOLVERS = list(SOLVER_COLORS.keys())


# =============================================================================
#   DASH APPLICATION — Layout
# =============================================================================

app = dash.Dash(__name__, title="Spring-Mass Solver")

# ── Reusable slider builder ────────────────────────────────────────────────────
def make_slider(slider_id, label, min_val, max_val, step, default, marks=None):
    return html.Div([
        html.Label(label, style={"color": "#aaaacc", "fontSize": "13px",
                                 "marginBottom": "4px", "display": "block"}),
        dcc.Slider(
            id      = slider_id,
            min     = min_val,
            max     = max_val,
            step    = step,
            value   = default,
            marks   = marks or {min_val: str(min_val), max_val: str(max_val)},
            tooltip = {"placement": "bottom", "always_visible": True},
            updatemode = "drag",
        ),
    ], style={"marginBottom": "20px"})


# ── Main layout ────────────────────────────────────────────────────────────────
app.layout = html.Div(style={
    "backgroundColor": "#0e0e1a",
    "minHeight": "100vh",
    "fontFamily": "'Segoe UI', sans-serif",
    "color": "#e0e0f0",
    "padding": "20px",
}, children=[

    # ── Header ──────────────────────────────────────────────────────────────
    html.Div([
        html.H1("⚙  Single DOF Spring-Mass-Damper Solver",
                style={"margin": "0", "fontSize": "24px", "color": "#ffffff"}),
        html.P("m·x″ + c·x′ + k·x = F₀·sin(ω·t)  —  Comparing ODE Solvers",
               style={"margin": "6px 0 0 0", "color": "#888aaa", "fontSize": "14px"}),
    ], style={"borderBottom": "1px solid #2a2a4a", "paddingBottom": "16px",
              "marginBottom": "20px"}),

    # ── Main content row ─────────────────────────────────────────────────────
    html.Div(style={"display": "flex", "gap": "24px", "flexWrap": "wrap"}, children=[

        # ── LEFT: Control Panel ──────────────────────────────────────────────
        html.Div(style={
            "width": "280px", "flexShrink": "0",
            "backgroundColor": "#161626",
            "borderRadius": "12px",
            "padding": "20px",
            "border": "1px solid #2a2a4a",
        }, children=[

            html.H3("System Parameters", style={"margin": "0 0 16px 0",
                    "color": "#aaaaff", "fontSize": "15px"}),

            make_slider("mass",    "Mass  m  (kg)",          0.1, 10.0, 0.1,  1.0,
                        {0.1:"0.1", 5:"5", 10:"10"}),
            make_slider("stiff",   "Stiffness  k  (N/m)",    1.0, 200.0, 1.0, 100.0,
                        {1:"1", 100:"100", 200:"200"}),
            make_slider("damp",    "Damping  c  (N·s/m)",    0.0, 20.0, 0.1,  2.0,
                        {0:"0", 10:"10", 20:"20"}),

            html.Hr(style={"borderColor": "#2a2a4a"}),
            html.H3("Forcing", style={"margin": "12px 0 16px 0",
                    "color": "#aaaaff", "fontSize": "15px"}),

            make_slider("force",   "Amplitude  F₀  (N)",     0.0, 50.0, 0.5,  10.0,
                        {0:"0", 25:"25", 50:"50"}),
            make_slider("freq",    "Frequency  ω  (rad/s)",  0.0, 30.0, 0.5,  10.0,
                        {0:"0", 10:"10", 30:"30"}),

            html.Hr(style={"borderColor": "#2a2a4a"}),
            html.H3("Initial Conditions", style={"margin": "12px 0 16px 0",
                    "color": "#aaaaff", "fontSize": "15px"}),

            make_slider("x0",      "x₀  — Initial Displacement (m)", -2.0, 2.0, 0.1, 0.0,
                        {-2:"-2", 0:"0", 2:"2"}),
            make_slider("v0",      "v₀  — Initial Velocity (m/s)",   -5.0, 5.0, 0.1, 0.0,
                        {-5:"-5", 0:"0", 5:"5"}),
            make_slider("t_end",   "Simulation Duration  T  (s)",    5.0, 60.0, 1.0, 20.0,
                        {5:"5", 30:"30", 60:"60"}),

            html.Hr(style={"borderColor": "#2a2a4a"}),
            html.H3("Select Solvers", style={"margin": "12px 0 10px 0",
                    "color": "#aaaaff", "fontSize": "15px"}),

            dcc.Checklist(
                id      = "solvers",
                options = [{"label": f"  {s}", "value": s} for s in ALL_SOLVERS],
                value   = ["RK45", "Radau"],          # default selection
                style   = {"color": "#ccccee", "lineHeight": "2"},
                inputStyle = {"marginRight": "8px"},
            ),
        ]),

        # ── RIGHT: Plots + Info ──────────────────────────────────────────────
        html.Div(style={"flex": "1", "minWidth": "400px"}, children=[

            # System properties info bar
            html.Div(id="sys-info", style={
                "backgroundColor": "#161626", "borderRadius": "10px",
                "padding": "12px 20px", "marginBottom": "16px",
                "border": "1px solid #2a2a4a", "fontSize": "13px",
                "display": "flex", "gap": "30px", "flexWrap": "wrap",
            }),

            # Main plots
            dcc.Graph(id="main-plot", style={"height": "640px"},
                      config={"displayModeBar": True}),

            # Solver performance table
            html.Div(id="solver-stats", style={
                "backgroundColor": "#161626", "borderRadius": "10px",
                "padding": "16px 20px", "marginTop": "16px",
                "border": "1px solid #2a2a4a",
            }),
        ]),
    ]),
])


# =============================================================================
#   DASH CALLBACK — Recompute and redraw when any input changes
# =============================================================================

@callback(
    Output("main-plot",   "figure"),
    Output("sys-info",    "children"),
    Output("solver-stats","children"),
    Input("mass",    "value"),
    Input("stiff",   "value"),
    Input("damp",    "value"),
    Input("force",   "value"),
    Input("freq",    "value"),
    Input("x0",      "value"),
    Input("v0",      "value"),
    Input("t_end",   "value"),
    Input("solvers", "value"),
)
def update_plots(m, k, c, F0, omega, x0, v0, t_end, selected_solvers):
    """This function runs every time the user moves a slider or toggles a solver."""

    # ── Guard: ensure we have at least one solver selected ──────────────────
    if not selected_solvers:
        selected_solvers = ["RK45"]

    # ── Compute system properties ────────────────────────────────────────────
    omega_n, zeta, omega_d = compute_system_properties(m, c, k)

    # ── Build the figure: 3 subplots ─────────────────────────────────────────
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            "Displacement  x(t)  vs  Time",
            "Velocity  ẋ(t)  vs  Time",
            "Phase Portrait  [ x  vs  ẋ ]",
        ),
        vertical_spacing=0.10,
        shared_xaxes=False,
    )

    stats_data = []   # collect solver performance info

    for solver_name in selected_solvers:
        color = SOLVER_COLORS[solver_name]

        # ── Solve the ODE ────────────────────────────────────────────────────
        t, x, v, nfev = run_solver(m, c, k, F0, omega, x0, v0, t_end, solver_name)

        line_style = dict(color=color, width=1.8)

        # ── Subplot 1: Displacement ──────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=t, y=x, mode="lines",
            name=solver_name, legendgroup=solver_name,
            line=line_style, showlegend=True,
        ), row=1, col=1)

        # ── Subplot 2: Velocity ──────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=t, y=v, mode="lines",
            name=solver_name, legendgroup=solver_name,
            line=line_style, showlegend=False,
        ), row=2, col=1)

        # ── Subplot 3: Phase Portrait ────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=x, y=v, mode="lines",
            name=solver_name, legendgroup=solver_name,
            line=line_style, showlegend=False,
        ), row=3, col=1)

        stats_data.append({
            "solver": solver_name,
            "nfev"  : nfev,
            "x_max" : round(float(np.max(np.abs(x))), 5),
            "v_max" : round(float(np.max(np.abs(v))), 5),
            "color" : color,
        })

    # ── Style the figure ─────────────────────────────────────────────────────
    fig.update_layout(
        paper_bgcolor = "#0e0e1a",
        plot_bgcolor  = "#12121f",
        font          = dict(color="#ccccee", size=12),
        legend        = dict(bgcolor="#161626", bordercolor="#2a2a4a",
                             borderwidth=1, font=dict(size=12)),
        margin        = dict(l=60, r=20, t=40, b=30),
    )
    # Axis labels
    fig.update_xaxes(title_text="Time (s)",         row=1, col=1,
                     gridcolor="#1e1e35", zerolinecolor="#333355")
    fig.update_xaxes(title_text="Time (s)",         row=2, col=1,
                     gridcolor="#1e1e35", zerolinecolor="#333355")
    fig.update_xaxes(title_text="Displacement (m)", row=3, col=1,
                     gridcolor="#1e1e35", zerolinecolor="#333355")
    fig.update_yaxes(title_text="x (m)",            row=1, col=1,
                     gridcolor="#1e1e35", zerolinecolor="#333355")
    fig.update_yaxes(title_text="ẋ (m/s)",          row=2, col=1,
                     gridcolor="#1e1e35", zerolinecolor="#333355")
    fig.update_yaxes(title_text="ẋ (m/s)",          row=3, col=1,
                     gridcolor="#1e1e35", zerolinecolor="#333355")

    # ── System info bar ───────────────────────────────────────────────────────
    def info_item(label, value, unit=""):
        return html.Span([
            html.Span(label + ": ", style={"color": "#666688"}),
            html.Span(f"{value:.4f} {unit}", style={"color": "#ffffff", "fontWeight": "600"}),
        ])

    regime = ("Under-damped"  if zeta < 1 else
              "Critically damped" if abs(zeta - 1) < 0.01 else "Over-damped")
    regime_color = "#A8FF78" if zeta < 1 else ("#FFD93D" if abs(zeta - 1) < 0.01 else "#FF6B6B")

    sys_info_children = [
        info_item("ωₙ (natural freq)", omega_n, "rad/s"),
        info_item("ζ (damping ratio)", zeta, ""),
        info_item("ωd (damped freq)",  omega_d, "rad/s"),
        html.Span([
            html.Span("Regime: ", style={"color": "#666688"}),
            html.Span(regime, style={"color": regime_color, "fontWeight": "700"}),
        ]),
    ]

    # ── Solver stats table ────────────────────────────────────────────────────
    def stat_row(d):
        return html.Div(style={
            "display": "flex", "gap": "30px", "padding": "6px 0",
            "borderBottom": "1px solid #1e1e35", "alignItems": "center",
        }, children=[
            html.Span("●", style={"color": d["color"], "fontSize": "18px"}),
            html.Span(d["solver"],          style={"width": "70px", "fontWeight": "700"}),
            html.Span(f"Evaluations: {d['nfev']}",  style={"color": "#aaaacc", "width": "160px"}),
            html.Span(f"|x|max: {d['x_max']} m",    style={"color": "#aaaacc", "width": "160px"}),
            html.Span(f"|ẋ|max: {d['v_max']} m/s",  style={"color": "#aaaacc"}),
        ])

    stats_children = [
        html.H4("Solver Performance", style={"margin": "0 0 10px 0",
                "color": "#aaaaff", "fontSize": "14px"}),
        html.P("'Evaluations' = how many times the solver computed the ODE — lower means more efficient.",
               style={"color": "#666688", "fontSize": "12px", "margin": "0 0 10px 0"}),
        *[stat_row(d) for d in stats_data],
    ]

    return fig, sys_info_children, stats_children


# =============================================================================
#   ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("  Spring-Mass-Damper Solver  —  Starting...")
    print("="*60)
    print("  Open your browser at:  http://127.0.0.1:8050")
    print("  Press  Ctrl+C  to stop the server.")
    print("="*60)

    app.run(
        debug = False,   # set True during development to see error messages
        host  = "0.0.0.0",   # 0.0.0.0 is needed for GitHub Codespaces
        port  = 8050,
    )
