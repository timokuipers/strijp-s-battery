import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.interpolate import interp1d

# ====================================================
# 1. DATA & CONFIGURATION (Based on Strijp-S Report)
# ====================================================

# Data from Report Table 3 (Page 17)
# Represents average daily values per season for the Strijp-S district
SEASON_PARAMS = {
    "Winter": {"demand_kwh": 16674, "pv_kwh": 195.8},
    "Spring": {"demand_kwh": 11132, "pv_kwh": 811.7},
    "Summer": {"demand_kwh": 8510,  "pv_kwh": 943.4}, 
    "Autumn": {"demand_kwh": 13210, "pv_kwh": 414.5},
}

# Standardized Dutch Household Load Profile (Normalized)
# Source: Klaassen et al., 2015 (Report Page 11, Fig 4)
BASE_LOAD_PROFILE_X = np.array([0, 2, 4, 6, 8, 10, 12, 14, 15, 16, 17, 18, 20, 22, 24])
BASE_LOAD_PROFILE_Y = np.array([0.5, 0.35, 0.30, 0.35, 0.60, 0.70, 0.725, 0.70,
                                0.65, 0.675, 0.75, 0.90, 1.00, 0.80, 0.50])

# Generate Time Axis (5-minute resolution = 288 steps)
x = np.linspace(0, 24, 288) 
dt = x[1] - x[0]

# Interpolate Base Shapes to 5-min resolution
interp_func = interp1d(BASE_LOAD_PROFILE_X, BASE_LOAD_PROFILE_Y, kind='linear')
base_shape = interp_func(x)
E_shape = np.trapz(base_shape, x) # Integral for normalization

def solar_curve(t):
    """
    Generates a normalized solar curve (Sinusoidal approximation).
    Active between 06:00 and 18:00.
    """
    y = np.zeros_like(t)
    mask = (t > 6) & (t < 18)
    y[mask] = np.sin(np.pi * (t[mask] - 6) / 12)
    return y

def calc_stats(p, t):
    """Calculates Energy (kWh), Peak (kW), and Load Factor."""
    energy = np.trapz(p, t)
    peak = p.max()
    avg = energy / 24.0
    # Avoid division by zero
    lf = avg / peak if peak > 1e-6 else 0
    return energy, peak, lf

# ====================================================
# 2. SOLVER LOGIC (Binary Search + Water Filling)
# ====================================================

def check_feasibility(level, y_net, dt, x, cap, soc_start, power, chg_start, chg_end, eff):
    """
    Binary Search Helper:
    Simulates the day with 'Greedy' logic to see if a specific 'Grid Target Level' 
    is mathematically possible without the battery hitting 0% charge during discharge.
    """
    soc = cap * soc_start
    
    for i in range(len(x)):
        load = y_net[i]
        time_h = x[i]
        
        # 1. DISCHARGE NEEDED? (Load > Target)
        if load > level:
            discharge_needed = load - level
            discharge_act = min(discharge_needed, power) 
            energy_out = discharge_act * dt
            
            if soc < energy_out:
                return False # FAILED: Battery died, target is too low
            soc -= energy_out
            
        # 2. CHARGE OPPORTUNITY? (Load < Target)
        elif load < level:
            if chg_start <= time_h < chg_end:
                headroom = level - load
                charge_act = min(headroom, power)
                energy_in = charge_act * dt * eff
                soc = min(soc + energy_in, cap)
                
    return True

# ====================================================
# 3. DASH APPLICATION LAYOUT
# ====================================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX], title="Strijp-S BESS Sim")
server = app.server

card_style = {"boxShadow": "0 4px 8px 0 rgba(0,0,0,0.1)", "border": "none", "marginBottom": "20px"}

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("🔋 Strijp-S: Community Battery Simulator", className="text-center mt-4 mb-2"),
            html.P("Interactive grid congestion model based on the 'Local Battery Implementation in Strijp-S' engineering report.", 
                   className="text-center text-muted mb-4"),
            html.Hr()
        ])
    ]),
    dbc.Row([
        # --- LEFT CONTROLS ---
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("BESS Configuration (Final Design)", className="bg-primary text-white"),
                dbc.CardBody([
                    html.Label("Battery Energy (kWh)", className="fw-bold"),
                    # Default 1200 based on Report Page 20
                    dbc.Input(id='input-cap', type='number', value=1200, step=50, min=0, className="mb-3"),
                    
                    html.Label("Inverter Power (kW)", className="fw-bold"),
                    # Default 500 based on Report Page 20
                    dbc.Input(id='input-power', type='number', value=500, step=10, min=0, className="mb-3"),
                    
                    html.Label("Round-Trip Efficiency", className="fw-bold"),
                    # Default 0.9 (90%) based on Report Page 17
                    dcc.Slider(id='slider-eff', min=0.7, max=1.0, step=0.01, value=0.9, 
                               marks={0.8:'80%', 0.9:'90%', 1.0:'100%'}, className="mb-3"),
                    
                    html.Label("Starting SoC (%)", className="fw-bold"),
                    dcc.Slider(id='slider-soc', min=0.0, max=1.0, step=0.1, value=1.0, 
                               marks={0:'0%', 0.5:'50%', 1.0:'100%'}, className="mb-3"),
                    
                    html.Hr(),
                    html.Label("Solar Charging Bias", className="fw-bold"),
                    # Default 0.75 based on Report Page 17
                    dcc.Slider(id='slider-bias', min=0.0, max=2.0, step=0.25, value=0.75, 
                               marks={0:'Grid', 0.75:'Mixed', 2:'Solar'}, className="mb-3"),
                    
                    html.Label("Allowed Charge Window", className="fw-bold"),
                    # Default 00:00 - 18:00 based on Report Page 17
                    dcc.RangeSlider(id='slider-window', min=0, max=24, step=1, value=[0, 18], 
                                    marks={0:'00:00', 12:'12:00', 18:'18:00', 24:'24:00'})
                ])
            ], style=card_style),
            
            # Legend Card
            dbc.Card([
                dbc.CardBody([
                    html.H6("Legend", className="card-title"),
                    html.Div([html.Span("■ ", style={'color': '#f39c12'}), "Solar Generation"]),
                    html.Div([html.Span("— ", style={'color': '#5dade2', 'fontWeight':'bold'}), "Optimized Grid Profile"]),
                    html.Div([html.Span("--- ", style={'color': 'gray'}), "Original Net Load"]),
                    html.Div([html.Span("■ ", style={'color': '#2ecc71'}), "Discharging (Peak Shaving)"]),
                    html.Div([html.Span("■ ", style={'color': '#e74c3c'}), "Charging"]),
                ])
            ], style=card_style)
        ], md=3),

        # --- RIGHT RESULTS ---
        dbc.Col([
            dbc.Row([
                dbc.Col(dbc.Card([dbc.CardBody([
                    html.H4(id='kpi-red', className="text-success text-center"), 
                    html.P("Avg Peak Reduction", className="text-muted text-center small mb-0")
                ])], style=card_style), width=4),
                
                dbc.Col(dbc.Card([dbc.CardBody([
                    html.H4(id='kpi-lf', className="text-primary text-center"), 
                    html.P("Avg Load Factor Gain", className="text-muted text-center small mb-0")
                ])], style=card_style), width=4),
                
                dbc.Col(dbc.Card([dbc.CardBody([
                    html.H4(id='kpi-season', className="text-info text-center"), 
                    html.P("Best Performing Season", className="text-muted text-center small mb-0")
                ])], style=card_style), width=4),
            ], className="mb-2"),
            
            dbc.Card([dbc.CardBody([
                dcc.Graph(id='main-graph', style={'height': '600px'}, config={'displayModeBar': False})
            ])], style=card_style),
            
            dbc.Card([
                dbc.CardHeader("Detailed Season Metrics (Matches Report Table 4)"),
                dash_table.DataTable(
                    id='results-table',
                    columns=[
                        {"name": "Season", "id": "Season"},
                        {"name": "Orig Peak", "id": "Orig Peak"},
                        {"name": "New Peak", "id": "New Peak"},
                        {"name": "Reduction", "id": "Red"},
                        {"name": "Orig Import (kWh)", "id": "Orig kWh"},
                        {"name": "New Import (kWh)", "id": "New kWh"},
                        {"name": "LF Gain", "id": "LF Gain"}
                    ],
                    style_as_list_view=True,
                    style_cell={'padding': '10px', 'textAlign': 'center', 'fontFamily': 'sans-serif'},
                    style_header={'backgroundColor': 'white','fontWeight': 'bold','borderBottom': '2px solid #eee'}
                )
            ], style=card_style)
        ], md=9)
    ])
], fluid=True, style={'backgroundColor': '#f4f6f8', 'minHeight': '100vh', 'paddingBottom': '50px'})


# ====================================================
# 4. CALLBACKS & CALCULATION ENGINE
# ====================================================
@app.callback(
    [Output('main-graph', 'figure'), 
     Output('results-table', 'data'), 
     Output('kpi-red', 'children'), 
     Output('kpi-lf', 'children'), 
     Output('kpi-season', 'children')],
    [Input('input-cap', 'value'), 
     Input('input-power', 'value'), 
     Input('slider-eff', 'value'), 
     Input('slider-soc', 'value'), 
     Input('slider-bias', 'value'), 
     Input('slider-window', 'value')]
)
def update_dashboard(cap, power, eff, soc_start, bias, window):
    # Safety Validation for Inputs
    if cap is None: cap = 0
    if power is None: power = 0
    if soc_start is None: soc_start = 1.0
    chg_start, chg_end = window

    # Prepare Plots
    fig = make_subplots(rows=2, cols=2, subplot_titles=list(SEASON_PARAMS.keys()), vertical_spacing=0.12)
    table_data, lf_gains, red_pcts = [], [], []
    row_col_map = [(1,1), (1,2), (2,1), (2,2)]
    
    # --- SIMULATION LOOP PER SEASON ---
    for idx, (season, vals) in enumerate(SEASON_PARAMS.items()):
        r, c = row_col_map[idx]
        
        # 1. GENERATE PROFILES
        # Scale base shape to Report's daily demand (kWh)
        y_res = base_shape * (vals["demand_kwh"] / E_shape)
        
        # Scale solar to Report's daily PV generation (kWh)
        pv_norm = solar_curve(x)
        E_pv_raw = np.trapz(pv_norm, x)
        y_pv = pv_norm * (vals["pv_kwh"] / E_pv_raw) if E_pv_raw > 0 else np.zeros_like(x)
        
        # Net Load = Demand - Solar
        y_net = y_res - y_pv 
        
        # 2. SOLVER: BINARY SEARCH
        # Find lowest feasible grid target (peak) that doesn't drain battery
        lo, hi = 0, max(0, y_net.max())
        target = hi
        # 15 iterations gives <0.1% error, sufficient for visual
        for _ in range(15): 
            mid = (lo + hi) / 2
            if check_feasibility(mid, y_net, dt, x, cap, soc_start, power, chg_start, chg_end, eff):
                target = mid; hi = mid 
            else: lo = mid 
        
        # 3. OPTIMIZATION: WATER FILLING (SMOOTH CHARGING)
        # Determine available power headroom below the target
        mask_window = (x >= chg_start) & (x < chg_end)
        headroom = np.maximum(0, target - y_net) * mask_window 
        p_avail = np.minimum(power, headroom) 
        
        # Calculate total energy needed to support the peak shaving
        discharge_needed = np.trapz(np.maximum(y_net - target, 0), dx=dt)
        # Add efficiency losses to requirement
        energy_deficit = (discharge_needed / eff) - (cap * soc_start)
        energy_deficit = max(0, energy_deficit)
        
        # Solar Bias Logic (Bias charging towards noon)
        pv_max = max(1, y_pv.max())
        bias_curve = 1 + (bias * y_pv / pv_max)
        
        y_fill = np.zeros_like(x)
        rem = energy_deficit
        
        # Iteratively fill the "charging bucket"
        for _ in range(15):
            if rem <= 0.01: break
            curr_h = p_avail - y_fill
            denom = np.trapz(bias_curve * (curr_h > 0.001), x)
            if denom < 1e-6: break
            scaler = rem / denom
            y_fill += np.minimum(bias_curve * scaler, curr_h)
            rem = energy_deficit - np.trapz(y_fill, x)
            
        # 4. FINAL PROFILE CALCULATION
        y_after = y_net.copy()
        soc = cap * soc_start
        
        for i in range(len(x)):
            # Charge Step
            p_charge = y_fill[i]
            e_in = p_charge * dt * eff
            if soc + e_in > cap:
               e_in = max(0, cap - soc)
               p_charge = e_in / (dt * eff)
            soc += e_in
            
            # Discharge Step
            p_req = max(0, y_net[i] - target)
            p_discharge = min(p_req, power)
            e_out = p_discharge * dt
            if soc >= e_out: soc -= e_out
            else: e_out = soc; p_discharge = e_out / dt; soc = 0
            
            # Update Grid Profile
            y_after[i] = y_after[i] + p_charge - p_discharge

        # 5. PLOTTING
        # Solar (Background)
        fig.add_trace(go.Scatter(x=x, y=y_pv, mode='lines', line=dict(color='rgba(243, 156, 18, 0.8)', width=1), fill='tozeroy', fillcolor='rgba(243, 156, 18, 0.1)', name="Solar PV", showlegend=(idx==0), hoverinfo='skip'), row=r, col=c)
        # Raw Demand (Faint)
        fig.add_trace(go.Scatter(x=x, y=y_res, mode='lines', line=dict(color='rgba(41, 128, 185, 0.3)', width=1), name="Raw Demand", showlegend=(idx==0), hoverinfo='skip'), row=r, col=c)
        # Net Load (Dashed)
        fig.add_trace(go.Scatter(x=x, y=y_net, mode='lines', line=dict(color='gray', dash='dash', width=1), name="Net Load", showlegend=(idx==0), hoverinfo='skip'), row=r, col=c)
        # Final Grid Profile (Blue)
        fig.add_trace(go.Scatter(x=x, y=y_after, mode='lines', line=dict(color='#5dade2', width=2.5), name="Final Grid Profile", showlegend=(idx==0)), row=r, col=c)
        
        # Fill Areas (Charge = Red, Discharge = Green)
        y_red_top = np.maximum(y_net, y_after)
        fig.add_trace(go.Scatter(x=x, y=y_red_top, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(231, 76, 60, 0.4)', name="Charge", showlegend=(idx==0), hoverinfo='skip'), row=r, col=c)
        y_green_top = np.maximum(y_net, y_after) 
        fig.add_trace(go.Scatter(x=x, y=y_green_top, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(46, 204, 113, 0.4)', name="Discharge", showlegend=(idx==0), hoverinfo='skip'), row=r, col=c)

        # 6. CALCULATE KPIs
        _, pkb, lfb = calc_stats(y_net, x)
        _, pka, lfa = calc_stats(y_after, x)
        
        import_orig = np.trapz(np.maximum(y_net, 0), x)
        import_new = np.trapz(np.maximum(y_after, 0), x)
        
        # Calculate % changes (Safety checks for div by zero)
        p_red_pct = (1 - (pka/pkb)) * 100 if pkb > 1 else 0.0
        lf_gain_pct = ((lfa - lfb)/lfb) * 100 if lfb > 0.01 else 0.0
        
        table_data.append({
            "Season": season, 
            "Orig Peak": f"{pkb:.0f} kW", 
            "New Peak": f"{pka:.0f} kW", 
            "Red": f"{p_red_pct:.1f}%", 
            "Orig kWh": f"{import_orig:.0f}",
            "New kWh": f"{import_new:.0f}",
            "LF Gain": f"+{lf_gain_pct:.1f}%"
        })
        
        lf_gains.append(lf_gain_pct)
        red_pcts.append(p_red_pct)

    # Final Layout Polish
    fig.update_layout(template="plotly_white", margin=dict(t=40, b=20, l=40, r=40))
    fig.update_yaxes(showgrid=True, gridcolor='#eee', title="Power (kW)")
    
    # Aggregate KPIs
    best_season = table_data[np.argmax(lf_gains)]['Season'] if lf_gains else "N/A"
    avg_red = np.mean(red_pcts) if red_pcts else 0
    avg_lf = np.mean(lf_gains) if lf_gains else 0

    return fig, table_data, f"{avg_red:.1f}%", f"+{avg_lf:.1f}%", best_season

if __name__ == '__main__':
    app.run_server(debug=True)