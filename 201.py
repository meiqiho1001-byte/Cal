import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
import re

# ---------------------------
# PAGE CONFIG & STYLE
# ---------------------------
st.set_page_config(page_title="Calculus Intelligence Pro", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
.stApp {
    background-color:#0d1117;
    color:#c9d1d9;
    font-family:'Inter',times new roman;
}
.topic-card {
    background:#161b22;
    padding:25px;
    border-radius:12px;
    margin-bottom:25px;
    border:1px solid #30363d;
    box-shadow:0 4px 10px rgba(0,0,0,0.2);
}
.topic-header {
    font-size:1.4rem;
    font-weight:600;
    margin-bottom:15px;
    padding:5px 0 5px 15px;
    display:flex;
    align-items:center;
}
.geom-h{border-left:5px solid #38bdf8;color:#38bdf8;}
.grad-h{border-left:5px solid #22c55e;color:#22c55e;}
.partial-h{border-left:5px solid #facc15;color:#facc15;}
.theory-h{border-left:5px solid #a78bfa;color:#a78bfa;}
.critical-h{border-left:5px solid #f472b6;color:#f472b6;}
.explanation-text{
    color:#8b949e;
    font-size:0.95rem;
    line-height:1.6;
    margin-bottom:15px;
}
.stTextInput input {
    background-color:#010409!important;
    border:1px solid #30363d!important;
    color:white!important;
}
.stSelectbox select {
    background-color:#010409!important;
    border:1px solid #30363d!important;
    color:white!important;
}
.stRadio div label, 
.stRadio div label span {
    color:white !important;
    white-space: nowrap !important;
    overflow: visible !important;
}
hr{border:0;height:1px;background:#30363d;margin:20px 0;}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# FUNCTIONS
# ---------------------------
def robust_math_parse(text):
    t = text.lower().replace(" ","")
    t = re.sub(r'(\d)([a-z\(])',r'\1*\2',t)
    t = re.sub(r'([x-z])([a-z\(])',r'\1*\2',t)
    return t.replace("^","**")

MATH_MAP={"sin":sp.sin,"cos":sp.cos,"tan":sp.tan,"exp":sp.exp,"sqrt":sp.sqrt,"log":sp.log}

def singular_points(expr):
    s=set()
    for n in sp.preorder_traversal(expr):
        if isinstance(n, sp.Pow) and n.exp.is_negative: s.add(n.base)
        if n.func in [sp.log, sp.sqrt]: s.add(n.args[0])
    return s

def continuity_statement(expr):
    s = singular_points(expr)
    if not s: return ("✅ The function is continuous everywhere.","✅ The function is differentiable everywhere.")
    pts=", ".join([f"{v}=0" for v in s])
    return (f"⚠️ The function is **not continuous** at {pts}.",f"✅ Continuous and differentiable on domain excluding {pts}.")

# ---------------------------
# SIDEBAR
# ---------------------------
with st.sidebar:
    st.header("📍 Domain")
    x_min = st.number_input("X min", -5.0)
    x_max = st.number_input("X max", 5.0)
    y_min = st.number_input("Y min", -5.0)
    y_max = st.number_input("Y max", 5.0)
    
# ---------------------------
# MAIN INTERFACE
# ---------------------------
st.title("Multivariable Calculus Explorer")
st.markdown("**Define your function:**")
user_raw = st.text_input("", value="sin(2*x) + 5*cos(y) - 8*z")
x,y,z = sp.symbols("x y z")
try:
    processed = robust_math_parse(user_raw)
    f_sym = parse_expr(processed, local_dict=MATH_MAP)
    vars_present = f_sym.free_symbols
    for v,sym in zip(['x','y','z'],[x,y,z]):
        if v in processed: vars_present |= {sym}
except Exception as e: st.error(f"Syntax Error: {e}"); st.stop()

fx, fy, fz = sp.diff(f_sym, x), sp.diff(f_sym, y), sp.diff(f_sym, z)

# ---------------------------
# ---------------------------
# I.GEOMETRY
# ---------------------------
col_left, col_right = st.columns([3, 2])
with col_left:
    st.markdown('<div class="topic-card">', unsafe_allow_html=True)
    st.markdown('<div class="topic-header geom-h">I. Geometric Meaning & Visualization</div>', unsafe_allow_html=True)

    if z in vars_present:
        st.markdown(
            '<div class="explanation-text">'
            'This is a <b>scalar field</b>. We fix <b>z = z₀</b> to obtain a level surface. We use a level surface (z-slice) to visualize the topography at a specific altitude. '
            '</div>',
            unsafe_allow_html=True
        )
        z_slice = st.slider("Fix z = z₀", -5.0, 5.0, 0.0, 0.1)
        f_plot = f_sym.subs(z, z_slice)
    else:
        st.markdown(
            '<div class="explanation-text">'
            'Standard surface z =f (x,y).'
            '</div>',
            unsafe_allow_html=True
        )
        f_plot = f_sym

    X, Y = np.meshgrid(
        np.linspace(x_min, x_max, 80),
        np.linspace(y_min, y_max, 80)
    )

    f_num = sp.lambdify((x, y), f_plot, "numpy")
    with np.errstate(all="ignore"):
        Z_vals = f_num(X, Y)

    Z_vals = np.real(Z_vals)
    Z_vals[~np.isfinite(Z_vals)] = np.nan

    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z_vals, colorscale="IceFire")])
    fig.update_layout(
        template="plotly_dark",
        height=500,
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write(f"**Domain:** $x\in[{x_min},{x_max}], y\in[{y_min},{y_max}]$")
    st.write(f"**Observed Range:** $f\in[{np.nanmin(Z_vals):.2f},{np.nanmax(Z_vals):.2f}]$")

    if any(t in user_raw.lower() for t in ["sin", "cos", "tan"]):
        st.markdown("---")
        st.write("**Shape:** Periodic/Wave Surface. The function oscillates between high and low points.")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# GRADIENT
# ---------------------------
with col_right:
    st.markdown('<div class="topic-card">',unsafe_allow_html=True)
    st.markdown('<div class="topic-header grad-h">II. Gradient & Steepest Ascent</div>',unsafe_allow_html=True)
    st.markdown('<div class="explanation-text">The <b>Gradient</b> is a vector field that lives in the domain of the function. Its most critical property is that at any given point, the gradient vector points in the <b>Direction of Steepest Ascent</b>. If you were standing on this surface, following the gradient would be the most efficient path to go uphill. The magnitude of the gradient tells you exactly how steep that slope is.</div>',unsafe_allow_html=True)
    g_comps=[sp.latex(fx),sp.latex(fy)]+([sp.latex(fz)] if z in vars_present else [])
    st.latex(rf"\nabla f = \langle {', '.join(g_comps)} \rangle")
    grad_mag=sp.sqrt(fx**2+fy**2+(fz**2 if z in vars_present else 0))
    st.latex(rf"\|\nabla f\| = {sp.latex(grad_mag)}")
    st.write("**Geometric Meaning:**")
    st.markdown("- **Direction:** Points toward local maxima.")
    st.markdown("- **Orthogonality:** The gradient is always perpendicular to the level curves (contours).")
    st.markdown("- **Rate of Change:** $|\\nabla f|$ is the maximum possible directional derivative.")
    st.markdown("""
**Steepest Ascent Interpretation:**
- The **direction** of steepest ascent is given by the gradient vector $\\nabla f$.
- The **maximum rate of increase** of the function at a point (the steepness) is given by the magnitude $|\\nabla f|$.
- A larger gradient magnitude indicates a steeper surface at that point.
""")
    st.markdown('</div>',unsafe_allow_html=True)

# ---------------------------
# PARTIAL DERIVATIVES
# ---------------------------
st.markdown('<div class="topic-card">',unsafe_allow_html=True)
st.markdown('<div class="topic-header partial-h">III. Partial Derivative Explorer</div>',unsafe_allow_html=True)
p_col1, p_col2 = st.columns(2)
with p_col1:
    st.markdown("**Differentiable with respect to:**")
    var_choice = st.selectbox("", [v for v in ['x','y','z'] if sp.Symbol(v) in vars_present])
with p_col2:
    st.markdown("**Order of derivative:**")
    order = st.radio("", [1, 2], horizontal=True)

var=sp.Symbol(var_choice)
partial=sp.diff(f_sym,var,order)
st.write("**Resulting Partial Derivative:**")
st.latex(rf"\frac{{\partial^{order} f}}{{\partial {var_choice}^{order}}} = {sp.latex(partial)}")
st.markdown('</div>',unsafe_allow_html=True)

# ---------------------------
# TOTAL DIFFERENTIAL & DIFFERENTIABILITY
# ---------------------------
t_col1,t_col2=st.columns(2)
with t_col1:
    st.markdown('<div class="topic-card">',unsafe_allow_html=True)
    st.markdown('<div class="topic-header theory-h">IV. Total Differential (df)</div>',unsafe_allow_html=True)
    st.markdown('<div class="explanation-text">The total differential represents the principal part of the change in a function with respect to changes in the independent variables.</div>',unsafe_allow_html=True)
    df_expr = rf"df = \left( {sp.latex(fx)} \right)dx + \left( {sp.latex(fy)} \right)dy"+(rf" + \left( {sp.latex(fz)} \right)dz" if z in vars_present else "")
    st.latex(df_expr); st.markdown('</div>',unsafe_allow_html=True)
with t_col2:
    st.markdown('<div class="topic-card">',unsafe_allow_html=True)
    st.markdown('<div class="topic-header theory-h">V. Differentiability</div>',unsafe_allow_html=True)
    st.markdown('<div class="explanation-text">Continuity and differentiability are fundamental properties. A function differentiable at a point is always continuous there. Elementary functions are continuous and differentiable on their domain.</div>',unsafe_allow_html=True)
cont_msg,diff_msg=continuity_statement(f_sym)
st.write(cont_msg); st.write(diff_msg)
st.markdown('</div>',unsafe_allow_html=True)

# ---------------------------
# CRITICAL POINTS
# ---------------------------
st.markdown('<div class="topic-card">',unsafe_allow_html=True)
st.markdown('<div class="topic-header critical-h">VI. Critical Points & Classification</div>',unsafe_allow_html=True)
st.markdown('<div class="explanation-text">A critical point occurs when a function derivative is zero or undefined, indicating a potential local maximum or minimum, or a saddle point where the slope changes direction. </div>',unsafe_allow_html=True)

eqs, vars_ = [], []
for sym,deriv in zip([x,y,z],[fx,fy,fz]):
    if sym in vars_present: eqs.append(deriv); vars_.append(sym)
try:
    sols = sp.solve(eqs, vars_, dict=True)
    if sols:
        for s in sols[:3]:
            fxx = sp.diff(f_sym, x, 2).subs(s) if x in vars_present else None
            fyy = sp.diff(f_sym, y, 2).subs(s) if y in vars_present else None
            fxy = sp.diff(f_sym, x, y).subs(s) if {x,y}.issubset(vars_present) else None
            label = "Critical Point"
            if fxx is not None and fyy is not None and fxy is not None:
                D=fxx*fyy-fxy**2
                label="Saddle Point" if D<0 else ("Local Minimum" if fxx>0 else "Local Maximum")
            point_coords = tuple(s[v] for v in vars_)
            st.write(f"📍 **Point {point_coords}** — **{label}**")
    else: st.info("No stationary points found.")
except: st.warning("Analytical solution too complex for real-time solver.")
st.markdown('</div>',unsafe_allow_html=True)
