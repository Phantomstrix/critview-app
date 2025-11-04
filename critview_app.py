import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
import altair as alt

# ---------------------------------------------------------------------
# 1. 单位转换数据库
# ---------------------------------------------------------------------
CONVERSION_FACTORS = {
    "length": {
        "centimeters (cm)": 1.0,
        "meters (m)": 100.0,
        "inches (in)": 2.54,
        "feet (ft)": 30.48,
    },
    "concentration_vol": {
        "g/cc": 1.0,
        "g/L": 1000.0,
        "kg/m^3": 1000.0,
        "g/ft^3": 28316.8466,
    },
    "mass": {
        "grams (g)": 1.0,
        "kilograms (kg)": 1000.0,
        "ounces (oz)": 28.3495,
        "pounds (lb)": 453.592,
    },
    "volume": {
        "cm^3 (cc)": 1.0,
        "liters (L)": 1000.0,
        "in^3": 16.3871,
        "ft^3": 28316.8466,
        "gallons (US)": 3785.41,
    },
    "concentration_linear": {
        "g/ft": 1.0,
        "g/cm": 30.48,
        "kg/m": 32.8084,
    },
    "concentration_areal": {
        "kg/ft^2": 1.0,
        "g/cm^2": 10.7639,
    }
}

# 【【【*** 这就是修复 (v6) ***】】】
# 键 (key) 必须与 CSV 文件中的 'X_Variable'/'Y_Variable' 字符串 *完全* 匹配
VARIABLE_TO_CATEGORY = {
    "Diameter in": ("length", "inches (in)"),
    "Height in": ("length", "inches (in)"),
    "Radius in": ("length", "inches (in)"),
    "Thickness in": ("length", "inches (in)"),
    "critconc g/cc": ("concentration_vol", "g/cc"),
    "critconc g/L": ("concentration_vol", "g/L"),
    "critconc_linear g/ft": ("concentration_linear", "g/ft"),
    "critconc_areal kg/ft2": ("concentration_areal", "kg/ft^2"),
    
    # --- 修复开始 ---
    # 旧的 (错误的): "Mass kg": ("mass", "kilograms (kg)"),
    "critmass kg": ("mass", "kilograms (kg)"),
    
    # 旧的 (错误的): "Volume L": ("volume", "liters (L)"),
    "volume L": ("volume", "liters (L)"),
    
    # 旧的 (错误的): "Volume gal": ("volume", "gallons (US)"),
    "volume gal": ("volume", "gallons (US)"), # 假设这个也是小写
    # --- 修复结束 ---
}

def get_unit_info(var_name):
    """
    根据变量名（例如 "Diameter in"）返回其类别和可用单位。
    """
    if var_name in VARIABLE_TO_CATEGORY:
        category, base_unit = VARIABLE_TO_CATEGORY[var_name]
        return category, base_unit, list(CONVERSION_FACTORS[category].keys())
    else:
        # 如果未在上面定义，则返回一个默认值，不允许转换
        return "unknown", var_name, [var_name]

# ---------------------------------------------------------------------
# 2. 数据加载和拟合函数 (未改变)
# ---------------------------------------------------------------------

@st.cache_data
def load_data(filepath="critview_data.csv"):
    """
    加载 CSV 数据并强制转换 X, Y 值为数字。
    """
    try:
        df = pd.read_csv(filepath)
        df.replace('nan', np.nan, inplace=True)
        df['X_Value'] = pd.to_numeric(df['X_Value'], errors='coerce')
        df['Y_Value'] = pd.to_numeric(df['Y_Value'], errors='coerce')
        df.dropna(subset=['X_Value', 'Y_Value'], inplace=True)
        return df
    except FileNotFoundError:
        st.error(f"错误: 未找到 '{filepath}'。请确保文件与 .py 脚本在同一目录中。")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"加载数据时出错: {e}")
        return pd.DataFrame()


@st.cache_data
def get_fitted_spline(curve_title, df):
    """
    使用参数化拟合 + 动态 K (样条阶数)。
    X = f(t), Y = g(t)
    """
    
    curve_df = df[df['title'] == curve_title]
    n_points = len(curve_df)
    
    if n_points < 2: 
        st.warning(f"曲线 '{curve_title}' 数据点不足 (<2)，无法拟合。")
        return None, None, None, None, None, None
    
    x_data = curve_df['X_Value'].values
    y_data = curve_df['Y_Value'].values
    x_var = curve_df['X_Variable'].iloc[0]
    y_var = curve_df['Y_Variable'].iloc[0]
    
    t_data = np.arange(n_points)
    k_spline = min(3, n_points - 1) 
    
    try:
        spline_x = UnivariateSpline(t_data, x_data, s=0, k=k_spline)
        spline_y = UnivariateSpline(t_data, y_data, s=0, k=k_spline)
    except Exception as e:
        st.error(f"为曲线 '{curve_title}' 创建样条拟合时出错 (k={k_spline}, points={n_points}): {e}")
        return None, None, None, None, None, None

    return spline_x, spline_y, x_data, y_data, x_var, y_var

# ---------------------------------------------------------------------
# 3. Streamlit 界面布局 (未改变)
# ---------------------------------------------------------------------

st.set_page_config(layout="wide")
st.title("CritView 数据库曲线拟合与可视化")

df = load_data()

if not df.empty:
    
    # --- 侧边栏：筛选器 ---
    st.sidebar.header("1. 筛选条件")
    filtered_df = df.copy()
    
    geom_options = np.insert(filtered_df['geometry'].dropna().unique().astype(str), 0, "All")
    selected_geom = st.sidebar.selectbox("几何形状 (Geometry):", geom_options)
    if selected_geom != "All":
        filtered_df = filtered_df[filtered_df['geometry'] == selected_geom]

    fiss_elem_options = np.insert(filtered_df['fiss-element'].dropna().unique().astype(str), 0, "All")
    selected_fiss_elem = st.sidebar.selectbox("裂变元素 (Fiss-element):", fiss_elem_options)
    if selected_fiss_elem != "All":
        filtered_df = filtered_df[filtered_df['fiss-element'] == selected_fiss_elem]

    critmat_options = np.insert(filtered_df['critmat'].dropna().unique().astype(str), 0, "All")
    selected_critmat = st.sidebar.selectbox("临界材料 (Critmat):", critmat_options)
    if selected_critmat != "All":
        filtered_df = filtered_df[filtered_df['critmat'] == selected_critmat]
        
    fiss_form_options = np.insert(filtered_df['fiss-form'].dropna().unique().astype(str), 0, "All")
    selected_fiss_form = st.sidebar.selectbox("材料形式 (Fiss-form):", fiss_form_options)
    if selected_fiss_form != "All":
        filtered_df = filtered_df[filtered_df['fiss-form'] == selected_fiss_form]

    isomat_options = np.insert(filtered_df['isomat'].dropna().unique().astype(str), 0, "All")
    selected_isomat = st.sidebar.selectbox("同位素 (Isomat):", isomat_options)
    if selected_isomat != "All":
        filtered_df = filtered_df[filtered_df['isomat'] == selected_isomat]

    modmat_options = np.insert(filtered_df['modmat'].dropna().unique().astype(str), 0, "All")
    selected_modmat = st.sidebar.selectbox("慢化剂 (Modmat):", modmat_options)
    if selected_modmat != "All":
        filtered_df = filtered_df[filtered_df['modmat'] == selected_modmat]

    refl_mat_options = np.insert(filtered_df['reflmat'].dropna().unique().astype(str), 0, "All")
    selected_refl_mat = st.sidebar.selectbox("反射层 (Reflmat):", refl_mat_options)
    if selected_refl_mat != "All":
        filtered_df = filtered_df[filtered_df['reflmat'] == selected_refl_mat]

    reflthick_options = np.insert(filtered_df['reflthick'].dropna().unique().astype(str), 0, "All")
    selected_reflthick = st.sidebar.selectbox("反射层厚度 (Reflthick):", reflthick_options)
    if selected_reflthick != "All":
        filtered_df = filtered_df[filtered_df['reflthick'] == selected_reflthick]

    x_var_options = np.insert(filtered_df['X_Variable'].dropna().unique().astype(str), 0, "All")
    selected_x_var = st.sidebar.selectbox("X 轴变量:", x_var_options)
    if selected_x_var != "All":
        filtered_df = filtered_df[filtered_df['X_Variable'] == selected_x_var]

    y_var_options = np.insert(filtered_df['Y_Variable'].dropna().unique().astype(str), 0, "All")
    selected_y_var = st.sidebar.selectbox("Y 轴变量:", y_var_options)
    if selected_y_var != "All":
        filtered_df = filtered_df[filtered_df['Y_Variable'] == selected_y_var]

    # --- 侧边栏：曲线选择 ---
    st.sidebar.header("2. 选择曲线")
    curve_titles = filtered_df['title'].unique()
    
    if len(curve_titles) == 0:
        st.sidebar.warning("在当前筛选条件下未找到曲线。")
        selected_title = None
    else:
        selected_title = st.sidebar.selectbox(
            f"选择要分析的曲线 (找到 {len(curve_titles)} 条):",
            curve_titles
        )

    # --- 侧边栏：单位换算 ---
    st.sidebar.header("3. 单位换算")
    
    # --- 主区域：拟合与绘图 ---
    if selected_title:
        
        spline_x, spline_y, x_raw, y_raw, x_var_name, y_var_name = get_fitted_spline(selected_title, df)
        
        if spline_x is None:
            pass
        else:
            # 单位换算选择
            x_category, x_base_unit, x_options = get_unit_info(x_var_name)
            y_category, y_base_unit, y_options = get_unit_info(y_var_name)

            st.sidebar.subheader(f"X 轴: {x_var_name.split(' ')[0]}")
            if x_category == "unknown":
                # 这就是您看到的警告的来源
                st.sidebar.warning(f"未知的 X 变量 '{x_var_name}'，无法转换单位。")
                x_unit_selected = x_var_name
            else:
                x_unit_selected = st.sidebar.selectbox("X 轴单位:", x_options, index=x_options.index(x_base_unit))

            st.sidebar.subheader(f"Y 轴: {y_var_name.split(' ')[0]}")
            if y_category == "unknown":
                # 这就是您看到的警告的来源
                st.sidebar.warning(f"未知的 Y 变量 '{y_var_name}'，无法转换单位。")
                y_unit_selected = y_var_name
            else:
                y_unit_selected = st.sidebar.selectbox("Y 轴单位:", y_options, index=y_options.index(y_base_unit))

            # 单位换算因子
            if x_category != "unknown":
                x_conv_factor = CONVERSION_FACTORS[x_category][x_base_unit] / CONVERSION_FACTORS[x_category][x_unit_selected]
            else:
                x_conv_factor = 1.0
            if y_category != "unknown":
                y_conv_factor = CONVERSION_FACTORS[y_category][y_base_unit] / CONVERSION_FACTORS[y_category][y_unit_selected]
            else:
                y_conv_factor = 1.0

            # 4. 准备用于绘图的数据
            
            # 4a. 原始数据点
            x_display = x_raw * x_conv_factor
            y_display = y_raw * y_conv_factor
            plot_df_scatter = pd.DataFrame({
                'x': x_display,
                'y': y_display,
                'order_col': np.arange(len(x_display))
            })

            # 4b. 拟合曲线
            t_fit = np.linspace(0, len(x_raw) - 1, 200)
            x_fit_raw = spline_x(t_fit)
            y_fit_raw = spline_y(t_fit)
            x_fit_display = x_fit_raw * x_conv_factor
            y_fit_display = y_fit_raw * y_conv_factor
            
            plot_df_line = pd.DataFrame({
                'x': x_fit_display,
                'y': y_fit_display,
                'order_col': t_fit
            })
            
            # 5. 使用 Altair 绘图
            st.header(f"曲线: {selected_title}")
            
            x_axis_title = f"X: {x_var_name.split(' ')[0]} ({x_unit_selected})"
            y_axis_title = f"Y: {y_var_name.split(' ')[0]} ({y_unit_selected})"

            # 原始数据点 (蓝色)
            scatter_plot = alt.Chart(plot_df_scatter).mark_circle(size=60, opacity=0.7).encode(
                x=alt.X('x', title=x_axis_title),
                y=alt.Y('y', title=y_axis_title),
                order='order_col',
                tooltip=[
                    alt.Tooltip('x', title=x_axis_title, format='.4e'),
                    alt.Tooltip('y', title=y_axis_title, format='.4e')
                ]
            ).interactive() 
            
            # 拟合曲线 (红色)
            line_plot = alt.Chart(plot_df_line).mark_line(color='red').encode(
                x=alt.X('x', title=x_axis_title),
                y=alt.Y('y', title=y_axis_title),
                order='order_col',
                tooltip=[
                    alt.Tooltip('x', title=x_axis_title, format='.4e'),
                    alt.Tooltip('y', title=y_axis_title, format='.4e')
                ]
            )
            
            final_chart = scatter_plot + line_plot
            st.altair_chart(final_chart, use_container_width=True)
            
            # 显示元数据
            with st.expander("查看此曲线的元数据"):
                meta_cols = [col for col in df.columns if col not in ['X_Value', 'Y_Value', 'X_Variable', 'Y_Variable']]
                meta = df[df['title'] == selected_title][meta_cols].iloc[0]
                st.dataframe(meta)