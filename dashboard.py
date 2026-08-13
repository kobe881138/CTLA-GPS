import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os

# ==========================================
# 🌟 基礎設定與澳洲隊基準數據 (Sixes 統一標準)
# ==========================================
st.set_page_config(page_title="🥍 Sixes Lacrosse GPS 戰情室", layout="wide")

AUS_TOP_SPEED = 7.29      
AUS_AVG_SPEED = 117.36      
AUS_HSD_RATIO = 5.07       

AUS_BASELINES = {
    'Australia Benchmark': {
        'dist': 5000, 
        'avg_spd': AUS_AVG_SPEED, 
        'top_spd': AUS_TOP_SPEED, 
        'hsd_ratio': AUS_HSD_RATIO
    }
}
default_baseline_name = list(AUS_BASELINES.keys())[0]
default_baseline_data = AUS_BASELINES[default_baseline_name]

# 🌟 全局字體與圖表外觀設定中心 (可隨時調整大小)
GLOBAL_FONT_SIZE = 16       # 座標軸、圖例的字體大小
DATA_LABEL_SIZE = 18        # 長條圖上的數字大小
TITLE_FONT_SIZE = 20        # 圖表標題/子標題字體大小

PLOTLY_CONFIG = {
    'displayModeBar': True,
    'toImageButtonOptions': {
        'format': 'png', 
        'filename': 'Lacrosse_GPS_Chart', 
        'scale': 3  # 高畫質 300 DPI
    }
}

def apply_chart_style(fig):
    """統一為所有圖表套用大字體與清晰排版"""
    fig.update_layout(
        font=dict(size=GLOBAL_FONT_SIZE, family="Arial, sans-serif"),
        legend=dict(font=dict(size=GLOBAL_FONT_SIZE)),
        xaxis=dict(tickfont=dict(size=GLOBAL_FONT_SIZE), titlefont=dict(size=GLOBAL_FONT_SIZE, weight='bold')),
        yaxis=dict(tickfont=dict(size=GLOBAL_FONT_SIZE), titlefont=dict(size=GLOBAL_FONT_SIZE, weight='bold')),
    )
    # 針對長條圖的數值標籤放大
    fig.update_traces(textfont_size=DATA_LABEL_SIZE, selector=dict(type='bar'))
    return fig

# ==========================================
# 🌟 資料載入與清理模組
# ==========================================
@st.cache_data
def load_data(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

df = load_data('Cleaned_GPS_Data.csv')

if df is None:
    st.error("❌ 找不到資料！請確認 Cleaned_GPS_Data.csv 是否存在。")
    st.stop()

if 'Zone 4 Ratio' in df.columns: df.rename(columns={'Zone 4 Ratio': 'HSD Ratio'}, inplace=True)
if 'Zone 4 Distance (m)' in df.columns: df.rename(columns={'Zone 4 Distance (m)': 'HSD (m)'}, inplace=True)

df = df[~df['Player'].astype(str).str.contains('#')].copy()
df['Date'] = df['Session'].astype(str).apply(lambda x: x.split()[0])

def get_month(date_str):
    try: return int(str(date_str).split('/')[0])
    except: return 0
df['Month'] = df['Date'].apply(get_month)

# ==========================================
# 🌟 高效動態聚合模組
# ==========================================
@st.cache_data
def generate_agg_df(subset_df, period_name):
    daily_totals = subset_df[subset_df['Session'].astype(str).str.contains('Total|total', case=False, na=False)]
    if daily_totals.empty:
        daily_totals = subset_df
        
    agg_funcs = {
        'Total Distance (m)': 'sum',
        'Avg Speed (m/min)': 'mean',
        'Top Speed (m/s)': 'max',
        'HSD Ratio': 'mean'
    }
    if 'RPE' in daily_totals.columns: agg_funcs['RPE'] = 'mean'
    
    agg = daily_totals.groupby('Player').agg(agg_funcs).reset_index()
    if 'RPE' in agg.columns: agg['RPE'] = agg['RPE'].round(1)
        
    agg['Date'] = period_name
    agg['Session'] = period_name + ' Total'
    return agg

agg_dfs = []
for m in df['Month'].unique():
    if m > 0:
        m_df = df[df['Month'] == m]
        if not m_df.empty:
            agg_dfs.append(generate_agg_df(m_df, f'{m}月份'))
            
q1_df = df[df['Month'].isin([1, 2, 3])]
if not q1_df.empty:
    agg_dfs.append(generate_agg_df(q1_df, 'Q1 (1-3月)'))

# ==========================================
# 🌟 側邊欄與自定義週期 (盃賽融合器)
# ==========================================
if 'custom_periods' not in st.session_state:
    st.session_state['custom_periods'] = {}

st.sidebar.title("🥍 戰情室導覽")
st.sidebar.markdown("### 🔄 建立專屬盃賽/週期")
raw_dates = [d for d in df['Date'].unique() if '/' in str(d)]

with st.sidebar.expander("🛠️ 點此展開盃賽融合器"):
    new_cycle_name = st.text_input("週期名稱 (例: 全國賽):")
    selected_cycle_dates = st.multiselect("選擇要融合的日期:", raw_dates)
    if st.button("➕ 建立專屬週期資料"):
        if new_cycle_name and selected_cycle_dates:
            st.session_state['custom_periods'][new_cycle_name] = selected_cycle_dates
            st.rerun()

for c_name, c_dates in st.session_state['custom_periods'].items():
    c_df = df[df['Date'].isin(c_dates)]
    if not c_df.empty:
        agg_dfs.append(generate_agg_df(c_df, c_name))

if agg_dfs:
    df = pd.concat([df] + agg_dfs, ignore_index=True)
    
custom_and_auto_names = list(st.session_state['custom_periods'].keys()) + ['Q1 (1-3月)'] + [f'{m}月份' for m in df['Month'].unique() if m > 0]

st.sidebar.markdown("---") 
page_mode = st.sidebar.radio(
    "📌 選擇分析模式：", 
    ["📊 團隊總覽 (Team Dashboard)", "👤 個人報告 (Player Profile)"]
)
st.sidebar.markdown("---") 

# ==========================================
# 🚀 模式一：團隊總覽 (Team Dashboard)
# ==========================================
if page_mode == "📊 團隊總覽 (Team Dashboard)":
    st.title("🥍 Sixes Lacrosse 團隊戰情室")
    st.caption("💡 提示：將滑鼠移至任意圖表右上角，點擊 **照相機圖示 📷** 即可下載高畫質 PNG 檔供報告使用。")
    
    st.sidebar.header("⚙️ 團隊設定面板")
    available_dates = df['Date'].dropna().unique().tolist()
    for name in reversed(custom_and_auto_names):
        if name in available_dates:
            available_dates.remove(name)
            available_dates.insert(0, name)
            
    selected_date = st.sidebar.selectbox("📅 第一步：選擇日期或週期", available_dates, key='team_date')
    sessions_for_date = df[df['Date'] == selected_date]['Session'].unique().tolist()
    selected_session = st.sidebar.selectbox("⏱️ 第二步：選擇時段", sessions_for_date, key='team_session')
    
    st.write("---")
    df_filtered = df[df['Session'] == selected_session]
    
    if not df_filtered.empty:
        agg_dict = {'Total Distance (m)': 'max', 'Avg Speed (m/min)': 'mean', 'Top Speed (m/s)': 'max', 'HSD Ratio': 'max'}
        if 'RPE' in df_filtered.columns: agg_dict['RPE'] = 'max'
        
        df_plot = df_filtered.groupby('Player').agg(agg_dict).reset_index()

        # ------------------------------------------
        # 1️⃣ 外部與內部負荷
        # ------------------------------------------
        st.subheader(f"1️⃣ {selected_session} 外部與內部負荷")
        fig1 = go.Figure()
        
        hover_text = df_plot.apply(lambda row: f"Distance: {row['Total Distance (m)']:.0f} m<br>RPE: {row['RPE']}" if 'RPE' in row and pd.notna(row['RPE']) else f"Distance: {row['Total Distance (m)']:.0f} m", axis=1)
        display_text = df_plot.apply(lambda row: f"{row['Total Distance (m)']:.0f}<br>(RPE: {row['RPE']})" if 'RPE' in row and pd.notna(row['RPE']) else f"{row['Total Distance (m)']:.0f}", axis=1)

        fig1.add_trace(go.Bar(
            x=df_plot['Player'], y=df_plot['Total Distance (m)'],
            text=display_text, textposition='auto', hoverinfo='text', hovertext=hover_text,
            marker_color='#4a86e8', name='Total Distance'
        ))
        
        team_avg_dist = df_plot['Total Distance (m)'].mean()
        if pd.notna(team_avg_dist):
            fig1.add_hline(y=team_avg_dist, line_dash="dash", line_color="#e06666", annotation_text="Team Avg", annotation_position="top right", annotation_font_size=GLOBAL_FONT_SIZE)
            
        fig1.update_layout(yaxis_title="<b>Total Distance (m)</b>", margin=dict(t=20, b=20), height=450)
        fig1 = apply_chart_style(fig1)
        st.plotly_chart(fig1, use_container_width=True, config=PLOTLY_CONFIG)

        col1, col2 = st.columns(2)
        with col1:
            # ------------------------------------------
            # 2️⃣ 平均速度表現
            # ------------------------------------------
            st.subheader("2️⃣ 平均速度表現")
            spd_mode = st.radio("顯示模式：", ["📌 當前時段", "📅 多日比較 (最多5天)"], horizontal=True, key='spd_mode')
            
            if spd_mode == "📌 當前時段":
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    x=df_plot['Player'], y=df_plot['Avg Speed (m/min)'],
                    text=df_plot['Avg Speed (m/min)'].round(1), textposition='auto',
                    marker_color='#8e7cc3', name='Avg Speed'
                ))
                fig2.add_hline(y=AUS_AVG_SPEED, line_width=3, line_color="gold", annotation_text="AUS SL", annotation_position="top right", annotation_font_size=GLOBAL_FONT_SIZE)
                
                team_avg_spd = df_plot['Avg Speed (m/min)'].mean()
                if pd.notna(team_avg_spd):
                    fig2.add_hline(y=team_avg_spd, line_dash="dash", line_color="red", opacity=0.5, annotation_text="Team Avg", annotation_font_size=GLOBAL_FONT_SIZE)
                    
                fig2.update_layout(yaxis_title="<b>Avg Speed (m/min)</b>", margin=dict(t=20, b=20), height=450)
                fig2 = apply_chart_style(fig2)
                st.plotly_chart(fig2, use_container_width=True, config=PLOTLY_CONFIG)
                
            else:
                valid_dates = [d for d in df['Date'].unique() if '/' in str(d) and d not in custom_and_auto_names]
                default_d = selected_date if selected_date in valid_dates else valid_dates[-1] if valid_dates else None
                selected_spd_dates = st.multiselect("選擇欲比較的日期 (最多5天)：", valid_dates, default=[default_d] if default_d else [], max_selections=5, key='spd_multi')
                
                if selected_spd_dates:
                    df_spd = df[(df['Date'].isin(selected_spd_dates)) & (df['Session'].astype(str).str.contains('Total|total', case=False, na=False))]
                    if not df_spd.empty:
                        fig2_multi = px.bar(df_spd, x='Player', y='Avg Speed (m/min)', color='Date', barmode='group', text_auto='.1f', color_discrete_sequence=px.colors.qualitative.Pastel)
                        fig2_multi.add_hline(y=AUS_AVG_SPEED, line_width=3, line_color="gold", annotation_text="AUS SL", annotation_font_size=GLOBAL_FONT_SIZE)
                        fig2_multi.update_layout(yaxis_title="<b>Avg Speed (m/min)</b>", margin=dict(t=20, b=20), height=450)
                        fig2_multi = apply_chart_style(fig2_multi)
                        st.plotly_chart(fig2_multi, use_container_width=True, config=PLOTLY_CONFIG)
                    else:
                        st.info("💡 找不到所選日期的 Total 數據來進行比較。")
                else:
                    st.info("💡 請至少選擇一個日期。")

        with col2:
            # ------------------------------------------
            # 3️⃣ 每日負荷消長 / 分段體能維持
            # ------------------------------------------
            is_custom_or_auto = selected_date in custom_and_auto_names
            if is_custom_or_auto:
                st.subheader(f"3️⃣ {selected_date} 每日負荷消長")
                if selected_date in st.session_state['custom_periods']: target_dates = st.session_state['custom_periods'][selected_date]
                elif selected_date == 'Q1 (1-3月)': target_dates = df[df['Month'].isin([1, 2, 3])]['Date'].unique().tolist()
                elif '月份' in selected_date:
                    m = int(selected_date.replace('月份', ''))
                    target_dates = df[df['Month'] == m]['Date'].unique().tolist()
                else: target_dates = []
                    
                target_dates = [d for d in target_dates if d not in custom_and_auto_names and '/' in str(d)]
                df_q = df[(df['Date'].isin(target_dates)) & (df['Session'].astype(str).str.contains('Total|total', case=False, na=False))]
                
                if not df_q.empty:
                    fig3_q = px.bar(df_q, x='Player', y='Total Distance (m)', color='Date', barmode='group', text_auto='.0f', color_discrete_sequence=px.colors.qualitative.Safe)
                    team_avg_q_dist = df_q['Total Distance (m)'].mean()
                    if pd.notna(team_avg_q_dist):
                        fig3_q.add_hline(y=team_avg_q_dist, line_dash="dash", line_color="#e06666", annotation_text="Period Daily Avg", annotation_font_size=GLOBAL_FONT_SIZE)
                    fig3_q.update_layout(yaxis_title="<b>Total Distance (m)</b>", margin=dict(t=20, b=20), height=450)
                    fig3_q = apply_chart_style(fig3_q)
                    st.plotly_chart(fig3_q, use_container_width=True, config=PLOTLY_CONFIG)
                else:
                    st.info("💡 此週期內找不到每日的 Total 資料。")
            else:
                st.subheader("3️⃣ 單節/分段 體能維持率")
                is_training = 'training' in selected_session.lower()
                quarter_sessions = [s for s in sessions_for_date if ('training' in str(s).lower()) == is_training and str(s).split()[-1].isdigit()]
                quarter_sessions = sorted(quarter_sessions)

                if len(quarter_sessions) > 0:
                    df_q = df[df['Session'].isin(quarter_sessions)]
                    fig3_q = px.bar(df_q, x='Player', y='Total Distance (m)', color='Session', barmode='group', text_auto='.0f', color_discrete_sequence=px.colors.qualitative.Safe)
                    team_avg_q_dist = df_q['Total Distance (m)'].mean()
                    if pd.notna(team_avg_q_dist):
                        fig3_q.add_hline(y=team_avg_q_dist, line_dash="dash", line_color="#e06666", annotation_text="Session Avg", annotation_font_size=GLOBAL_FONT_SIZE)
                    fig3_q.update_layout(yaxis_title="<b>Total Distance (m)</b>", margin=dict(t=20, b=20), height=450)
                    fig3_q = apply_chart_style(fig3_q)
                    st.plotly_chart(fig3_q, use_container_width=True, config=PLOTLY_CONFIG)
                else:
                    st.info("💡 此時段無單節資料或為單日加總資料。")

        # ------------------------------------------
        # 4️⃣ 爆發力象限圖 
        # ------------------------------------------
        st.write("<br>", unsafe_allow_html=True)
        st.subheader("4️⃣ 爆發力象限圖")
        spacer1, col_center, spacer2 = st.columns([1, 4, 1])
        with col_center:
            x_data = df_plot['HSD Ratio'] * 100
            y_data = df_plot['Top Speed (m/s)']
            session_avg_hsd = x_data.mean()
            session_avg_top = y_data.mean()
            
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(
                x=x_data, y=y_data, mode='markers+text', text=df_plot['Player'], textposition="top center",
                textfont=dict(size=DATA_LABEL_SIZE, color="black", weight="bold"),
                marker=dict(color='#3d85c6', size=14, line=dict(width=1, color='white')), name='Players',
                hovertemplate='<b>%{text}</b><br>HSD Ratio: %{x:.1f}%<br>Top Speed: %{y:.1f} m/s<extra></extra>'
            ))

            if pd.notna(session_avg_hsd) and pd.notna(session_avg_top):
                fig4.add_trace(go.Scatter(
                    x=[session_avg_hsd], y=[session_avg_top], mode='markers',
                    marker=dict(color='#38761d', symbol='cross', size=16), name='Team Avg',
                    hovertemplate='<b>團隊平均</b><br>HSD Ratio: %{x:.1f}%<br>Top Speed: %{y:.1f} m/s<extra></extra>'
                ))
                fig4.add_vline(x=session_avg_hsd, line_dash="dash", line_color="#38761d", opacity=0.5)
                fig4.add_hline(y=session_avg_top, line_dash="dash", line_color="#38761d", opacity=0.5)

            fig4.add_trace(go.Scatter(
                x=[AUS_HSD_RATIO], y=[AUS_TOP_SPEED], mode='markers',
                marker=dict(color='red', symbol='star', size=20, line=dict(width=1, color='darkgray')), name=default_baseline_name,
                hovertemplate=f'<b>{default_baseline_name}</b><br>HSD Ratio: %{{x:.1f}}%<br>Top Speed: %{{y:.1f}} m/s<extra></extra>'
            ))

            fig4.update_layout(
                xaxis_title='<b>HSD Ratio (%)</b>', yaxis_title='<b>Top Speed (m/s)</b>',
                margin=dict(l=20, r=20, t=30, b=20), hovermode='closest', height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig4 = apply_chart_style(fig4)
            st.plotly_chart(fig4, use_container_width=True, config=PLOTLY_CONFIG)
            
    else:
        st.warning("此時段沒有數據喔！")

# ==========================================
# 🚀 模式二：個人專屬報告 (Player Profile)
# ==========================================
elif page_mode == "👤 個人報告 (Player Profile)":
    st.title("🥍 Sixes Lacrosse 個人分析報告")
    st.caption("💡 提示：本區的雷達圖已鎖定 ±2 標準差比例，方便您匯出報告時進行不同球員的視覺化交叉對比。")
    st.sidebar.header("👤 個人報告設定")
    
    all_players = sorted(df['Player'].unique().tolist())
    selected_player = st.sidebar.selectbox("🏃 選擇選手：", all_players)
    
    player_sessions = df[df['Player'] == selected_player]['Session'].dropna().unique().tolist()
    all_sessions = df['Session'].dropna().unique().tolist()
    
    custom_session_names = [f"{name} Total" for name in custom_and_auto_names]
    for name in reversed(custom_session_names):
        if name in player_sessions:
            player_sessions.remove(name)
            player_sessions.insert(0, name)
        if name in all_sessions:
            all_sessions.remove(name)
            all_sessions.insert(0, name)
    
    if not player_sessions:
        st.warning(f"💡 找不到 {selected_player} 的任何數據。")
    else:
        st.write("---")
        st.subheader(f"🛡️ {selected_player} - 個人表現分析報告")

        col_radar, col_bar = st.columns([1, 1.5])

        with col_radar:
            # ------------------------------------------
            # 📍 鎖定比例的 Z-score 雷達圖 (字體特別優化)
            # ------------------------------------------
            st.markdown(f"##### 📍 六角雷達圖：對標團隊平均")
            radar_session = st.selectbox("📅 選擇雷達圖檢視事件：", player_sessions, index=0)
            
            team_radar_df = df[df['Session'] == radar_session]
            team_mean = team_radar_df[['Total Distance (m)', 'Avg Speed (m/min)', 'Top Speed (m/s)', 'HSD Ratio']].mean()
            team_std = team_radar_df[['Total Distance (m)', 'Avg Speed (m/min)', 'Top Speed (m/s)', 'HSD Ratio']].std().replace(0, 1).fillna(1)
            
            player_radar = df[(df['Player'] == selected_player) & (df['Session'] == radar_session)].iloc[0]
            categories = ['Total Distance', 'Avg Speed', 'Max Speed', 'HSD Ratio']
            
            def calc_z(col):
                if pd.isna(player_radar[col]) or pd.isna(team_mean[col]): return 0
                z = (player_radar[col] - team_mean[col]) / team_std[col]
                return np.clip(z, -2, 2)
                
            player_ratios = [calc_z('Total Distance (m)'), calc_z('Avg Speed (m/min)'), calc_z('Top Speed (m/s)'), calc_z('HSD Ratio')]
            player_ratios += [player_ratios[0]]
            team_ratios = [0, 0, 0, 0, 0]
            categories_plot = categories + [categories[0]]

            fig_r = go.Figure()
            fig_r.add_trace(go.Scatterpolar(
                r=team_ratios, theta=categories_plot, fill='toself', name=f'{radar_session} Team Avg (0)',
                line_color='#e06666', opacity=0.8
            ))
            fig_r.add_trace(go.Scatterpolar(
                r=player_ratios, theta=categories_plot, fill='toself', name=selected_player,
                line_color='#4a86e8', fillcolor='rgba(74, 134, 232, 0.4)'
            ))

            fig_r.update_layout(
                font=dict(size=GLOBAL_FONT_SIZE), # 放大整體字體
                polar=dict(
                    radialaxis=dict(
                        visible=True, range=[-2, 2], tickvals=[-2, -1, 0, 1, 2], ticktext=['-2', '-1', '0', '1', '2'],
                        tickfont=dict(size=GLOBAL_FONT_SIZE) # 放大量尺數字
                    ),
                    angularaxis=dict(
                        tickfont=dict(size=TITLE_FONT_SIZE, weight='bold', color='black') # 放大外圍類別文字
                    )
                ),
                margin=dict(l=60, r=60, t=40, b=40), height=450,
                legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5, font=dict(size=GLOBAL_FONT_SIZE))
            )
            st.plotly_chart(fig_r, use_container_width=True, config=PLOTLY_CONFIG)

        with col_bar:
            # ------------------------------------------
            # 📈 歷史進步軌跡 (子圖字體優化)
            # ------------------------------------------
            st.markdown("##### 📈 歷史進步軌跡")
            compare_mode = st.radio("📊 選擇比較模式：", ["雙期比較 (2個數據)", "三期比較 (3個數據)"], horizontal=True)
            baseline_options = [default_baseline_name] + all_sessions
            
            if compare_mode == "雙期比較 (2個數據)":
                col_b1, col_b2 = st.columns(2)
                with col_b1: player_selected_session = st.selectbox("📅 當前檢視事件：", player_sessions)
                with col_b2: selected_baseline1 = st.selectbox("📉 比較基準：", baseline_options)
                selected_baseline2 = None
            else:
                col_b1, col_b2, col_b3 = st.columns(3)
                with col_b1: player_selected_session = st.selectbox("📅 當前檢視事件：", player_sessions)
                with col_b2: selected_baseline1 = st.selectbox("📉 比較基準 1：", baseline_options)
                with col_b3: 
                    default_b2_idx = 1 if len(baseline_options) > 1 else 0
                    selected_baseline2 = st.selectbox("📉 比較基準 2：", baseline_options, index=default_b2_idx)

            player_current_bar = df[(df['Player'] == selected_player) & (df['Session'] == player_selected_session)].iloc[0]
            
            def get_baseline_data(b_name):
                if b_name == default_baseline_name:
                    target = default_baseline_data
                    return {
                        'Total Distance (m)': target['dist'], 'Avg Speed (m/min)': target['avg_spd'],
                        'Top Speed (m/s)': target['top_spd'], 'HSD Ratio': target['hsd_ratio'] / 100 
                    }, "AUS Avg"
                else:
                    past_data = df[(df['Player'] == selected_player) & (df['Session'] == b_name)]
                    if not past_data.empty: return past_data[['Total Distance (m)', 'Avg Speed (m/min)', 'Top Speed (m/s)', 'HSD Ratio']].mean(), b_name
                    return None, b_name

            b1_data, b1_label = get_baseline_data(selected_baseline1)
            b2_data, b2_label = None, None
            if selected_baseline2: b2_data, b2_label = get_baseline_data(selected_baseline2)

            hist_records = []
            def add_record(data_source, label_name):
                if data_source is not None:
                    hist_records.append({
                        'Period': label_name,
                        'Total Distance (m)': data_source['Total Distance (m)'],
                        'Avg Speed (m/min)': data_source['Avg Speed (m/min)'],
                        'Top Speed (m/s)': data_source['Top Speed (m/s)'],
                        'HSD Ratio (%)': data_source['HSD Ratio'] * 100 if label_name != "AUS Avg" else data_source['HSD Ratio'] * 100
                    })

            add_record(b2_data, b2_label)
            add_record(b1_data, b1_label)
            add_record(player_current_bar, player_selected_session)
            
            if hist_records:
                df_hist = pd.DataFrame(hist_records)
                df_hist_melted = df_hist.melt(id_vars=['Period'], value_vars=['Total Distance (m)', 'Avg Speed (m/min)', 'Top Speed (m/s)', 'HSD Ratio (%)'], var_name='Metric', value_name='Value')
                
                fig_hist = px.bar(
                    df_hist_melted, x='Period', y='Value', color='Period', facet_col='Metric', 
                    text_auto='.1f', color_discrete_sequence=px.colors.qualitative.Pastel
                )
                
                fig_hist.update_yaxes(matches=None, showticklabels=True, title="", tickfont=dict(size=GLOBAL_FONT_SIZE))
                fig_hist.update_xaxes(title="", showticklabels=False)
                # 放大子圖的標題 (例如 Total Distance (m))
                fig_hist.for_each_annotation(lambda a: a.update(text=f"<b>{a.text.split('=')[-1]}</b>", font=dict(size=TITLE_FONT_SIZE, color="black")))
                
                fig_hist.update_layout(margin=dict(t=50, b=20), height=450, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5, font=dict(size=GLOBAL_FONT_SIZE)))
                # 放大数据標籤
                fig_hist.update_traces(textfont_size=DATA_LABEL_SIZE, textfont_color="black")
                st.plotly_chart(fig_hist, use_container_width=True, config=PLOTLY_CONFIG)