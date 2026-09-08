import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

# ==========================================
# 🌟 基礎設定與基準數據 (Australia Benchmark)
# ==========================================
st.set_page_config(page_title="🥍 Sixes Lacrosse 運科戰情室", layout="wide")

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

# 🌟 全局字體與圖表外觀設定中心
GLOBAL_FONT_SIZE = 16       
DATA_LABEL_SIZE = 18        
TITLE_FONT_SIZE = 20        

PLOTLY_CONFIG = {
    'displayModeBar': True,
    'toImageButtonOptions': {'format': 'png', 'filename': 'Lacrosse_GPS_Chart', 'scale': 3}
}

def apply_chart_style(fig):
    fig.update_layout(
        font=dict(size=GLOBAL_FONT_SIZE, family="Arial, sans-serif"),
        legend=dict(font=dict(size=GLOBAL_FONT_SIZE)),
        xaxis=dict(tickfont=dict(size=GLOBAL_FONT_SIZE), title=dict(font=dict(size=GLOBAL_FONT_SIZE))),
        yaxis=dict(tickfont=dict(size=GLOBAL_FONT_SIZE), title=dict(font=dict(size=GLOBAL_FONT_SIZE))),
    )
    for trace in fig.data:
        if hasattr(trace, 'textfont'):
            trace.textfont = dict(size=DATA_LABEL_SIZE)
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

# 物理學外掛：預先計算所有資料的衝刺密度
df['HSD Density (m/min)'] = df['Avg Speed (m/min)'] * df['HSD Ratio']
df['HSD Ratio (%)'] = df['HSD Ratio'] * 100

# ==========================================
# 🌟 高效動態聚合模組 (針對跨日週期)
# ==========================================
@st.cache_data
def generate_agg_df(subset_df, period_name):
    daily_totals = subset_df[subset_df['Session'].astype(str).str.contains('Total|total', case=False, na=False)]
    if daily_totals.empty: daily_totals = subset_df
        
    agg_funcs = {'Total Distance (m)': 'sum', 'Avg Speed (m/min)': 'mean', 'Top Speed (m/s)': 'max', 'HSD Ratio': 'mean'}
    if 'RPE' in daily_totals.columns: agg_funcs['RPE'] = 'mean'
    
    agg = daily_totals.groupby('Player').agg(agg_funcs).reset_index()
    if 'RPE' in agg.columns: agg['RPE'] = agg['RPE'].round(1)
        
    agg['Date'], agg['Session'] = period_name, period_name + ' Total'
    agg['HSD Ratio (%)'] = agg['HSD Ratio'] * 100
    agg['HSD Density (m/min)'] = agg['Avg Speed (m/min)'] * agg['HSD Ratio']
    return agg

agg_dfs = []
for m in df['Month'].unique():
    if m > 0:
        m_df = df[df['Month'] == m]
        if not m_df.empty: agg_dfs.append(generate_agg_df(m_df, f'{m}月份'))
            
q1_df = df[df['Month'].isin([1, 2, 3])]
if not q1_df.empty: agg_dfs.append(generate_agg_df(q1_df, 'Q1 (1-3月)'))

if 'custom_periods' not in st.session_state: st.session_state['custom_periods'] = {}

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
    if not c_df.empty: agg_dfs.append(generate_agg_df(c_df, c_name))

if agg_dfs: df = pd.concat([df] + agg_dfs, ignore_index=True)
custom_and_auto_names = list(st.session_state['custom_periods'].keys()) + ['Q1 (1-3月)'] + [f'{m}月份' for m in df['Month'].unique() if m > 0]

st.sidebar.markdown("---") 
page_mode = st.sidebar.radio(
    "📌 選擇報告層級：", 
    [
        "📊 1. During Event (當日團隊總覽)", 
        "📈 2. Post Event (賽後進步診斷)",
        "👤 3. Individual (個人歷史履歷)"
    ]
)
st.sidebar.markdown("---") 

# ==========================================
# 🚀 模式一：During Event Report (當日團隊總覽)
# ==========================================
if page_mode == "📊 1. During Event (當日團隊總覽)":
    st.title("🥍 During Event Report - 單日團隊負荷診斷")
    
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

        st.subheader(f"1️⃣ {selected_session} 外部與內部總負荷 (Loading)")
        fig1 = go.Figure()
        hover_text = df_plot.apply(lambda row: f"Distance: {row['Total Distance (m)']:.0f} m<br>RPE: {row['RPE']}" if 'RPE' in row and pd.notna(row['RPE']) else f"Distance: {row['Total Distance (m)']:.0f} m", axis=1)
        display_text = df_plot.apply(lambda row: f"{row['Total Distance (m)']:.0f}<br>(RPE: {row['RPE']})" if 'RPE' in row and pd.notna(row['RPE']) else f"{row['Total Distance (m)']:.0f}", axis=1)

        fig1.add_trace(go.Bar(x=df_plot['Player'], y=df_plot['Total Distance (m)'], text=display_text, textposition='auto', hoverinfo='text', hovertext=hover_text, marker_color='#4a86e8', name='Total Distance'))
        team_avg_dist = df_plot['Total Distance (m)'].mean()
        if pd.notna(team_avg_dist): fig1.add_hline(y=team_avg_dist, line_dash="dash", line_color="#e06666", annotation_text="Team Avg", annotation_position="top right", annotation_font_size=GLOBAL_FONT_SIZE)
        fig1.update_layout(yaxis_title="<b>Total Distance (m)</b>", margin=dict(t=20, b=20), height=450)
        fig1 = apply_chart_style(fig1)
        st.plotly_chart(fig1, use_container_width=True, config=PLOTLY_CONFIG)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("2️⃣ 平均速度表現")
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=df_plot['Player'], y=df_plot['Avg Speed (m/min)'], text=df_plot['Avg Speed (m/min)'].round(1), textposition='auto', marker_color='#8e7cc3'))
            fig2.add_hline(y=AUS_AVG_SPEED, line_width=3, line_color="gold", annotation_text="AUS SL", annotation_position="top right", annotation_font_size=GLOBAL_FONT_SIZE)
            team_avg_spd = df_plot['Avg Speed (m/min)'].mean()
            if pd.notna(team_avg_spd): fig2.add_hline(y=team_avg_spd, line_dash="dash", line_color="red", opacity=0.5, annotation_text="Team Avg", annotation_font_size=GLOBAL_FONT_SIZE)
            fig2.update_layout(yaxis_title="<b>Avg Speed (m/min)</b>", margin=dict(t=20, b=20), height=450)
            fig2 = apply_chart_style(fig2)
            st.plotly_chart(fig2, use_container_width=True, config=PLOTLY_CONFIG)

        with col2:
            st.subheader("3️⃣ 單節/單一科目 體能消長")
            drill_sessions = [s for s in sessions_for_date if 'total' not in str(s).lower()]
            drill_sessions = sorted(drill_sessions)
            df_q = pd.DataFrame()
            if len(drill_sessions) > 0:
                df_q = df[df['Session'].isin(drill_sessions)]
                fig3_q = px.bar(df_q, x='Player', y='Total Distance (m)', color='Session', barmode='group', text_auto='.0f', color_discrete_sequence=px.colors.qualitative.Safe)
                team_avg_q_dist = df_q['Total Distance (m)'].mean()
                if pd.notna(team_avg_q_dist): fig3_q.add_hline(y=team_avg_q_dist, line_dash="dash", line_color="#e06666", annotation_text="Drill Avg", annotation_font_size=GLOBAL_FONT_SIZE)
                fig3_q.update_layout(yaxis_title="<b>Total Distance (m)</b>", margin=dict(t=20, b=20), height=450)
                fig3_q = apply_chart_style(fig3_q)
                st.plotly_chart(fig3_q, use_container_width=True, config=PLOTLY_CONFIG)
            else: st.info("💡 此時段為單日加總資料，無獨立 Drill。")

        if not df_q.empty:
            st.write("<br>", unsafe_allow_html=True)
            st.subheader("4️⃣ 訓練特徵熱力圖矩陣 (Heatmap Matrix)")
            heatmap_metric = st.radio("請選擇熱力圖色彩權重：", ["⚡ 檢視 衝刺密度 (HSD m/min) - 關注絕對無氧消耗", "📊 檢視 HSD 佔比 (%) - 關注爆發力跑動特徵"], horizontal=True)
            
            if "密度" in heatmap_metric:
                val_col = 'HSD Density (m/min)'
                color_scale = 'OrRd'  
                z_mean = df_q[val_col].mean()
                z_std = df_q[val_col].std()
                z_max = z_mean + 2 * z_std if pd.notna(z_std) and z_std > 0 else df_q[val_col].max()
            else:
                val_col = 'HSD Ratio (%)'
                color_scale = 'YlGnBu' 
                z_max = df_q[val_col].max()
                
            if pd.isna(z_max) or z_max == 0: z_max = 1  
            pivot_df = df_q.pivot_table(index='Player', columns='Session', values=val_col, aggfunc='max').sort_index()
            
            fig4 = go.Figure(data=go.Heatmap(
                z=pivot_df.values, x=pivot_df.columns, y=pivot_df.index, colorscale=color_scale, zmin=0, zmax=z_max,
                text=np.round(pivot_df.values, 1), texttemplate="%{text}", textfont={"size": DATA_LABEL_SIZE}, hoverongaps=False,
                hovertemplate="Player: %{y}<br>Session: %{x}<br>Value: %{z:.1f}<extra></extra>"
            ))
            dynamic_height = max(350, len(pivot_df.index) * 45 + 100)
            fig4.update_layout(xaxis_title="<b>Session</b>", yaxis_title="<b>Player</b>", margin=dict(t=20, b=20), height=dynamic_height)
            fig4 = apply_chart_style(fig4)
            st.plotly_chart(fig4, use_container_width=True, config=PLOTLY_CONFIG)
    else: st.warning("此時段沒有數據喔！")

# ==========================================
# 🚀 模式二：Post Event Report (賽後進步診斷)
# ==========================================
elif page_mode == "📈 2. Post Event (賽後進步診斷)":
    st.title("🥍 Post Event Report - 進步與實戰診斷書")
    
    # 抓取所有實際包含 '/' 的單日日期，並照月份/日期排序
    actual_dates = [d for d in df['Date'].unique() if '/' in str(d)]
    actual_dates = sorted(actual_dates, key=lambda x: (int(x.split('/')[0]), int(x.split('/')[1])))

    if not actual_dates:
        st.warning("目前沒有足夠的單日集訓數據可供分析。")
        st.stop()

    st.sidebar.header("🎯 診斷設定")
    all_players = sorted(df['Player'].unique().tolist())
    target_players = st.sidebar.multiselect("1. 選擇目標選手 (Target Players)：", all_players, default=all_players[:2] if len(all_players)>1 else all_players)
    
    default_trend_dates = actual_dates[-5:] if len(actual_dates) >= 5 else actual_dates
    selected_trend_dates = st.sidebar.multiselect("2. 選擇趨勢圖要顯示的日期：", actual_dates, default=default_trend_dates)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🏆 突破榜單設定**")
    compare_base = st.sidebar.selectbox("比較基準 (Base Event)：", actual_dates, index=max(0, len(actual_dates)-2))
    compare_curr = st.sidebar.selectbox("當前驗收 (Current Event)：", actual_dates, index=len(actual_dates)-1)
    
    st.write("---")
    # ------------------------------------------
    # 模組一：雙軌趨勢圖 (Dual-Axis Progression)
    # ------------------------------------------
    st.subheader("📊 模組一：目標選手進步追蹤 (Loading vs Progression)")
    
    progression_metric = st.radio(
        "選擇右軸 (折線圖) 要顯示的指標：", 
        ["HSD per min", "HSD ratio", "Max Speed"], 
        horizontal=True
    )

    if not target_players:
        st.info("請從左側面板選擇 Target Players。")
    elif not selected_trend_dates:
        st.warning("請至少選擇一個日期來繪製趨勢圖。")
    else:
        # 確保使用者選取的日期能按照時間先後排序
        sorted_trend_dates = sorted(selected_trend_dates, key=lambda x: (int(x.split('/')[0]), int(x.split('/')[1])))
        
        for player in target_players:
            p_df = df[df['Player'] == player]
            p_stats = []
            
            for d in sorted_trend_dates:
                d_df = p_df[p_df['Date'] == d]
                total_sessions = d_df[d_df['Session'].astype(str).str.contains('Total', case=False)]
                drill_sessions = d_df[~d_df['Session'].astype(str).str.contains('Total', case=False)]
                
                if not total_sessions.empty: tot_dist = total_sessions['Total Distance (m)'].sum()
                else: tot_dist = drill_sessions['Total Distance (m)'].sum() if not drill_sessions.empty else 0
                
                peak_val = 0
                if not drill_sessions.empty:
                    if progression_metric == "HSD per min":
                        peak_val = drill_sessions['HSD Density (m/min)'].max()
                    elif progression_metric == "HSD ratio":
                        peak_val = drill_sessions['HSD Ratio (%)'].max()
                    elif progression_metric == "Max Speed":
                        peak_val = drill_sessions['Top Speed (m/s)'].max()
                
                if tot_dist > 0 or peak_val > 0:
                    p_stats.append({'Date': d, 'Distance': tot_dist, 'Target Metric': peak_val})
                    
            if p_stats:
                p_stat_df = pd.DataFrame(p_stats)
                
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Bar(
                    x=p_stat_df['Date'], y=p_stat_df['Distance'], 
                    name="Distance (m)", marker_color="#4a86e8", 
                    text=p_stat_df['Distance'].astype(int), textposition='inside', opacity=0.7
                ), secondary_y=False)
                
                fig.add_trace(go.Scatter(
                    x=p_stat_df['Date'], y=p_stat_df['Target Metric'], 
                    name=progression_metric, mode="lines+markers+text", 
                    line=dict(color="#d35400", width=4), marker=dict(size=12, symbol="diamond"), 
                    text=p_stat_df['Target Metric'].round(1), textposition="top center", 
                    textfont=dict(color="#d35400", size=DATA_LABEL_SIZE)
                ), secondary_y=True)
                
                fig.update_layout(
                    title=dict(text=f"<b>{player}</b>", font=dict(size=TITLE_FONT_SIZE)), 
                    height=400, margin=dict(l=20, r=20, t=50, b=20), 
                    showlegend=False, hovermode='x unified'
                )
                
                fig.update_yaxes(title_text="<b>Distance (m)</b>", secondary_y=False, showgrid=False, range=[0, max(p_stat_df['Distance']) * 1.3])
                fig.update_yaxes(title_text=f"<b>{progression_metric}</b>", secondary_y=True, showgrid=False, range=[0, max(p_stat_df['Target Metric']) * 1.3])
                
                fig = apply_chart_style(fig)
                st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    st.write("---")
    
    col2_1, col2_2 = st.columns(2)
    
    # ------------------------------------------
    # 模組二：實戰對抗強度演進
    # ------------------------------------------
    with col2_1:
        st.subheader("⚔️ 模組二：實戰對抗強度演進 (Game/Scrimmage)")
        game_df = df[df['Session'].astype(str).str.contains('Game|Scrimmage|比賽', case=False, na=False)].copy()
        
        if not game_df.empty:
            game_trend = game_df.groupby('Date')['HSD Density (m/min)'].mean().reset_index()
            game_trend['Date'] = pd.Categorical(game_trend['Date'], categories=actual_dates, ordered=True)
            game_trend = game_trend.dropna().sort_values('Date')
            
            if selected_trend_dates:
                game_trend = game_trend[game_trend['Date'].isin(selected_trend_dates)]
            
            if not game_trend.empty:
                fig_game = go.Figure()
                fig_game.add_trace(go.Bar(
                    x=game_trend['Date'], y=game_trend['HSD Density (m/min)'], 
                    text=game_trend['HSD Density (m/min)'].round(2), textposition='auto', marker_color='#27ae60'
                ))
                fig_game.update_layout(title=dict(text="<b>全隊平均實戰衝刺密度 (HSD per min)</b>"), yaxis_title="<b>HSD per min</b>", height=400, margin=dict(t=40, b=20))
                fig_game = apply_chart_style(fig_game)
                st.plotly_chart(fig_game, use_container_width=True, config=PLOTLY_CONFIG)
            else:
                st.info("您選取的日期區間內，沒有進行 Game 或 Scrimmage 實戰。")
        else:
            st.info("歷史資料中找不到包含 'Game' 或 'Scrimmage' 的科目。")

    # ------------------------------------------
    # 模組三：體能動態榜單 (Delta Board)
    # ------------------------------------------
    with col2_2:
        st.subheader(f"🚨 模組三：體能動態榜 ({compare_base} vs {compare_curr})")
        
        def get_peak_stats(date_str):
            sub_df = df[df['Date'] == date_str]
            drill_df = sub_df[~sub_df['Session'].astype(str).str.contains('Total', case=False)]
            if drill_df.empty: return pd.DataFrame()
            return drill_df.groupby('Player').agg({'Top Speed (m/s)': 'max', 'HSD Density (m/min)': 'max'}).reset_index()
            
        base_df = get_peak_stats(compare_base)
        curr_df = get_peak_stats(compare_curr)
        
        if base_df.empty or curr_df.empty:
            st.warning("所選的日期缺乏獨立訓練科目可供比較。")
        else:
            delta_df = pd.merge(base_df, curr_df, on='Player', suffixes=('_base', '_curr'))
            delta_df['Speed_Delta'] = delta_df['Top Speed (m/s)_curr'] - delta_df['Top Speed (m/s)_base']
            delta_df['HSD_Delta'] = delta_df['HSD Density (m/min)_curr'] - delta_df['HSD Density (m/min)_base']
            
            risers = delta_df.sort_values('HSD_Delta', ascending=False).head(3)
            fallers = delta_df.sort_values('HSD_Delta', ascending=True).head(3)
            fallers = fallers[fallers['HSD_Delta'] < 0] 
            
            st.markdown("#### 📈 狀態上升榜 (Risers - 衝刺密度提升)")
            for _, row in risers.iterrows():
                if row['HSD_Delta'] > 0:
                    st.success(f"**{row['Player']}** | 密度躍升: **+{row['HSD_Delta']:.2f}** m/min (極速變化: {row['Speed_Delta']:+.1f} m/s)")
            
            st.markdown("#### 📉 疲勞/退步警示榜 (Fallers - 衝刺密度下滑)")
            if fallers.empty: st.info("無顯著衰退者，全隊維持良好輸出！")
            for _, row in fallers.iterrows():
                st.error(f"**{row['Player']}** | 密度下滑: **{row['HSD_Delta']:.2f}** m/min (極速變化: {row['Speed_Delta']:+.1f} m/s)")

# ==========================================
# 🚀 模式三：Individual Report (個人歷史履歷)
# ==========================================
elif page_mode == "👤 3. Individual (個人歷史履歷)":
    st.title("🥍 Individual Report - 個人歷史狀態體檢")
    
    st.sidebar.header("👤 監控對象設定")
    all_players = sorted(df['Player'].unique().tolist())
    selected_player = st.sidebar.selectbox("🏃 選擇監控選手：", all_players)
    
    player_sessions = df[df['Player'] == selected_player]['Session'].dropna().unique().tolist()
    player_sessions = sorted(player_sessions, key=lambda x: (0 if 'total' in str(x).lower() else 1, x))
    
    if not player_sessions:
        st.warning(f"💡 找不到 {selected_player} 的任何數據。")
    else:
        st.write("---")
        st.subheader("🔍 步驟一：選擇比較母體與檢視視角")
        view_mode = st.radio("請選擇您要如何詮釋這批數據：", ["📊 訓練量視角 (關注總量)", "⚡ 專項強度視角 (關注高強度跑動佔比)"], horizontal=True)
        
        default_selections = player_sessions[-5:] if len(player_sessions) >= 5 else player_sessions
        selected_hist_sessions = st.multiselect("請勾選歷史事件以動態計算統計基準 (μ ± σ)：", player_sessions, default=default_selections)

        if not selected_hist_sessions:
            st.info("請至少挑選一個歷史事件。")
        else:
            df_hist = df[(df['Player'] == selected_player) & (df['Session'].isin(selected_hist_sessions))].copy()
            df_hist['Session'] = pd.Categorical(df_hist['Session'], categories=selected_hist_sessions, ordered=True)
            df_hist = df_hist.sort_values('Session')

            mean_dist = df_hist['Total Distance (m)'].mean()
            std_dist = df_hist['Total Distance (m)'].std() if len(df_hist) > 1 else 0
            mean_hsd = df_hist['HSD Ratio (%)'].mean()
            std_hsd = df_hist['HSD Ratio (%)'].std() if len(df_hist) > 1 else 0
            
            vol_upper_bound = mean_dist + 2 * std_dist
            int_upper_bound = mean_hsd + 1 * std_hsd
            pr_top_speed = df_hist['Top Speed (m/s)'].max()
            mean_avg_speed = df_hist['Avg Speed (m/min)'].mean()

            latest_record = df_hist.iloc[-1]
            latest_session_name = latest_record['Session']
            
            st.write("<br>", unsafe_allow_html=True)
            st.markdown(f"#### 🔎 {latest_session_name} 的客觀狀態標籤")
            
            if "訓練量視角" in view_mode:
                latest_dist = latest_record['Total Distance (m)']
                diff_dist = latest_dist - mean_dist
                is_vol_extreme = (std_dist > 0) and (latest_dist > vol_upper_bound)
                label_color = "#1f77b4" if not is_vol_extreme else "#2c3e50" 
                label_text = f"總距離：{latest_dist:.0f} m"
                sub_text = f"與所選平均差：{'+' if diff_dist>0 else ''}{diff_dist:.0f} m"
                
                if is_vol_extreme: st.markdown(f"<div style='background-color: #e8f4f8; padding: 15px; border-left: 5px solid {label_color}; border-radius: 5px;'><h4 style='color: {label_color}; margin:0;'>[ 極端訓練量 ( > +2 SD ) ]</h4><p style='margin:5px 0 0 0; font-size:16px;'>本期{label_text}。{sub_text}，已超出常態分佈範圍。</p></div>", unsafe_allow_html=True)
                else: st.markdown(f"<div style='background-color: #f8f9fa; padding: 15px; border-left: 5px solid #808080; border-radius: 5px;'><h4 style='color: #808080; margin:0;'>[ 常態訓練量水位 ]</h4><p style='margin:5px 0 0 0; font-size:16px;'>本期{label_text}。{sub_text}，落於合理分佈區間。</p></div>", unsafe_allow_html=True)
            else:
                latest_hsd = latest_record['HSD Ratio (%)']
                diff_hsd = latest_hsd - mean_hsd
                is_int_extreme = (std_hsd > 0) and (latest_hsd > int_upper_bound)
                label_color = "#d35400" if not is_int_extreme else "#c0392b"
                label_text = f"HSD 佔比：{latest_hsd:.1f}%"
                sub_text = f"與所選平均差：{'+' if diff_hsd>0 else ''}{diff_hsd:.1f}%"
                
                if is_int_extreme: st.markdown(f"<div style='background-color: #fdedec; padding: 15px; border-left: 5px solid {label_color}; border-radius: 5px;'><h4 style='color: {label_color}; margin:0;'>[ 高強度負荷 ( > +1 SD ) ]</h4><p style='margin:5px 0 0 0; font-size:16px;'>本期{label_text}。{sub_text}，顯示經歷了高於常態的神經與無氧消耗。</p></div>", unsafe_allow_html=True)
                else: st.markdown(f"<div style='background-color: #f8f9fa; padding: 15px; border-left: 5px solid #808080; border-radius: 5px;'><h4 style='color: #808080; margin:0;'>[ 常態專項強度 ]</h4><p style='margin:5px 0 0 0; font-size:16px;'>本期{label_text}。{sub_text}，無顯著強度極端值。</p></div>", unsafe_allow_html=True)

            st.write("<br>", unsafe_allow_html=True)
            st.subheader(f"📈 步驟二：{selected_player} 統計趨勢分析")
            col_t1, col_t2 = st.columns(2)
            
            def create_trend_chart(metric_col, title, color, ref_line=None, ref_label="", ref_color="gold"):
                fig = go.Figure()
                fig.add_trace(go.Bar(x=df_hist['Session'], y=df_hist[metric_col], text=df_hist[metric_col].round(1) if 'Distance' not in metric_col else df_hist[metric_col].astype(int), textposition='auto', marker_color=color, name=selected_player))
                if ref_line is not None and pd.notna(ref_line) and ref_line > 0: fig.add_hline(y=ref_line, line_width=3, line_dash="dash", line_color=ref_color, annotation_text=ref_label, annotation_position="top right", annotation_font_size=GLOBAL_FONT_SIZE)
                fig.update_layout(title=dict(text=f"<b>{title}</b>", font=dict(size=TITLE_FONT_SIZE)), margin=dict(t=40, b=20), height=350, showlegend=False)
                return apply_chart_style(fig)

            with col_t1:
                sd2_line = vol_upper_bound if "訓練量視角" in view_mode and std_dist > 0 else mean_dist
                sd2_label = "μ + 2σ (極端值)" if "訓練量視角" in view_mode and std_dist > 0 else "μ (歷史平均)"
                fig_dist = create_trend_chart('Total Distance (m)', '總跑動距離 (Volume)', '#4a86e8', ref_line=sd2_line, ref_label=sd2_label, ref_color="#1f77b4")
                st.plotly_chart(fig_dist, use_container_width=True, config=PLOTLY_CONFIG)
                
                fig_top = create_trend_chart('Top Speed (m/s)', '最高極速表現 (Intensity/Power)', '#f6b26b', ref_line=pr_top_speed, ref_label="個人最佳 (PR)", ref_color="#e67e22")
                st.plotly_chart(fig_top, use_container_width=True, config=PLOTLY_CONFIG)

            with col_t2:
                fig_avg = create_trend_chart('Avg Speed (m/min)', '平均移動速度 (Intensity/Work Rate)', '#8e7cc3', ref_line=mean_avg_speed, ref_label="μ (歷史平均)", ref_color="#8e44ad")
                st.plotly_chart(fig_avg, use_container_width=True, config=PLOTLY_CONFIG)
                
                sd1_line = int_upper_bound if "專項強度" in view_mode and std_hsd > 0 else mean_hsd
                sd1_label = "μ + 1σ (高強度)" if "專項強度" in view_mode and std_hsd > 0 else "μ (歷史平均)"
                fig_hsd = create_trend_chart('HSD Ratio (%)', '高強度跑動佔比 (Explosiveness)', '#93c47d', ref_line=sd1_line, ref_label=sd1_label, ref_color="#c0392b")
                st.plotly_chart(fig_hsd, use_container_width=True, config=PLOTLY_CONFIG)

            st.write("<br>", unsafe_allow_html=True)
            st.subheader("📋 步驟三：課表設計數據參考中心 (Programming Data)")
            table_cols = ['Date', 'Session', 'Total Distance (m)', 'Avg Speed (m/min)', 'Top Speed (m/s)', 'HSD Ratio (%)']
            if 'RPE' in df.columns: table_cols.append('RPE')
                
            df_table = df_hist[table_cols].copy()
            numeric_cols = [col for col in table_cols if col not in ['Date', 'Session']]
            max_vals, avg_vals = df_table[numeric_cols].max(), df_table[numeric_cols].mean()
            
            df_summary = pd.DataFrame([{'Date': '---', 'Session': '🏆 所選事件最大值 (MAX/PR)'}, {'Date': '---', 'Session': '📊 所選事件平均值 (AVG/μ)'}])
            for col in numeric_cols: df_summary.loc[0, col], df_summary.loc[1, col] = max_vals[col], avg_vals[col]
                
            df_table['Total Distance (m)'], df_summary['Total Distance (m)'] = df_table['Total Distance (m)'].round(0).astype(int), df_summary['Total Distance (m)'].round(0).astype(int)
            for col in ['Avg Speed (m/min)', 'Top Speed (m/s)', 'HSD Ratio (%)', 'RPE']:
                if col in df_table.columns: df_table[col], df_summary[col] = df_table[col].round(2), df_summary[col].round(2)

            st.dataframe(pd.concat([df_table, df_summary], ignore_index=True), use_container_width=True, hide_index=True)