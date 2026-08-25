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

# 🌟 全局字體與圖表外觀設定中心
GLOBAL_FONT_SIZE = 16       
DATA_LABEL_SIZE = 18        
TITLE_FONT_SIZE = 20        

PLOTLY_CONFIG = {
    'displayModeBar': True,
    'toImageButtonOptions': {
        'format': 'png', 
        'filename': 'Lacrosse_GPS_Chart', 
        'scale': 3  
    }
}

def apply_chart_style(fig):
    """統一為所有圖表套用大字體與清晰排版"""
    fig.update_layout(
        font=dict(size=GLOBAL_FONT_SIZE, family="Arial, sans-serif"),
        legend=dict(font=dict(size=GLOBAL_FONT_SIZE)),
        xaxis=dict(tickfont=dict(size=GLOBAL_FONT_SIZE), title=dict(font=dict(size=GLOBAL_FONT_SIZE))),
        yaxis=dict(tickfont=dict(size=GLOBAL_FONT_SIZE), title=dict(font=dict(size=GLOBAL_FONT_SIZE))),
    )
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
# 🌟 側邊欄與自定義週期
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

        st.subheader(f"1️⃣ {selected_session} 外部與內部負荷")
        fig1 = go.Figure()
        hover_text = df_plot.apply(lambda row: f"Distance: {row['Total Distance (m)']:.0f} m<br>RPE: {row['RPE']}" if 'RPE' in row and pd.notna(row['RPE']) else f"Distance: {row['Total Distance (m)']:.0f} m", axis=1)
        display_text = df_plot.apply(lambda row: f"{row['Total Distance (m)']:.0f}<br>(RPE: {row['RPE']})" if 'RPE' in row and pd.notna(row['RPE']) else f"{row['Total Distance (m)']:.0f}", axis=1)

        fig1.add_trace(go.Bar(
            x=df_plot['Player'], y=df_plot['Total Distance (m)'], text=display_text, textposition='auto', hoverinfo='text', hovertext=hover_text,
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
            st.subheader("2️⃣ 平均速度表現")
            spd_mode = st.radio("顯示模式：", ["📌 當前時段", "📅 多日比較 (最多5天)"], horizontal=True, key='spd_mode')
            if spd_mode == "📌 當前時段":
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(x=df_plot['Player'], y=df_plot['Avg Speed (m/min)'], text=df_plot['Avg Speed (m/min)'].round(1), textposition='auto', marker_color='#8e7cc3'))
                fig2.add_hline(y=AUS_AVG_SPEED, line_width=3, line_color="gold", annotation_text="AUS SL", annotation_position="top right", annotation_font_size=GLOBAL_FONT_SIZE)
                team_avg_spd = df_plot['Avg Speed (m/min)'].mean()
                if pd.notna(team_avg_spd): fig2.add_hline(y=team_avg_spd, line_dash="dash", line_color="red", opacity=0.5, annotation_text="Team Avg", annotation_font_size=GLOBAL_FONT_SIZE)
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
                    else: st.info("💡 找不到所選日期的 Total 數據來進行比較。")
                else: st.info("💡 請至少選擇一個日期。")

        with col2:
            is_custom_or_auto = selected_date in custom_and_auto_names
            if is_custom_or_auto:
                st.subheader(f"3️⃣ {selected_date} 每日負荷消長")
                if selected_date in st.session_state['custom_periods']: target_dates = st.session_state['custom_periods'][selected_date]
                elif selected_date == 'Q1 (1-3月)': target_dates = df[df['Month'].isin([1, 2, 3])]['Date'].unique().tolist()
                elif '月份' in selected_date: target_dates = df[df['Month'] == int(selected_date.replace('月份', ''))]['Date'].unique().tolist()
                else: target_dates = []
                    
                target_dates = [d for d in target_dates if d not in custom_and_auto_names and '/' in str(d)]
                df_q = df[(df['Date'].isin(target_dates)) & (df['Session'].astype(str).str.contains('Total|total', case=False, na=False))]
                
                if not df_q.empty:
                    fig3_q = px.bar(df_q, x='Player', y='Total Distance (m)', color='Date', barmode='group', text_auto='.0f', color_discrete_sequence=px.colors.qualitative.Safe)
                    team_avg_q_dist = df_q['Total Distance (m)'].mean()
                    if pd.notna(team_avg_q_dist): fig3_q.add_hline(y=team_avg_q_dist, line_dash="dash", line_color="#e06666", annotation_text="Period Daily Avg", annotation_font_size=GLOBAL_FONT_SIZE)
                    fig3_q.update_layout(yaxis_title="<b>Total Distance (m)</b>", margin=dict(t=20, b=20), height=450)
                    fig3_q = apply_chart_style(fig3_q)
                    st.plotly_chart(fig3_q, use_container_width=True, config=PLOTLY_CONFIG)
                else: st.info("💡 此週期內找不到每日的 Total 資料。")
            else:
                st.subheader("3️⃣ 單節/單一科目 體能消長")
                drill_sessions = [s for s in sessions_for_date if 'total' not in str(s).lower()]
                drill_sessions = sorted(drill_sessions)

                if len(drill_sessions) > 0:
                    df_q = df[df['Session'].isin(drill_sessions)]
                    fig3_q = px.bar(df_q, x='Player', y='Total Distance (m)', color='Session', barmode='group', text_auto='.0f', color_discrete_sequence=px.colors.qualitative.Safe)
                    team_avg_q_dist = df_q['Total Distance (m)'].mean()
                    if pd.notna(team_avg_q_dist): fig3_q.add_hline(y=team_avg_q_dist, line_dash="dash", line_color="#e06666", annotation_text="Drill Avg", annotation_font_size=GLOBAL_FONT_SIZE)
                    fig3_q.update_layout(yaxis_title="<b>Total Distance (m)</b>", margin=dict(t=20, b=20), height=450)
                    fig3_q = apply_chart_style(fig3_q)
                    st.plotly_chart(fig3_q, use_container_width=True, config=PLOTLY_CONFIG)
                else: st.info("💡 此時段為單日加總資料，無獨立 Drill。")

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
                textfont=dict(size=DATA_LABEL_SIZE, color="black"),
                marker=dict(color='#3d85c6', size=14, line=dict(width=1, color='white')), name='Players',
                hovertemplate='<b>%{text}</b><br>HSD Ratio: %{x:.1f}%<br>Top Speed: %{y:.1f} m/s<extra></extra>'
            ))
            if pd.notna(session_avg_hsd) and pd.notna(session_avg_top):
                fig4.add_trace(go.Scatter(x=[session_avg_hsd], y=[session_avg_top], mode='markers', marker=dict(color='#38761d', symbol='cross', size=16), name='Team Avg', hovertemplate='<b>團隊平均</b><br>HSD Ratio: %{x:.1f}%<br>Top Speed: %{y:.1f} m/s<extra></extra>'))
                fig4.add_vline(x=session_avg_hsd, line_dash="dash", line_color="#38761d", opacity=0.5)
                fig4.add_hline(y=session_avg_top, line_dash="dash", line_color="#38761d", opacity=0.5)
            fig4.add_trace(go.Scatter(x=[AUS_HSD_RATIO], y=[AUS_TOP_SPEED], mode='markers', marker=dict(color='red', symbol='star', size=20, line=dict(width=1, color='darkgray')), name=default_baseline_name, hovertemplate=f'<b>{default_baseline_name}</b><br>HSD Ratio: %{{x:.1f}}%<br>Top Speed: %{{y:.1f}} m/s<extra></extra>'))
            fig4.update_layout(xaxis_title='<b>HSD Ratio (%)</b>', yaxis_title='<b>Top Speed (m/s)</b>', margin=dict(l=20, r=20, t=30, b=20), hovermode='closest', height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            fig4 = apply_chart_style(fig4)
            st.plotly_chart(fig4, use_container_width=True, config=PLOTLY_CONFIG)
    else: st.warning("此時段沒有數據喔！")

# ==========================================
# 🚀 模式二：個人戰情與開表中心 (情境與客觀統計版)
# ==========================================
elif page_mode == "👤 個人報告 (Player Profile)":
    st.title("🥍 Sixes Lacrosse 個人狀態體檢與開表中心")
    
    st.sidebar.header("👤 監控對象設定")
    all_players = sorted(df['Player'].unique().tolist())
    selected_player = st.sidebar.selectbox("🏃 選擇監控選手：", all_players)
    
    player_sessions = df[df['Player'] == selected_player]['Session'].dropna().unique().tolist()
    # 讓有 Total 字眼的排在前面
    player_sessions = sorted(player_sessions, key=lambda x: (0 if 'total' in str(x).lower() else 1, x))
    
    if not player_sessions:
        st.warning(f"💡 找不到 {selected_player} 的任何數據。")
    else:
        st.write("---")
        
        # ------------------------------------------
        # 🎛️ 1. 無限制多選篩選器 & 情境切換開關
        # ------------------------------------------
        st.subheader("🔍 步驟一：選擇比較母體與檢視視角")
        
        # 情境切換開關 (控制下方圖表要畫的輔助線與 KPI 邏輯)
        view_mode = st.radio(
            "請選擇您要如何詮釋這批數據：", 
            ["📊 訓練量視角 (關注總量，預設)", "⚡ 專項強度/比賽視角 (關注高強度跑動佔比)"], 
            horizontal=True
        )
        
        default_selections = player_sessions[-5:] if len(player_sessions) >= 5 else player_sessions
        selected_hist_sessions = st.multiselect(
            "請自由勾選做為比較母體的歷史事件 (將依此動態計算平均值與標準差)：", 
            player_sessions, 
            default=default_selections
        )

        if not selected_hist_sessions:
            st.info("請從上方選單中至少挑選一個歷史事件來進行分析。")
        else:
            # 準備 DataFrame 並統一 HSD 單位
            df_hist = df[(df['Player'] == selected_player) & (df['Session'].isin(selected_hist_sessions))].copy()
            df_hist['Session'] = pd.Categorical(df_hist['Session'], categories=selected_hist_sessions, ordered=True)
            df_hist = df_hist.sort_values('Session')
            df_hist['HSD Ratio (%)'] = (df_hist['HSD Ratio'] * 100).round(2)

            # 動態統計計算 (母體 = 勾選出來的事件)
            mean_dist = df_hist['Total Distance (m)'].mean()
            std_dist = df_hist['Total Distance (m)'].std() if len(df_hist) > 1 else 0
            
            mean_hsd = df_hist['HSD Ratio (%)'].mean()
            std_hsd = df_hist['HSD Ratio (%)'].std() if len(df_hist) > 1 else 0
            
            # 定義基準線變數
            vol_upper_bound = mean_dist + 2 * std_dist
            int_upper_bound = mean_hsd + 1 * std_hsd
            pr_top_speed = df_hist['Top Speed (m/s)'].max()
            mean_avg_speed = df_hist['Avg Speed (m/min)'].mean()

            # 抓取最新一筆 (最後勾選) 的數據進行 KPI 判定
            latest_record = df_hist.iloc[-1]
            latest_session_name = latest_record['Session']
            
            # ------------------------------------------
            # 💡 2. 客觀指標卡片 (KPI Cards)
            # ------------------------------------------
            st.write("<br>", unsafe_allow_html=True)
            st.markdown(f"#### 🔎 {latest_session_name} 的客觀狀態標籤")
            
            if "訓練量視角" in view_mode:
                latest_dist = latest_record['Total Distance (m)']
                diff_dist = latest_dist - mean_dist
                is_vol_extreme = (std_dist > 0) and (latest_dist > vol_upper_bound)
                
                label_color = "#1f77b4" if not is_vol_extreme else "#2c3e50" 
                label_text = f"總距離：{latest_dist:.0f} m"
                sub_text = f"與所選平均差：{'+' if diff_dist>0 else ''}{diff_dist:.0f} m"
                
                if is_vol_extreme:
                    st.markdown(f"<div style='background-color: #e8f4f8; padding: 15px; border-left: 5px solid {label_color}; border-radius: 5px;'>"
                                f"<h4 style='color: {label_color}; margin:0;'>[ 極端訓練量 ( > +2 SD ) ]</h4>"
                                f"<p style='margin:5px 0 0 0; font-size:16px;'>本期{label_text}。{sub_text}，已超出常態分佈範圍。</p></div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='background-color: #f8f9fa; padding: 15px; border-left: 5px solid #808080; border-radius: 5px;'>"
                                f"<h4 style='color: #808080; margin:0;'>[ 常態訓練量水位 ]</h4>"
                                f"<p style='margin:5px 0 0 0; font-size:16px;'>本期{label_text}。{sub_text}，落於合理分佈區間。</p></div>", unsafe_allow_html=True)

            else:
                latest_hsd = latest_record['HSD Ratio (%)']
                diff_hsd = latest_hsd - mean_hsd
                is_int_extreme = (std_hsd > 0) and (latest_hsd > int_upper_bound)
                
                label_color = "#d35400" if not is_int_extreme else "#c0392b"
                label_text = f"HSD 佔比：{latest_hsd:.1f}%"
                sub_text = f"與所選平均差：{'+' if diff_hsd>0 else ''}{diff_hsd:.1f}%"
                
                if is_int_extreme:
                    st.markdown(f"<div style='background-color: #fdedec; padding: 15px; border-left: 5px solid {label_color}; border-radius: 5px;'>"
                                f"<h4 style='color: {label_color}; margin:0;'>[ 高強度負荷 ( > +1 SD ) ]</h4>"
                                f"<p style='margin:5px 0 0 0; font-size:16px;'>本期{label_text}。{sub_text}，顯示經歷了高於常態的神經與無氧消耗。</p></div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='background-color: #f8f9fa; padding: 15px; border-left: 5px solid #808080; border-radius: 5px;'>"
                                f"<h4 style='color: #808080; margin:0;'>[ 常態專項強度 ]</h4>"
                                f"<p style='margin:5px 0 0 0; font-size:16px;'>本期{label_text}。{sub_text}，無顯著強度極端值。</p></div>", unsafe_allow_html=True)

            # ------------------------------------------
            # 📈 3. 縱向趨勢圖表 (帶有動態統計線)
            # ------------------------------------------
            st.write("<br>", unsafe_allow_html=True)
            st.subheader(f"📈 步驟二：{selected_player} 統計趨勢分析")
            
            col_t1, col_t2 = st.columns(2)
            
            def create_trend_chart(metric_col, title, color, ref_line=None, ref_label="", ref_color="gold"):
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=df_hist['Session'], y=df_hist[metric_col],
                    text=df_hist[metric_col].round(1) if 'Distance' not in metric_col else df_hist[metric_col].astype(int),
                    textposition='auto', marker_color=color, name=selected_player
                ))
                if ref_line is not None and pd.notna(ref_line) and ref_line > 0:
                    fig.add_hline(y=ref_line, line_width=3, line_dash="dash", line_color=ref_color, annotation_text=ref_label, annotation_position="top right", annotation_font_size=GLOBAL_FONT_SIZE)
                
                fig.update_layout(title=dict(text=f"<b>{title}</b>", font=dict(size=TITLE_FONT_SIZE)), margin=dict(t=40, b=20), height=350, showlegend=False)
                return apply_chart_style(fig)

            with col_t1:
                # 總跑動距離：畫上 +2SD 的輔助線 (藍線)
                sd2_line = vol_upper_bound if "訓練量視角" in view_mode and std_dist > 0 else mean_dist
                sd2_label = "μ + 2σ (極端值)" if "訓練量視角" in view_mode and std_dist > 0 else "μ (歷史平均)"
                fig_dist = create_trend_chart('Total Distance (m)', '總跑動距離 (Volume)', '#4a86e8', ref_line=sd2_line, ref_label=sd2_label, ref_color="#1f77b4")
                st.plotly_chart(fig_dist, use_container_width=True, config=PLOTLY_CONFIG)
                
                # 最高極速：畫上 PR 個人最佳紀錄線 (橘線)
                fig_top = create_trend_chart('Top Speed (m/s)', '最高極速表現 (Intensity/Power)', '#f6b26b', ref_line=pr_top_speed, ref_label="個人最佳 (PR)", ref_color="#e67e22")
                st.plotly_chart(fig_top, use_container_width=True, config=PLOTLY_CONFIG)

            with col_t2:
                # 平均移動速度：畫上歷史平均線 (紫線)
                fig_avg = create_trend_chart('Avg Speed (m/min)', '平均移動速度 (Intensity/Work Rate)', '#8e7cc3', ref_line=mean_avg_speed, ref_label="μ (歷史平均)", ref_color="#8e44ad")
                st.plotly_chart(fig_avg, use_container_width=True, config=PLOTLY_CONFIG)
                
                # HSD 比例：畫上 +1SD 的輔助線 (紅線)
                sd1_line = int_upper_bound if "專項強度" in view_mode and std_hsd > 0 else mean_hsd
                sd1_label = "μ + 1σ (高強度)" if "專項強度" in view_mode and std_hsd > 0 else "μ (歷史平均)"
                fig_hsd = create_trend_chart('HSD Ratio (%)', '高強度跑動佔比 (Explosiveness)', '#93c47d', ref_line=sd1_line, ref_label=sd1_label, ref_color="#c0392b")
                st.plotly_chart(fig_hsd, use_container_width=True, config=PLOTLY_CONFIG)

            # ------------------------------------------
            # 📋 4. 課表設計專用數據表
            # ------------------------------------------
            st.write("<br>", unsafe_allow_html=True)
            st.subheader("📋 步驟三：課表設計數據參考中心 (Programming Data)")
            st.markdown("您可以直接參考最下方的 **最大值 (PR)** 與 **平均值 (AVG)**，作為設定下週訓練配速、衝刺標竿或體能總量的實務基準。")
            
            table_cols = ['Date', 'Session', 'Total Distance (m)', 'Avg Speed (m/min)', 'Top Speed (m/s)', 'HSD Ratio (%)']
            if 'RPE' in df.columns:
                table_cols.append('RPE')
                
            df_table = df_hist[table_cols].copy()
            
            numeric_cols = [col for col in table_cols if col not in ['Date', 'Session']]
            max_vals = df_table[numeric_cols].max()
            avg_vals = df_table[numeric_cols].mean()
            
            summary_data = []
            summary_data.append({'Date': '---', 'Session': '🏆 所選事件最大值 (MAX/PR)'})
            summary_data.append({'Date': '---', 'Session': '📊 所選事件平均值 (AVG/μ)'})
            df_summary = pd.DataFrame(summary_data)
            
            for col in numeric_cols:
                df_summary.loc[0, col] = max_vals[col]
                df_summary.loc[1, col] = avg_vals[col]
                
            df_table['Total Distance (m)'] = df_table['Total Distance (m)'].round(0).astype(int)
            df_summary['Total Distance (m)'] = df_summary['Total Distance (m)'].round(0).astype(int)
            for col in ['Avg Speed (m/min)', 'Top Speed (m/s)', 'HSD Ratio (%)', 'RPE']:
                if col in df_table.columns:
                    df_table[col] = df_table[col].round(2)
                    df_summary[col] = df_summary[col].round(2)

            final_table = pd.concat([df_table, df_summary], ignore_index=True)
            st.dataframe(final_table, use_container_width=True, hide_index=True)