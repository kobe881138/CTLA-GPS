import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import plotly.graph_objects as go
import os
import shutil

# ==========================================
# 🌟 終極防破圖系統：暴力清快取 + 絕對路徑字體
# ==========================================
cache_dir = mpl.get_cachedir()
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir, ignore_errors=True)

current_dir = os.path.dirname(os.path.abspath(__file__))
font_path = os.path.join(current_dir, "NotoSansTC-Regular.ttf")

if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = prop.get_name()
    plt.rcParams['font.sans-serif'] = [prop.get_name(), 'sans-serif']
else:
    st.warning("⚠️ 找不到 NotoSansTC-Regular.ttf 字體檔！請確認已上傳至 GitHub。目前暫時使用系統備用字體。")
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Arial Unicode MS', 'sans-serif']

plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 🌟 澳洲隊基準數據
# ==========================================
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

# ==========================================
# 🌟 智慧階梯算法 (2k, 4k, 6k, 8k, 10k, 20k...)
# ==========================================
def get_dist_ymax(max_val):
    if pd.isna(max_val) or max_val <= 0: return 2000
    if max_val <= 2000: return 2000
    elif max_val <= 4000: return 4000
    elif max_val <= 6000: return 6000
    elif max_val <= 8000: return 8000
    elif max_val <= 10000: return 10000
    else: return (int(max_val) // 10000 + 1) * 10000

st.set_page_config(page_title="球隊 GPS 數據儀表板", layout="wide")

@st.cache_data
def load_data(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

df = load_data('Cleaned_GPS_Data.csv')

if df is None:
    st.error("❌ 找不到資料！請確認 Cleaned_GPS_Data.csv 是否存在。")
else:
    if 'Zone 4 Ratio' in df.columns: df.rename(columns={'Zone 4 Ratio': 'HSD Ratio'}, inplace=True)
    if 'Zone 4 Distance (m)' in df.columns: df.rename(columns={'Zone 4 Distance (m)': 'HSD (m)'}, inplace=True)
    
    df = df[~df['Player'].astype(str).str.contains('#')]
    df['Date'] = df['Session'].astype(str).apply(lambda x: x.split()[0])
    
    def get_month(date_str):
        try:
            return int(str(date_str).split('/')[0])
        except:
            return 0
    df['Month'] = df['Date'].apply(get_month)

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
        if 'Position' in daily_totals.columns: agg_funcs['Position'] = 'first'
        
        agg = daily_totals.groupby('Player').agg(agg_funcs).reset_index()
        
        # 🌟 將 RPE 限制為小數點後 1 位，畫面更乾淨
        if 'RPE' in agg.columns:
            agg['RPE'] = agg['RPE'].round(1)
            
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
    # 模式一：團隊總覽 (Team Dashboard)
    # ==========================================
    if page_mode == "📊 團隊總覽 (Team Dashboard)":
        st.title("🥍 男網模擬賽 GPS 戰情室 - 團隊總覽")
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
            if 'Position' in df_filtered.columns: agg_dict['Position'] = 'first'
            df_plot = df_filtered.groupby('Player').agg(agg_dict).reset_index()

            st.subheader(f"1️⃣ {selected_session} 外部與內部負荷")
            fig1, ax1 = plt.subplots(figsize=(12, 3.5))
            bars1 = ax1.bar(df_plot['Player'], df_plot['Total Distance (m)'], color='#4a86e8', width=0.5)
            
            team_avg_dist = df_plot['Total Distance (m)'].mean()
            if pd.notna(team_avg_dist):
                ax1.axhline(y=team_avg_dist, color='#e06666', linestyle='--', label='Team Avg')
            
            for bar in bars1:
                yval = bar.get_height()
                if pd.notna(yval) and yval > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2, yval/2 + 200, int(yval), ha='center', va='center', color='white', fontweight='bold', fontsize=12)
                    if 'RPE' in df_plot.columns:
                        rpe_val = df_plot.loc[df_plot['Total Distance (m)'] == yval, 'RPE'].values[0]
                        if pd.notna(rpe_val) and rpe_val > 0:
                            ax1.text(bar.get_x() + bar.get_width()/2, yval/2 - 200, f"RPE: {rpe_val}", ha='center', va='center', color='#ffd966', fontweight='bold', fontsize=11)
            
            ax1.margins(x=0.05)
            
            # 🎯 級距邏輯：套用自訂的 get_dist_ymax
            ax1.set_ylim(0, get_dist_ymax(df_plot['Total Distance (m)'].max()))
            
            ax1.legend()
            st.pyplot(fig1)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("2️⃣ 平均速度表現 (vs. Australia)")
                spd_mode = st.radio("顯示模式：", ["📌 當前時段", "📅 多日比較 (最多5天)"], horizontal=True, key='spd_mode')
                
                if spd_mode == "📌 當前時段":
                    fig2, ax2 = plt.subplots(figsize=(6, 4))
                    bars2 = ax2.bar(df_plot['Player'], df_plot['Avg Speed (m/min)'], color='#8e7cc3', width=0.5)
                    ax2.axhline(y=AUS_AVG_SPEED, color='gold', linestyle='-', linewidth=2, label='Australia SL')
                    
                    team_avg_spd = df_plot['Avg Speed (m/min)'].mean()
                    if pd.notna(team_avg_spd):
                        ax2.axhline(y=team_avg_spd, color='red', linestyle='--', alpha=0.5, label='Team Avg')
                    
                    ax2.margins(x=0.1)
                    
                    max_spd = df_plot['Avg Speed (m/min)'].max()
                    max_spd = max(max_spd, AUS_AVG_SPEED) if pd.notna(max_spd) else AUS_AVG_SPEED
                    y_max_spd = max(100, (int(max_spd) // 20 + 1) * 20)
                    ax2.set_ylim(0, y_max_spd)
                    
                    ax2.legend(loc='lower right')
                    st.pyplot(fig2)
                else:
                    valid_dates = [d for d in df['Date'].unique() if '/' in str(d) and d not in custom_and_auto_names]
                    default_d = selected_date if selected_date in valid_dates else valid_dates[-1] if valid_dates else None
                    selected_spd_dates = st.multiselect("選擇欲比較的日期 (最多5天)：", valid_dates, default=[default_d] if default_d else [], max_selections=5, key='spd_multi')
                    
                    if selected_spd_dates:
                        df_spd = df[(df['Date'].isin(selected_spd_dates)) & (df['Session'].astype(str).str.contains('Total|total', case=False, na=False))]
                        if not df_spd.empty:
                            players_spd = sorted(df_spd['Player'].unique())
                            fig2, ax2 = plt.subplots(figsize=(6, 4))
                            x = np.arange(len(players_spd))
                            width = 0.8 / len(selected_spd_dates)
                            colors_spd = ['#8e7cc3', '#c27ba0', '#e06666', '#f6b26b', '#ffd966']
                            
                            ax2.axhline(y=AUS_AVG_SPEED, color='gold', linestyle='-', linewidth=2, label='Australia SL')
                            
                            for i, d_date in enumerate(selected_spd_dates):
                                d_data = df_spd[df_spd['Date'] == d_date]
                                y_vals = [d_data[d_data['Player'] == p]['Avg Speed (m/min)'].max() if not d_data[d_data['Player'] == p].empty else 0 for p in players_spd]
                                offset = i * width - (0.8/2) + (width/2)
                                ax2.bar(x + offset, y_vals, width, label=f"{d_date}", color=colors_spd[i%len(colors_spd)])
                            
                            ax2.set_xticks(x)
                            ax2.set_xticklabels(players_spd)
                            ax2.margins(x=0.05)
                            
                            max_spd = df_spd['Avg Speed (m/min)'].max()
                            max_spd = max(max_spd, AUS_AVG_SPEED) if pd.notna(max_spd) else AUS_AVG_SPEED
                            y_max_spd = max(100, (int(max_spd) // 20 + 1) * 20)
                            ax2.set_ylim(0, y_max_spd)
                            
                            ax2.legend(loc='lower right', fontsize='small')
                            st.pyplot(fig2)
                        else:
                            st.info("💡 找不到所選日期的 Total 數據來進行比較。")
                    else:
                        st.info("💡 請至少選擇一個日期。")

            with col2:
                is_custom_or_auto = selected_date in custom_and_auto_names
                if is_custom_or_auto:
                    st.subheader(f"3️⃣ {selected_date} 每日負荷消長")
                    if selected_date in st.session_state['custom_periods']:
                        target_dates = st.session_state['custom_periods'][selected_date]
                    elif selected_date == 'Q1 (1-3月)':
                        target_dates = df[df['Month'].isin([1, 2, 3])]['Date'].unique().tolist()
                    elif '月份' in selected_date:
                        m = int(selected_date.replace('月份', ''))
                        target_dates = df[df['Month'] == m]['Date'].unique().tolist()
                    else:
                        target_dates = []
                        
                    target_dates = [d for d in target_dates if d not in custom_and_auto_names and '/' in str(d)]
                    df_q = df[(df['Date'].isin(target_dates)) & (df['Session'].astype(str).str.contains('Total|total', case=False, na=False))]
                    
                    if not df_q.empty:
                        daily_sessions = sorted(df_q['Date'].unique().tolist())
                        players = sorted(df_q['Player'].unique())
                        fig3_q, ax3_q = plt.subplots(figsize=(6, 4))
                        x = np.arange(len(players))
                        width = 0.8 / len(daily_sessions) if len(daily_sessions) > 0 else 0.8
                        colors = ['#6fa8dc', '#f6b26b', '#93c47d', '#ffd966', '#c27ba0', '#8e7cc3']
                        
                        for i, d_date in enumerate(daily_sessions):
                            d_data = df_q[df_q['Date'] == d_date]
                            y_vals = [d_data[d_data['Player'] == p]['Total Distance (m)'].max() if not d_data[d_data['Player'] == p].empty else 0 for p in players]
                            offset = i * width - (0.8/2) + (width/2)
                            ax3_q.bar(x + offset, y_vals, width, label=f"{d_date}", color=colors[i%len(colors)])
                            
                        team_avg_q_dist = df_q['Total Distance (m)'].mean()
                        if pd.notna(team_avg_q_dist):
                            ax3_q.axhline(team_avg_q_dist, color='#e06666', linestyle='--', label='Period Daily Avg')
                            
                        ax3_q.set_xticks(x)
                        ax3_q.set_xticklabels(players)
                        ax3_q.margins(x=0.05)
                        
                        # 🎯 級距邏輯
                        ax3_q.set_ylim(0, get_dist_ymax(df_q['Total Distance (m)'].max()))
                        
                        ax3_q.legend(loc='upper right', fontsize='small')
                        st.pyplot(fig3_q)
                    else:
                        st.info("💡 此週期內找不到每日的 Total 資料來進行拆解。")
                        
                else:
                    st.subheader("3️⃣ 單節/分段 體能維持率")
                    is_training = 'training' in selected_session.lower()
                    if is_training:
                        quarter_sessions = [s for s in sessions_for_date if 'training' in str(s).lower() and str(s).split()[-1].isdigit()]
                    else:
                        quarter_sessions = [s for s in sessions_for_date if 'training' not in str(s).lower() and str(s).split()[-1].isdigit()]

                    quarter_sessions = sorted(quarter_sessions)

                    if len(quarter_sessions) > 0:
                        df_q = df[df['Session'].isin(quarter_sessions)]
                        players = sorted(df_q['Player'].unique())
                        fig3_q, ax3_q = plt.subplots(figsize=(6, 4))
                        x = np.arange(len(players))
                        width = 0.8 / len(quarter_sessions)
                        colors = ['#6fa8dc', '#f6b26b', '#93c47d', '#ffd966']
                        
                        for i, q_sess in enumerate(quarter_sessions):
                            q_data = df_q[df_q['Session'] == q_sess]
                            y_vals = [q_data[q_data['Player'] == p]['Total Distance (m)'].max() if not q_data[q_data['Player'] == p].empty else 0 for p in players]
                            offset = i * width - (0.8/2) + (width/2)
                            ax3_q.bar(x + offset, y_vals, width, label=f"{q_sess}", color=colors[i%len(colors)])
                            
                        team_avg_q_dist = df_q['Total Distance (m)'].mean()
                        if pd.notna(team_avg_q_dist):
                            ax3_q.axhline(team_avg_q_dist, color='#e06666', linestyle='--', label='Session Avg')
                            
                        ax3_q.set_xticks(x)
                        ax3_q.set_xticklabels(players)
                        ax3_q.margins(x=0.05)
                        
                        # 🎯 級距邏輯：套用智慧縮放
                        ax3_q.set_ylim(0, get_dist_ymax(df_q['Total Distance (m)'].max()))
                        
                        ax3_q.legend(loc='upper right', fontsize='small')
                        st.pyplot(fig3_q)
                    else:
                        st.info("💡 此時段無單節資料或為單日加總資料。")

            st.write("<br>", unsafe_allow_html=True)
            st.subheader("4️⃣ 爆發力象限圖 (Plotly 互動版)")
            spacer1, col_center, spacer2 = st.columns([1, 4, 1])
            with col_center:
                x_data = df_plot['HSD Ratio'] * 100
                y_data = df_plot['Top Speed (m/s)']
                session_avg_hsd = x_data.mean()
                session_avg_top = y_data.mean()
                
                fig4 = go.Figure()
                fig4.add_trace(go.Scatter(
                    x=x_data, y=y_data, mode='markers+text',
                    text=df_plot['Player'], textposition="top center",
                    marker=dict(color='#3d85c6', size=12, line=dict(width=1, color='white')), name='Players',
                    hovertemplate='<b>%{text}</b><br>HSD Ratio: %{x:.1f}%<br>Top Speed: %{y:.1f} m/s<extra></extra>'
                ))

                if pd.notna(session_avg_hsd) and pd.notna(session_avg_top):
                    fig4.add_trace(go.Scatter(
                        x=[session_avg_hsd], y=[session_avg_top], mode='markers',
                        marker=dict(color='#38761d', symbol='cross', size=14), name='Session Avg',
                        hovertemplate='<b>團隊平均</b><br>HSD Ratio: %{x:.1f}%<br>Top Speed: %{y:.1f} m/s<extra></extra>'
                    ))
                    fig4.add_vline(x=session_avg_hsd, line_dash="dash", line_color="#38761d", opacity=0.5)
                    fig4.add_hline(y=session_avg_top, line_dash="dash", line_color="#38761d", opacity=0.5)

                fig4.add_trace(go.Scatter(
                    x=[AUS_HSD_RATIO], y=[AUS_TOP_SPEED], mode='markers',
                    marker=dict(color='red', symbol='star', size=18, line=dict(width=1, color='darkgray')), name=default_baseline_name,
                    hovertemplate=f'<b>{default_baseline_name}</b><br>HSD Ratio: %{{x:.1f}}%<br>Top Speed: %{{y:.1f}} m/s<extra></extra>'
                ))

                max_hsd_plot = max(x_data.max(), AUS_HSD_RATIO) if not x_data.empty else AUS_HSD_RATIO
                max_top_plot = max(y_data.max(), AUS_TOP_SPEED) if not y_data.empty else AUS_TOP_SPEED
                
                x_max_plot = max(20, (int(max_hsd_plot) // 10 + 1) * 10)
                y_max_plot = max(10, (int(max_top_plot) // 2 + 1) * 2)

                fig4.update_layout(
                    xaxis_title='<b>HSD Ratio (%)</b>', yaxis_title='<b>Top Speed (m/s)</b>',
                    xaxis=dict(range=[0, x_max_plot]), yaxis=dict(range=[0, y_max_plot]),
                    margin=dict(l=20, r=20, t=30, b=20), hovermode='closest',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(
                    fig4, 
                    use_container_width=True, 
                    config={
                        'editable': True,  
                        'displayModeBar': True,
                        'toImageButtonOptions': {'format': 'png', 'filename': 'Mens_GPS_Scatter', 'scale': 3} 
                    }
                )
        else:
            st.warning("此時段沒有數據喔！")

    # ==========================================
    # 模式二：個人專屬報告 (Player Profile - Total Focus)
    # ==========================================
    elif page_mode == "👤 個人報告 (Player Profile)":
        st.title("🥍 男網模擬賽 GPS 戰情室 - 個人報告")
        st.sidebar.header("👤 個人報告設定")
        
        all_players = sorted(df['Player'].unique().tolist())
        selected_player = st.sidebar.selectbox("🏃 選擇選手：", all_players)
        
        # 🌟 解鎖事件限制：現在可以選擇「任何 Session」，不再只限 Total！
        player_sessions = df[df['Player'] == selected_player]['Session'].dropna().unique().tolist()
        all_sessions = df['Session'].dropna().unique().tolist()
        
        # 建立自訂週期的名稱清單 (加上 Total 綴飾)
        custom_session_names = [f"{name} Total" for name in custom_and_auto_names]
        
        # 強制將自訂週期排在最上方
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
            # 取得位置
            raw_pos = str(df[df['Player'] == selected_player]['Position'].iloc[0])
            
            st.write("---")
            st.subheader(f"🛡️ {selected_player} (長條圖對標: {default_baseline_name}) - 個人表現分析報告")

            col_radar, col_bar = st.columns([1, 1.5])

            with col_radar:
                st.markdown(f"##### 📍 六角雷達圖：對標團隊平均")
                # 雷達圖也改為選擇 Session 事件
                radar_session = st.selectbox("📅 選擇雷達圖檢視事件：", player_sessions, index=0)
                
                team_radar_df = df[df['Session'] == radar_session]
                team_mean = team_radar_df[['Total Distance (m)', 'Avg Speed (m/min)', 'Top Speed (m/s)', 'HSD Ratio']].mean()
                team_std = team_radar_df[['Total Distance (m)', 'Avg Speed (m/min)', 'Top Speed (m/s)', 'HSD Ratio']].std().replace(0, 1).fillna(1)
                
                player_radar = df[(df['Player'] == selected_player) & (df['Session'] == radar_session)].iloc[0]

                categories = ['Total Distance', 'Average Speed', 'Max Speed', 'HSD Ratio']
                N = len(categories)
                
                def calc_z(col):
                    if pd.isna(player_radar[col]) or pd.isna(team_mean[col]): return 0
                    z = (player_radar[col] - team_mean[col]) / team_std[col]
                    return np.clip(z, -2, 2)
                    
                p_dist = calc_z('Total Distance (m)')
                p_avg_spd = calc_z('Avg Speed (m/min)')
                p_top_spd = calc_z('Top Speed (m/s)')
                p_hsd = calc_z('HSD Ratio')
                
                player_ratios = [p_dist, p_avg_spd, p_top_spd, p_hsd]
                player_ratios += player_ratios[:1] 
                team_ratios = [0, 0, 0, 0, 0] 
                
                angles = [n / float(N) * 2 * np.pi for n in range(N)]
                angles += angles[:1]

                fig_r, ax_r = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
                ax_r.set_theta_offset(np.pi / 2)
                ax_r.set_theta_direction(-1)
                ax_r.set_xticks(angles[:-1])
                ax_r.set_xticklabels(categories, fontsize=12, fontweight='bold')
                
                ax_r.set_ylim(-2, 2)
                ax_r.set_yticks([-2, -1, 0, 1, 2])
                ax_r.set_yticklabels(['-2', '-1', '0', '1', '2'], color="grey", size=9, alpha=0.7)
                
                ax_r.plot(angles, team_ratios, linewidth=2, linestyle='dashed', color='#e06666', label=f'{radar_session} Team Avg (0)')
                ax_r.fill(angles, team_ratios, color='#e06666', alpha=0.1)
                ax_r.plot(angles, player_ratios, linewidth=2.5, color='#4a86e8', label=f'{selected_player}')
                ax_r.fill(angles, player_ratios, color='#4a86e8', alpha=0.3)
                ax_r.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
                st.pyplot(fig_r)

            with col_bar:
                st.markdown("##### 📈 歷史進步軌跡")
                compare_mode = st.radio("📊 選擇比較模式：", ["雙期比較 (2個數據)", "三期比較 (3個數據)"], horizontal=True)
                
                baseline_options = [default_baseline_name] + all_sessions
                
                if compare_mode == "雙期比較 (2個數據)":
                    col_b1, col_b2 = st.columns(2)
                    with col_b1: 
                        player_selected_session = st.selectbox("📅 當前檢視事件 (Current)：", player_sessions)
                    with col_b2:
                        selected_baseline1 = st.selectbox("📉 比較基準 (Baseline)：", baseline_options)
                    selected_baseline2 = None
                else:
                    col_b1, col_b2, col_b3 = st.columns(3)
                    with col_b1: 
                        player_selected_session = st.selectbox("📅 當前檢視事件 (Current)：", player_sessions)
                    with col_b2:
                        selected_baseline1 = st.selectbox("📉 比較基準 1 (Baseline 1)：", baseline_options)
                    with col_b3:
                        default_b2_idx = 1 if len(baseline_options) > 1 else 0
                        selected_baseline2 = st.selectbox("📉 比較基準 2 (Baseline 2)：", baseline_options, index=default_b2_idx)

                player_current_bar = df[(df['Player'] == selected_player) & (df['Session'] == player_selected_session)].iloc[0]
                
                def get_baseline_data(b_name):
                    if b_name == default_baseline_name:
                        target = default_baseline_data
                        return {
                            'Total Distance (m)': target['dist'],
                            'Avg Speed (m/min)': target['avg_spd'],
                            'Top Speed (m/s)': target['top_spd'],
                            'HSD Ratio': target['hsd_ratio'] / 100 
                        }, "AUS Avg"
                    else:
                        past_data = df[(df['Player'] == selected_player) & (df['Session'] == b_name)]
                        if not past_data.empty:
                            return past_data[['Total Distance (m)', 'Avg Speed (m/min)', 'Top Speed (m/s)', 'HSD Ratio']].mean(), b_name
                        else:
                            return None, b_name

                b1_data, b1_label = get_baseline_data(selected_baseline1)
                b2_data, b2_label = None, None
                if selected_baseline2:
                    b2_data, b2_label = get_baseline_data(selected_baseline2)

                warnings = []
                if b1_data is None: warnings.append(f"💡 貼心提醒：{selected_player} 在 {selected_baseline1} 剛好沒有紀錄。")
                if selected_baseline2 and b2_data is None: warnings.append(f"💡 貼心提醒：{selected_player} 在 {selected_baseline2} 剛好沒有紀錄。")
                for w in warnings: st.info(w)

                fig_b, axes = plt.subplots(1, 4, figsize=(10, 4))
                
                metrics = [
                    ('Total Distance', 'Total Distance (m)', ['#c9daf8', '#6fa8dc', '#4a86e8']),
                    ('Average Speed', 'Avg Speed (m/min)', ['#ead1dc', '#d5a6bd', '#8e7cc3']),
                    ('Max Speed', 'Top Speed (m/s)', ['#fff2cc', '#fce5cd', '#f6b26b']),
                    ('HSD Ratio (%)', 'HSD Ratio', ['#eff5e1', '#d9ead3', '#93c47d'])
                ]
                
                # 🌟 淨化標籤名稱，如果文字太長自動換行，且移除 B1、Cur 等字眼
                def format_label(text):
                    return text.replace(' ', '\n', 1)

                for i, (title, col_name, color_palette) in enumerate(metrics):
                    plot_labels = []
                    plot_vals = []
                    plot_colors = []
                    
                    if b2_data is not None:
                        plot_labels.append(format_label(b2_label))
                        v = b2_data[col_name] if pd.notna(b2_data[col_name]) else 0
                        plot_vals.append(v * 100 if 'Ratio' in col_name and b2_label != "AUS Avg" else v)
                        plot_colors.append(color_palette[0]) 
                        
                    if b1_data is not None:
                        plot_labels.append(format_label(b1_label))
                        v = b1_data[col_name] if pd.notna(b1_data[col_name]) else 0
                        plot_vals.append(v * 100 if 'Ratio' in col_name and b1_label != "AUS Avg" else v)
                        plot_colors.append(color_palette[1] if b2_data is not None else color_palette[0])
                        
                    plot_labels.append(format_label(player_selected_session))
                    v = player_current_bar[col_name] if pd.notna(player_current_bar[col_name]) else 0
                    plot_vals.append(v * 100 if 'Ratio' in col_name else v)
                    plot_colors.append(color_palette[2]) 
                    
                    bars = axes[i].bar(plot_labels, plot_vals, color=plot_colors, width=0.6)
                    axes[i].set_title(title, fontweight='bold', fontsize=11)
                    axes[i].spines['top'].set_visible(False)
                    axes[i].spines['right'].set_visible(False)
                    
                    if plot_vals:
                        max_y = max(plot_vals)
                        if pd.notna(max_y) and max_y >= 0:
                            if 'Total Distance' in title:
                                axes[i].set_ylim(0, get_dist_ymax(max_y))
                            elif 'Average Speed' in title:
                                y_max = max(100, (int(max_y) // 20 + 1) * 20)
                                axes[i].set_ylim(0, y_max)
                            elif 'Max Speed' in title:
                                y_max = max(10, (int(max_y) // 2 + 1) * 2)
                                axes[i].set_ylim(0, y_max)
                            elif 'HSD Ratio' in title:
                                y_max = max(20, (int(max_y) // 10 + 1) * 10)
                                axes[i].set_ylim(0, y_max)
                    
                    for bar in bars:
                        yval = bar.get_height()
                        if pd.notna(yval) and yval > 0:
                            format_str = f"{int(yval)}" if 'Distance' in title else f"{yval:.1f}"
                            axes[i].text(bar.get_x() + bar.get_width()/2, yval + (yval*0.02), format_str, ha='center', va='bottom', fontweight='bold', fontsize=10)

                plt.tight_layout()
                st.pyplot(fig_b)