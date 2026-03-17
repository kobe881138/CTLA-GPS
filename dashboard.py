import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 澳洲隊基準數據
AUS_TOP_SPEED = 7.29      
AUS_AVG_SPEED = 117.36      
AUS_HSD_RATIO = 5.07       

st.set_page_config(page_title="球隊 GPS 數據儀表板", layout="wide")
plt.rcParams['font.sans-serif'] = ['Arial', 'sans-serif'] 
plt.rcParams['axes.unicode_minus'] = False

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

    df['Date'] = df['Session'].astype(str).apply(lambda x: x.split()[0])
    
    # ==========================================
    # 🌟 側邊欄主導航列
    # ==========================================
    st.sidebar.title("🥍 戰情室導覽")
    page_mode = st.sidebar.radio(
        "📌 選擇分析模式：", 
        ["📊 團隊總覽 (Team Dashboard)", "👤 個人報告 (Player Profile)"]
    )
    st.sidebar.markdown("---") 

    # ==========================================
    # 模式一：團隊總覽
    # ==========================================
    if page_mode == "📊 團隊總覽 (Team Dashboard)":
        st.title("🥍 模擬賽 GPS 戰情室 - 團隊總覽")
        
        st.sidebar.header("⚙️ 團隊設定面板")
        available_dates = df['Date'].dropna().unique().tolist()
        selected_date = st.sidebar.selectbox("📅 第一步：選擇日期", available_dates, key='team_date')
        
        sessions_for_date = df[df['Date'] == selected_date]['Session'].unique().tolist()
        selected_session = st.sidebar.selectbox("⏱️ 第二步：選擇時段", sessions_for_date, key='team_session')
        
        st.write("---")
        df_filtered = df[df['Session'] == selected_session]
        
        if not df_filtered.empty:
            agg_dict = {'Total Distance (m)': 'max', 'Avg Speed (m/min)': 'mean', 'Top Speed (m/s)': 'max', 'HSD Ratio': 'max'}
            if 'RPE' in df_filtered.columns: agg_dict['RPE'] = 'max'
            df_plot = df_filtered.groupby('Player').agg(agg_dict).reset_index()

            st.subheader(f"1️⃣ {selected_session} 外部與內部負荷")
            fig1, ax1 = plt.subplots(figsize=(12, 3.5))
            bars1 = ax1.bar(df_plot['Player'], df_plot['Total Distance (m)'], color='#4a86e8', width=0.5)
            ax1.axhline(y=df_plot['Total Distance (m)'].mean(), color='#e06666', linestyle='--', label='Team Avg')
            
            for bar in bars1:
                yval = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2, yval/2 + 200, int(yval), ha='center', va='center', color='white', fontweight='bold', fontsize=12)
                if 'RPE' in df_plot.columns:
                    rpe_val = df_plot.loc[df_plot['Total Distance (m)'] == yval, 'RPE'].values[0]
                    if rpe_val > 0:
                        ax1.text(bar.get_x() + bar.get_width()/2, yval/2 - 200, f"RPE: {rpe_val}", ha='center', va='center', color='#ffd966', fontweight='bold', fontsize=11)
            ax1.legend()
            st.pyplot(fig1)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("2️⃣ 平均速度表現 (vs. Australia)")
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                bars2 = ax2.bar(df_plot['Player'], df_plot['Avg Speed (m/min)'], color='#8e7cc3', width=0.5)
                ax2.axhline(y=AUS_AVG_SPEED, color='gold', linestyle='-', linewidth=2, label='Australia SL')
                ax2.axhline(y=df_plot['Avg Speed (m/min)'].mean(), color='red', linestyle='--', alpha=0.5, label='Team Avg')
                for bar in bars2:
                    yval = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}", ha='center', va='bottom', fontweight='bold')
                ax2.legend(loc='lower right')
                st.pyplot(fig2)

            with col2:
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
                        bars_q = ax3_q.bar(x + offset, y_vals, width, label=f"{q_sess}", color=colors[i%len(colors)])
                        for bar in bars_q:
                            h = bar.get_height()
                            if h > 0: ax3_q.text(bar.get_x() + bar.get_width()/2, h/2, int(h), ha='center', va='center', color='white', fontsize=10, fontweight='bold', rotation=90)
                    
                    ax3_q.axhline(df_q['Total Distance (m)'].mean(), color='#e06666', linestyle='--', label='Session Avg')
                    ax3_q.set_xticks(x)
                    ax3_q.set_xticklabels(players)
                    ax3_q.legend(loc='upper right', fontsize='small')
                    st.pyplot(fig3_q)
                else:
                    st.info("💡 此時段無單節資料。")

            st.write("<br>", unsafe_allow_html=True)
            st.subheader("4️⃣ 爆發力象限圖")
            spacer1, col_center, spacer2 = st.columns([1, 2, 1])
            with col_center:
                fig4, ax4 = plt.subplots(figsize=(8, 5))
                x_data = df_plot['HSD Ratio'] * 100
                y_data = df_plot['Top Speed (m/s)']
                session_avg_hsd = x_data.mean()
                session_avg_top = y_data.mean()
                
                ax4.scatter(x_data, y_data, color='#3d85c6', s=150, zorder=5, label='Players')
                for i, player in enumerate(df_plot['Player']):
                    ax4.text(x_data[i] + 0.1, y_data[i], player, fontsize=10, fontweight='bold', va='center')

                ax4.scatter(session_avg_hsd, session_avg_top, color='#38761d', marker='P', s=200, zorder=6, label='Session Avg')
                ax4.axvline(x=session_avg_hsd, color='#38761d', linestyle='--', alpha=0.5)
                ax4.axhline(y=session_avg_top, color='#38761d', linestyle='--', alpha=0.5)

                ax4.scatter(AUS_HSD_RATIO, AUS_TOP_SPEED, color='red', marker='*', s=250, zorder=10, label='Australia SL')
                ax4.set_xlabel('HSD Ratio (%)', fontweight='bold')
                ax4.set_ylabel('Top Speed (m/s)', fontweight='bold')
                ax4.legend(loc='upper left', bbox_to_anchor=(1, 1))
                st.pyplot(fig4)

        else:
            st.warning("此時段沒有數據喔！")

    # ==========================================
    # 模式二：個人專屬報告 (Player Profile - Total Focus)
    # ==========================================
    elif page_mode == "👤 個人報告 (Player Profile)":
        st.title("🥍 模擬賽 GPS 戰情室 - 個人報告")
        
        # 側邊欄現在只剩下選擇選手
        st.sidebar.header("👤 個人報告設定")
        
        all_players = sorted(df['Player'].unique().tolist())
        selected_player = st.sidebar.selectbox("🏃 選擇選手：", all_players)
        
        df_total_only = df[df['Session'].str.lower().str.contains('total')]
        player_dates_with_total = df_total_only[df_total_only['Player'] == selected_player]['Date'].dropna().unique().tolist()
        
        if not player_dates_with_total:
            st.warning(f"💡 找不到 {selected_player} 的 Total 加總數據。")
        else:
            st.write("---")
            st.subheader(f"🛡️ {selected_player} 個人表現分析報告")

            col_radar, col_bar = st.columns([1, 1.5])

            # 🎯 圖一：雷達圖區塊
            with col_radar:
                st.markdown("##### 📍 六角雷達圖")
                
                # 雷達圖的專屬日期設定 (預設抓該選手的第一筆資料)
                radar_date = st.selectbox(
                    "📅 選擇雷達圖檢視日期：", 
                    player_dates_with_total, 
                    index=0,
                    key='radar_date'
                )
                
                team_radar_df = df_total_only[df_total_only['Date'] == radar_date]
                team_mean = team_radar_df[['Total Distance (m)', 'Avg Speed (m/min)', 'Top Speed (m/s)', 'HSD Ratio']].mean()
                team_std = team_radar_df[['Total Distance (m)', 'Avg Speed (m/min)', 'Top Speed (m/s)', 'HSD Ratio']].std()
                team_std = team_std.replace(0, 1).fillna(1)
                
                player_radar = df_total_only[(df_total_only['Player'] == selected_player) & (df_total_only['Date'] == radar_date)].iloc[0]

                categories = ['Total Distance', 'Average Speed', 'Max Speed', 'HSD Ratio']
                N = len(categories)
                
                def calc_z(col):
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
                
                ax_r.plot(angles, team_ratios, linewidth=2, linestyle='dashed', color='#e06666', label='Team Avg (0)')
                ax_r.fill(angles, team_ratios, color='#e06666', alpha=0.1)
                
                ax_r.plot(angles, player_ratios, linewidth=2.5, color='#4a86e8', label=f'{selected_player}')
                ax_r.fill(angles, player_ratios, color='#4a86e8', alpha=0.3)

                ax_r.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
                st.pyplot(fig_r)

            # 📊 圖二：歷史長條圖區塊
            with col_bar:
                st.markdown("##### 📈 歷史進步軌跡")
                
                # 🌟 將下拉選單移進來，並使用雙欄位並排，視覺更緊湊
                col_b1, col_b2 = st.columns(2)
                with col_b1:
                    player_selected_date = st.selectbox("📅 當前表現 (Current)：", player_dates_with_total, key='player_date')
                with col_b2:
                    baseline_options = ["2025 Avg (11/30, 12/14, 12/28)"]
                    all_total_dates = df_total_only['Date'].dropna().unique().tolist()
                    baseline_options.extend(all_total_dates) 
                    selected_baseline = st.selectbox("📉 比較基準 (Baseline)：", baseline_options)
                
                # 抓取並運算長條圖資料
                player_current_bar = df_total_only[(df_total_only['Player'] == selected_player) & (df_total_only['Date'] == player_selected_date)].iloc[0]
                
                if selected_baseline.startswith("2025"):
                    past_data = df_total_only[(df_total_only['Player'] == selected_player) & (df_total_only['Date'].isin(['11/30', '12/14', '12/28']))]
                    baseline_label = "2025 Avg"
                else:
                    past_data = df_total_only[(df_total_only['Player'] == selected_player) & (df_total_only['Date'] == selected_baseline)]
                    baseline_label = f"{selected_baseline} Total"
                
                if not past_data.empty:
                    past_avg = past_data[['Total Distance (m)', 'Avg Speed (m/min)', 'Top Speed (m/s)', 'HSD Ratio']].mean()
                    
                    fig_b, axes = plt.subplots(1, 4, figsize=(10, 4))
                    metrics = [
                        ('Total Distance', 'Total Distance (m)', '#4a86e8', '#a4c2f4'),
                        ('Average Speed', 'Avg Speed (m/min)', '#8e7cc3', '#d9d2e9'),
                        ('Max Speed', 'Top Speed (m/s)', '#f6b26b', '#fce5cd'),
                        ('HSD Ratio (%)', 'HSD Ratio', '#93c47d', '#d9ead3')
                    ]
                    
                    labels = [baseline_label, 'Current']
                    
                    for i, (title, col_name, color_curr, color_past) in enumerate(metrics):
                        val_past = past_avg[col_name]
                        val_curr = player_current_bar[col_name]
                        
                        if 'Ratio' in col_name:
                            val_past *= 100
                            val_curr *= 100
                            
                        bars = axes[i].bar(labels, [val_past, val_curr], color=[color_past, color_curr], width=0.6)
                        axes[i].set_title(title, fontweight='bold', fontsize=11)
                        axes[i].spines['top'].set_visible(False)
                        axes[i].spines['right'].set_visible(False)
                        
                        for bar in bars:
                            yval = bar.get_height()
                            format_str = f"{int(yval)}" if 'Distance' in title else f"{yval:.1f}"
                            axes[i].text(bar.get_x() + bar.get_width()/2, yval + (yval*0.02), format_str, ha='center', va='bottom', fontweight='bold', fontsize=10)

                    plt.tight_layout()
                    st.pyplot(fig_b)
                else:
                    st.info(f"💡 {selected_player} 在你選擇的基準「{selected_baseline}」中沒有 Total 數據可供比較。")