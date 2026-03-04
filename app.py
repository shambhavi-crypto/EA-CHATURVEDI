"""
Nikhil's Software Development Company — Workforce Intelligence Dashboard
Classification + Clustering + Association Rule Mining
Descriptive → Diagnostic → Predictive → Prescriptive
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIG & CUSTOM CSS
# ============================================================
st.set_page_config(page_title="Nikhil's Workforce Intelligence", layout="wide", page_icon="🏢")

COLORS = {
    'primary': '#6ee7b7',       # Emerald
    'secondary': '#a78bfa',     # Violet
    'accent': '#fbbf24',        # Amber
    'danger': '#f87171',        # Red
    'safe': '#34d399',          # Green
    'info': '#60a5fa',          # Blue
    'bg_card': 'rgba(30,41,59,0.7)',
    'bg_dark': '#0f172a',
    'text': '#e2e8f0',
    'grid': 'rgba(99,102,241,0.06)',
}
SKILL_COLORS = {'Python': '#3776ab', 'English': '#e74c3c', 'French': '#1e3a5f'}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Mono:wght@400;700&display=swap');
html, body, .stApp { background: #0f172a; color: #e2e8f0; font-family: 'DM Sans', sans-serif; }
.block-container { max-width: 1280px; padding: 1rem 2rem; }
h1, h2, h3 { font-family: 'Space Mono', monospace !important; }
.main-header {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 50%, #1e1b4b 100%);
    border: 1px solid rgba(110, 231, 183, 0.15);
    border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 1.5rem;
    box-shadow: 0 0 40px rgba(110, 231, 183, 0.05);
}
.main-header h1 { color: #6ee7b7; font-size: 1.8rem; margin: 0; letter-spacing: -0.5px; }
.main-header p { color: #94a3b8; margin: 0.3rem 0 0; font-size: 0.95rem; }
.kpi-card {
    background: rgba(30,41,59,0.8); border: 1px solid rgba(110,231,183,0.1);
    border-radius: 12px; padding: 1.2rem; text-align: center;
}
.kpi-card .value { font-family: 'Space Mono', monospace; font-size: 2rem; font-weight: 700; }
.kpi-card .label { color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; margin-top: 0.3rem; }
.story-box {
    background: rgba(30,41,59,0.6); border-left: 4px solid #6ee7b7;
    border-radius: 0 12px 12px 0; padding: 1.2rem 1.5rem; margin: 1rem 0;
}
.insight-box {
    background: rgba(251,191,36,0.08); border-left: 4px solid #fbbf24;
    border-radius: 0 10px 10px 0; padding: 1rem 1.2rem; margin: 0.8rem 0;
    color: #fde68a; font-size: 0.9rem;
}
.danger-box {
    background: rgba(248,113,113,0.08); border-left: 4px solid #f87171;
    border-radius: 0 10px 10px 0; padding: 1rem 1.2rem; margin: 0.8rem 0;
    color: #fca5a5;
}
.rec-card-high { background: rgba(248,113,113,0.1); border: 1px solid rgba(248,113,113,0.3); border-radius: 12px; padding: 1.2rem; margin: 0.5rem 0; }
.rec-card-med { background: rgba(251,191,36,0.1); border: 1px solid rgba(251,191,36,0.3); border-radius: 12px; padding: 1.2rem; margin: 0.5rem 0; }
.rec-card-low { background: rgba(110,231,183,0.1); border: 1px solid rgba(110,231,183,0.3); border-radius: 12px; padding: 1.2rem; margin: 0.5rem 0; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] {
    background: rgba(30,41,59,0.6); border-radius: 8px 8px 0 0;
    color: #94a3b8; padding: 0.6rem 1.2rem; font-family: 'Space Mono', monospace; font-size: 0.78rem;
}
.stTabs [aria-selected="true"] { background: rgba(110,231,183,0.15); color: #6ee7b7 !important; }
div[data-testid="stSidebar"] { background: #1e293b; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv('nikhil_company_data.csv')
    df['RequiredSkillsCount'] = df['Skill_Python'] + df['Lang_English'] + df['Lang_French']
    df['TotalTechSkills'] = df['Skill_Python'] + df['Skill_Java'] + df['Skill_SQL'] + df['Skill_JavaScript'] + df['Skill_CSharp']
    df['TotalLanguages'] = df['Lang_English'] + df['Lang_French'] + df['Lang_Hindi']
    df['Attrition_Flag'] = (df['Attrition_Risk_3Months'] == 'Yes').astype(int)
    # Composite trainability score
    df['TrainabilityScore'] = (
        df['LearningAttitude'] * 0.25 +
        df['ManagerFeedback'] * 0.20 +
        df['PerformanceRating'] * 2.0 * 0.20 +
        df['SkillsUpgradedLastYear'] * 2.0 * 0.15 +
        df['TrainingHoursLastYear'] / 12.0 * 0.10 +
        df['TeamCollaborationScore'] * 0.10
    ).round(2)
    return df

df = load_data()
triple_skilled = df[df['RequiredSkillsCount'] == 3]
remaining_97 = df[df['RequiredSkillsCount'] < 3]

# ============================================================
# PLOTLY THEME
# ============================================================
def apply_theme(fig, height=420):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='DM Sans', color='#e2e8f0', size=12),
        margin=dict(l=20, r=20, t=50, b=20), height=height,
        xaxis=dict(gridcolor=COLORS['grid'], zeroline=False),
        yaxis=dict(gridcolor=COLORS['grid'], zeroline=False),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=11)),
    )
    return fig

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>🏢 Nikhil's Workforce Intelligence Suite</h1>
    <p>COVID-era survival analytics — Classification · Clustering · Association Rule Mining</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# KPIs
# ============================================================
k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    st.markdown(f'<div class="kpi-card"><div class="value" style="color:#6ee7b7">{len(df)}</div><div class="label">Total Employees</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="kpi-card"><div class="value" style="color:#f87171">{len(triple_skilled)}</div><div class="label">Have All 3 Skills</div></div>', unsafe_allow_html=True)
with k3:
    gap = 30 - len(triple_skilled)
    st.markdown(f'<div class="kpi-card"><div class="value" style="color:#fbbf24">{gap}</div><div class="label">Skill Gap (Need 30)</div></div>', unsafe_allow_html=True)
with k4:
    at_risk = len(df[df['Attrition_Risk_3Months']=='Yes'])
    st.markdown(f'<div class="kpi-card"><div class="value" style="color:#f87171">{at_risk}</div><div class="label">Attrition Risk</div></div>', unsafe_allow_html=True)
with k5:
    has_one = len(remaining_97[remaining_97['RequiredSkillsCount'] >= 1])
    st.markdown(f'<div class="kpi-card"><div class="value" style="color:#a78bfa">{has_one}</div><div class="label">Have ≥1 Skill (of 97)</div></div>', unsafe_allow_html=True)
with k6:
    st.markdown(f'<div class="kpi-card"><div class="value" style="color:#60a5fa">37</div><div class="label">Need to Select</div></div>', unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================
tabs = st.tabs(["📖 The Story", "📊 Descriptive", "🔍 Diagnostic", "🎯 Clustering", "🤖 Classification", "🔗 Association Rules", "💊 Prescriptive"])

# ============================================================
# TAB 0: THE STORY
# ============================================================
with tabs[0]:
    st.markdown("### 📖 The Business Scenario")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("""
<div class="story-box">
<b>2019</b> — Nikhil founds a software development company, hiring <b>100 employees</b> with a mix of developers, testers, 
project managers, and admin staff. The company thrives with a developer-heavy workforce.
</div>
<div class="story-box">
<b>2020 — COVID strikes</b> — The company is in rough waters. Then a lifeline arrives: <b>Ritika</b>, a client, approaches Nikhil 
to develop <b>two software products</b> within <b>3 months</b>. The catch? Every assigned employee must know 
<b>Python + English + French</b>.
</div>
<div class="story-box">
<b>The Gap</b> — Nikhil needs <b>30 dedicated employees</b> (with 15-day buffer). He asks <b>Anshul (Head HR)</b> to allocate 30 
employees with all 3 skills. Anshul discovers: <span style="color:#f87171;font-weight:700">only 3 employees have all 3 skills.</span>
</div>
<div class="story-box">
<b>The Mission</b> — Nikhil needs <b>37 employees</b> (27 to fill the gap + 10 buffer) from the remaining 97 who can be 
<b>trained</b>. They must have ≥1 required skill, good manager feedback, strong performance, learning attitude, and 
skill growth history. Additionally, Nikhil must <b>predict attrition risk</b> — he can't lose trainees mid-project.
</div>
""", unsafe_allow_html=True)
    
    with c2:
        st.markdown("#### Required Skills")
        fig = go.Figure()
        skills_count = [int(df['Skill_Python'].sum()), int(df['Lang_English'].sum()), int(df['Lang_French'].sum())]
        fig.add_trace(go.Bar(
            x=['Python', 'English', 'French'], y=skills_count,
            marker=dict(color=['#3776ab', '#e74c3c', '#1e3a5f']),
            text=skills_count, textposition='outside',
        ))
        fig.update_layout(yaxis_title='Employees', title='Required Skill Availability')
        apply_theme(fig, 300)
        st.plotly_chart(fig, use_container_width=True)

    # Skill overlap analysis
    st.markdown("### 🔀 Required Skills Overlap Analysis")
    c1, c2, c3 = st.columns(3)
    
    # Count employees by skill combination
    p = set(df[df['Skill_Python']==1].index)
    e = set(df[df['Lang_English']==1].index)
    f = set(df[df['Lang_French']==1].index)
    
    only_p = len(p - e - f)
    only_e = len(e - p - f)
    only_f = len(f - p - e)
    p_and_e = len((p & e) - f)
    p_and_f = len((p & f) - e)
    e_and_f = len((e & f) - p)
    all_three = len(p & e & f)
    none = len(set(range(100)) - p - e - f)
    
    with c1:
        combo_data = pd.DataFrame({
            'Combination': ['Python Only', 'English Only', 'French Only', 'Python+English', 
                          'Python+French', 'English+French', 'All Three', 'None'],
            'Count': [only_p, only_e, only_f, p_and_e, p_and_f, e_and_f, all_three, none]
        })
        fig = px.bar(combo_data, x='Combination', y='Count', 
                     color='Count', color_continuous_scale='Viridis',
                     title='Skill Combination Distribution')
        fig.update_layout(xaxis_tickangle=-45)
        apply_theme(fig, 380)
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        fig = go.Figure(data=[go.Pie(
            labels=[f'{i} Skills' for i in range(4)],
            values=[len(df[df['RequiredSkillsCount']==i]) for i in range(4)],
            hole=0.6,
            marker=dict(colors=['#64748b', '#fbbf24', '#a78bfa', '#6ee7b7']),
            textinfo='label+value'
        )])
        fig.update_layout(title='Employees by # of Required Skills',
                         annotations=[dict(text=f"<b>3</b><br>have all", x=0.5, y=0.5, font_size=16, font_color='#6ee7b7', showarrow=False)])
        apply_theme(fig, 380)
        st.plotly_chart(fig, use_container_width=True)
    
    with c3:
        # The 3 triple-skilled employees
        st.markdown("#### ✅ The 3 Triple-Skilled Employees")
        for _, row in triple_skilled.iterrows():
            st.markdown(f"""
<div style="background:rgba(110,231,183,0.1);border:1px solid rgba(110,231,183,0.3);border-radius:8px;padding:0.8rem;margin:0.4rem 0;">
<b style="color:#6ee7b7">{row['Name']}</b> ({row['EmployeeID']})<br>
<span style="color:#94a3b8">{row['Role']} · {row['Department']}</span><br>
<span style="color:#94a3b8">Perf: {row['PerformanceRating']} · Mgr: {row['ManagerFeedback']}</span>
</div>""", unsafe_allow_html=True)
        st.markdown(f"""
<div class="danger-box">
<b>Crisis:</b> Need 30, have 3. Must find and train <b>37 more</b> from the remaining 97 employees.
</div>""", unsafe_allow_html=True)

# ============================================================
# TAB 1: DESCRIPTIVE
# ============================================================
with tabs[1]:
    st.markdown("### 📊 Workforce Profiling — What Does Our Team Look Like?")
    
    # Row 1: Department & Role
    c1, c2 = st.columns(2)
    with c1:
        dept_counts = df['Department'].value_counts().reset_index()
        fig = px.bar(dept_counts, x='count', y='Department', orientation='h',
                     color='count', color_continuous_scale='Tealgrn',
                     title='Department Distribution', text='count')
        fig.update_traces(textposition='outside')
        apply_theme(fig, 380)
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        role_counts = df['Role'].value_counts().reset_index()
        fig = px.bar(role_counts, x='count', y='Role', orientation='h',
                     color='count', color_continuous_scale='Purp',
                     title='Role Distribution', text='count')
        fig.update_traces(textposition='outside')
        apply_theme(fig, 380)
        st.plotly_chart(fig, use_container_width=True)

    # Row 2: Age & Salary distributions
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x='Age', color='Gender', barmode='overlay', nbins=20,
                          color_discrete_map={'Male': '#60a5fa', 'Female': '#f472b6'},
                          title='Age Distribution by Gender')
        fig.update_traces(opacity=0.7)
        apply_theme(fig, 380)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.box(df, x='Department', y='MonthlySalary_INR', color='Department',
                     title='Salary Distribution by Department')
        apply_theme(fig, 380)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Row 3: Skill Heatmap
    st.markdown("### 🧩 Complete Skill Matrix")
    skill_cols = ['Skill_Python','Skill_Java','Skill_SQL','Skill_JavaScript','Skill_CSharp','Lang_English','Lang_French','Lang_Hindi']
    skill_by_role = df.groupby('Role')[skill_cols].mean().round(2)
    skill_by_role.columns = [c.replace('Skill_','').replace('Lang_','') for c in skill_by_role.columns]
    fig = px.imshow(skill_by_role, text_auto='.0%', color_continuous_scale='YlGnBu',
                    title='Skill Prevalence by Role (% of employees in each role)')
    apply_theme(fig, 450)
    st.plotly_chart(fig, use_container_width=True)
    
    # Row 4: Required skills deep dive
    st.markdown("### 🎯 Required Skills Gap — Deep Dive")
    c1, c2, c3 = st.columns(3)
    for col_widget, skill_name, skill_col, color in [
        (c1, 'Python', 'Skill_Python', '#3776ab'),
        (c2, 'English', 'Lang_English', '#e74c3c'),
        (c3, 'French', 'Lang_French', '#1e3a5f')
    ]:
        with col_widget:
            by_dept = df.groupby('Department')[skill_col].agg(['sum','count']).reset_index()
            by_dept['rate'] = (by_dept['sum'] / by_dept['count'] * 100).round(1)
            fig = px.bar(by_dept, x='Department', y='rate', title=f'{skill_name} by Department (%)',
                        text='rate', color_discrete_sequence=[color])
            fig.update_traces(texttemplate='%{text}%', textposition='outside')
            fig.update_layout(yaxis_title='% with Skill')
            apply_theme(fig, 350)
            st.plotly_chart(fig, use_container_width=True)

    # Row 5: Performance metrics
    st.markdown("### 📈 Performance & Learning Landscape")
    c1, c2, c3 = st.columns(3)
    with c1:
        fig = px.histogram(df, x='PerformanceRating', nbins=20, color_discrete_sequence=['#a78bfa'],
                          title='Performance Rating Distribution')
        apply_theme(fig, 350)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(df, x='LearningAttitude', nbins=20, color_discrete_sequence=['#6ee7b7'],
                          title='Learning Attitude Distribution')
        apply_theme(fig, 350)
        st.plotly_chart(fig, use_container_width=True)
    with c3:
        fig = px.histogram(df, x='SkillsUpgradedLastYear', color_discrete_sequence=['#fbbf24'],
                          title='Skills Upgraded Last Year')
        apply_theme(fig, 350)
        st.plotly_chart(fig, use_container_width=True)

    # Row 6: Scatter - Learning vs Performance
    fig = px.scatter(df, x='LearningAttitude', y='PerformanceRating', 
                     color='Department', size='TrainingHoursLastYear',
                     hover_data=['Name', 'Role', 'RequiredSkillsCount'],
                     title='Learning Attitude vs Performance (size = Training Hours)')
    apply_theme(fig, 450)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 2: DIAGNOSTIC
# ============================================================
with tabs[2]:
    st.markdown("### 🔍 Root Cause Analysis — Why the Skill Gap?")
    
    # Why is French so rare?
    st.markdown("""
<div class="insight-box">
<b>Key Question:</b> Why do only 3 employees have all 3 required skills? The bottleneck is clear: 
<b>French is extremely rare</b> — only {:.0f}% of employees know French, compared to {:.0f}% for Python and {:.0f}% for English.
</div>""".format(
        df['Lang_French'].mean()*100, df['Skill_Python'].mean()*100, df['Lang_English'].mean()*100
    ), unsafe_allow_html=True)
    
    # Correlation with Attrition
    st.markdown("### 📉 What Drives Attrition Risk?")
    numeric_cols = ['Age','MonthlySalary_INR','ManagerFeedback','PerformanceRating','LearningAttitude',
                    'SkillsUpgradedLastYear','TrainingHoursLastYear','ProjectsCompleted','Certifications',
                    'LastPromotionMonthsAgo','WorkLifeBalance','JobSatisfaction','EnvironmentSatisfaction',
                    'AbsenteeismDays','TeamCollaborationScore','TotalTechSkills','RequiredSkillsCount']
    
    corr_with_attrition = []
    for col in numeric_cols:
        r, p = stats.pointbiserialr(df['Attrition_Flag'], df[col])
        corr_with_attrition.append({'Feature': col, 'Correlation': round(r, 3), 'p_value': round(p, 4)})
    corr_df = pd.DataFrame(corr_with_attrition).sort_values('Correlation')
    
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = go.Figure()
        colors = ['#f87171' if v > 0 else '#34d399' for v in corr_df['Correlation']]
        fig.add_trace(go.Bar(
            x=corr_df['Correlation'], y=corr_df['Feature'], orientation='h',
            marker=dict(color=colors), text=corr_df['Correlation'], textposition='outside'
        ))
        fig.update_layout(title='Point-Biserial Correlation with Attrition Risk', xaxis_title='Correlation')
        apply_theme(fig, 550)
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        st.markdown("#### Interpretation Guide")
        st.markdown("""
<div class="story-box">
<b style="color:#f87171">Red bars (positive)</b> → Higher values increase attrition risk (e.g., AbsenteeismDays, OverTime)<br><br>
<b style="color:#34d399">Green bars (negative)</b> → Higher values reduce attrition risk (e.g., JobSatisfaction, MonthlySalary)
</div>""", unsafe_allow_html=True)
        sig = corr_df[corr_df['p_value'] < 0.05]
        st.markdown(f"**Statistically significant features (p < 0.05):** {len(sig)} of {len(corr_df)}")
        st.dataframe(sig[['Feature','Correlation','p_value']].style.format({'Correlation':'{:.3f}','p_value':'{:.4f}'}), hide_index=True, height=300)

    # Chi-Square for categorical
    st.markdown("### 📋 Categorical Feature Analysis (Chi-Square)")
    cat_cols = ['Department', 'Role', 'Gender', 'OverTime']
    chi_results = []
    for col in cat_cols:
        ct = pd.crosstab(df[col], df['Attrition_Risk_3Months'])
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        n_obs = ct.sum().sum()
        k = min(ct.shape)
        cramers_v = np.sqrt(chi2 / (n_obs * (k - 1))) if (k - 1) > 0 else 0
        chi_results.append({'Feature': col, 'Chi²': round(chi2, 2), 'p-value': round(p, 4),
                           "Cramér's V": round(cramers_v, 3), 'Significant': '✅' if p < 0.05 else '❌'})
    
    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(pd.DataFrame(chi_results), hide_index=True)
    with c2:
        chi_df = pd.DataFrame(chi_results).sort_values("Cramér's V")
        fig = go.Figure(go.Bar(
            x=chi_df["Cramér's V"], y=chi_df['Feature'], orientation='h',
            marker=dict(color=chi_df["Cramér's V"], colorscale='Oryel'),
            text=chi_df["Cramér's V"], textposition='outside'
        ))
        fig.update_layout(title="Cramér's V — Effect Size", xaxis_title="Cramér's V")
        apply_theme(fig, 300)
        st.plotly_chart(fig, use_container_width=True)

    # Attrition by key factors
    st.markdown("### 🔬 Attrition Rate Drill-Downs")
    c1, c2, c3 = st.columns(3)
    with c1:
        ot_attr = df.groupby('OverTime')['Attrition_Flag'].mean().reset_index()
        ot_attr['Rate'] = (ot_attr['Attrition_Flag'] * 100).round(1)
        fig = px.bar(ot_attr, x='OverTime', y='Rate', text='Rate',
                     color='OverTime', color_discrete_map={'Yes':'#f87171','No':'#34d399'},
                     title='Attrition Rate by OverTime')
        fig.update_traces(texttemplate='%{text}%')
        apply_theme(fig, 350)
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        dept_attr = df.groupby('Department')['Attrition_Flag'].mean().reset_index()
        dept_attr['Rate'] = (dept_attr['Attrition_Flag'] * 100).round(1)
        dept_attr = dept_attr.sort_values('Rate')
        fig = px.bar(dept_attr, x='Rate', y='Department', orientation='h', text='Rate',
                     color='Rate', color_continuous_scale='RdYlGn_r',
                     title='Attrition Rate by Department')
        fig.update_traces(texttemplate='%{text}%')
        apply_theme(fig, 350)
        st.plotly_chart(fig, use_container_width=True)
    
    with c3:
        sat_cols = ['JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance']
        stayed = df[df['Attrition_Flag']==0][sat_cols].mean()
        left = df[df['Attrition_Flag']==1][sat_cols].mean()
        sat_compare = pd.DataFrame({'Factor': sat_cols, 'Stayed': stayed.values, 'Left': left.values})
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Stayed', x=sat_compare['Factor'], y=sat_compare['Stayed'], marker_color='#34d399'))
        fig.add_trace(go.Bar(name='Left', x=sat_compare['Factor'], y=sat_compare['Left'], marker_color='#f87171'))
        fig.update_layout(title='Satisfaction: Stayed vs Left', barmode='group', yaxis_title='Avg Score')
        apply_theme(fig, 350)
        st.plotly_chart(fig, use_container_width=True)

    # Feature correlation heatmap
    st.markdown("### 🗺️ Feature Correlation Heatmap")
    key_features = ['Attrition_Flag','ManagerFeedback','PerformanceRating','LearningAttitude',
                    'MonthlySalary_INR','AbsenteeismDays','JobSatisfaction','WorkLifeBalance',
                    'SkillsUpgradedLastYear','TrainingHoursLastYear','RequiredSkillsCount']
    corr_matrix = df[key_features].corr().round(2)
    fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                    title='Correlation Matrix — Key Variables')
    apply_theme(fig, 550)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 3: CLUSTERING
# ============================================================
with tabs[3]:
    st.markdown("### 🎯 Employee Segmentation — Finding the Trainable 37")
    
    st.markdown("""
<div class="story-box">
<b>Objective:</b> Cluster the 97 remaining employees (excluding the 3 triple-skilled) into meaningful segments 
to identify the best <b>37 candidates</b> for training. Clustering uses: LearningAttitude, ManagerFeedback, 
PerformanceRating, SkillsUpgradedLastYear, TrainingHoursLastYear, RequiredSkillsCount, TeamCollaborationScore.
</div>""", unsafe_allow_html=True)
    
    # Prepare clustering data
    cluster_features = ['LearningAttitude','ManagerFeedback','PerformanceRating',
                       'SkillsUpgradedLastYear','TrainingHoursLastYear','RequiredSkillsCount',
                       'TeamCollaborationScore','Certifications']
    
    X_cluster = remaining_97[cluster_features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # Elbow method
    c1, c2 = st.columns(2)
    with c1:
        inertias = []
        K_range = range(2, 9)
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X_scaled)
            inertias.append(km.inertia_)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(K_range), y=inertias, mode='lines+markers',
                                marker=dict(size=10, color='#a78bfa'), line=dict(color='#a78bfa')))
        fig.add_vline(x=4, line_dash='dash', line_color='#fbbf24', annotation_text='Optimal K=4')
        fig.update_layout(title='Elbow Method — Optimal Clusters', xaxis_title='K', yaxis_title='Inertia')
        apply_theme(fig, 380)
        st.plotly_chart(fig, use_container_width=True)
    
    # Apply K=4
    km_final = KMeans(n_clusters=4, random_state=42, n_init=10)
    remaining_97 = remaining_97.copy()
    remaining_97['Cluster'] = km_final.fit_predict(X_scaled)
    
    # Cluster profiles
    cluster_profiles = remaining_97.groupby('Cluster')[cluster_features].mean().round(2)
    
    # Name clusters based on profiles
    cluster_means = cluster_profiles.mean(axis=1)
    sorted_clusters = cluster_means.sort_values(ascending=False).index.tolist()
    cluster_names = {}
    labels = ['⭐ Star Candidates', '📈 High Potential', '📊 Average Pool', '📉 Low Readiness']
    for i, c in enumerate(sorted_clusters):
        cluster_names[c] = labels[i]
    remaining_97['ClusterName'] = remaining_97['Cluster'].map(cluster_names)
    
    with c2:
        # Cluster size
        cl_counts = remaining_97['ClusterName'].value_counts().reset_index()
        cl_counts.columns = ['Cluster', 'Count']
        fig = px.pie(cl_counts, values='Count', names='Cluster', hole=0.55,
                     color_discrete_sequence=['#6ee7b7','#a78bfa','#fbbf24','#f87171'],
                     title='Cluster Distribution (97 employees)')
        fig.update_traces(textinfo='label+value')
        apply_theme(fig, 380)
        st.plotly_chart(fig, use_container_width=True)

    # Cluster radar
    st.markdown("### 📡 Cluster Profiles — Radar Comparison")
    fig = go.Figure()
    radar_features = ['LearningAttitude','ManagerFeedback','PerformanceRating','SkillsUpgradedLastYear','TeamCollaborationScore']
    colors_radar = ['#6ee7b7','#a78bfa','#fbbf24','#f87171']
    for idx, (cluster_id, row) in enumerate(cluster_profiles.iterrows()):
        name = cluster_names[cluster_id]
        vals = [row[f] for f in radar_features]
        vals.append(vals[0])
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=radar_features + [radar_features[0]],
            fill='toself', name=name, line=dict(color=colors_radar[idx]),
            fillcolor=f'rgba({",".join(str(int(colors_radar[idx].lstrip("#")[i:i+2],16)) for i in (0,2,4))},0.15)'
        ))
    fig.update_layout(title='Cluster Comparison Radar', polar=dict(
        radialaxis=dict(visible=True, range=[0, 10], gridcolor='rgba(99,102,241,0.1)'),
        bgcolor='rgba(0,0,0,0)'
    ))
    apply_theme(fig, 480)
    st.plotly_chart(fig, use_container_width=True)

    # 2D scatter visualization
    st.markdown("### 🗺️ Cluster Visualization (2D Projection)")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    remaining_97['PCA1'] = X_pca[:, 0]
    remaining_97['PCA2'] = X_pca[:, 1]
    
    fig = px.scatter(remaining_97, x='PCA1', y='PCA2', color='ClusterName',
                     hover_data=['Name','Role','RequiredSkillsCount','LearningAttitude','PerformanceRating'],
                     symbol='ClusterName', size='LearningAttitude',
                     title=f'Employee Clusters (PCA — {pca.explained_variance_ratio_.sum()*100:.1f}% variance explained)',
                     color_discrete_sequence=['#6ee7b7','#a78bfa','#fbbf24','#f87171'])
    apply_theme(fig, 500)
    st.plotly_chart(fig, use_container_width=True)

    # SELECT THE 37
    st.markdown("### ✅ Selecting the Best 37 Candidates")
    st.markdown("""
<div class="insight-box">
<b>Selection Logic:</b> From the 97 remaining employees, select the top 37 by:<br>
1. Must have <b>≥ 1 required skill</b> (Python / English / French)<br>
2. Ranked by composite <b>Trainability Score</b> = weighted average of Learning Attitude (25%), Manager Feedback (20%), 
Performance Rating (20%), Skills Upgraded (15%), Training Hours (10%), Team Collaboration (10%)
</div>""", unsafe_allow_html=True)
    
    eligible = remaining_97[remaining_97['RequiredSkillsCount'] >= 1].copy()
    eligible = eligible.sort_values('TrainabilityScore', ascending=False)
    top_37 = eligible.head(37)
    
    c1, c2 = st.columns([3, 1])
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=top_37['TrainabilityScore'], y=top_37['Name'], orientation='h',
            marker=dict(
                color=top_37['TrainabilityScore'],
                colorscale='Viridis',
                colorbar=dict(title='Score')
            ),
            text=[f"{s:.1f} | {r} skill(s)" for s, r in zip(top_37['TrainabilityScore'], top_37['RequiredSkillsCount'])],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Score: %{x:.2f}<br>Cluster: %{customdata[0]}<br>Role: %{customdata[1]}<extra></extra>',
            customdata=list(zip(top_37['ClusterName'], top_37['Role']))
        ))
        fig.update_layout(title='Top 37 Candidates — Ranked by Trainability Score',
                         xaxis_title='Trainability Score', yaxis=dict(autorange='reversed'))
        apply_theme(fig, max(500, len(top_37) * 18))
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        st.markdown("#### Selection Summary")
        cluster_breakdown = top_37['ClusterName'].value_counts()
        for name, count in cluster_breakdown.items():
            st.markdown(f"**{name}:** {count}")
        st.markdown("---")
        st.markdown(f"**Avg Trainability:** {top_37['TrainabilityScore'].mean():.2f}")
        st.markdown(f"**With Python:** {top_37['Skill_Python'].sum():.0f}")
        st.markdown(f"**With English:** {top_37['Lang_English'].sum():.0f}")
        st.markdown(f"**With French:** {top_37['Lang_French'].sum():.0f}")
        at_risk_37 = top_37['Attrition_Risk_3Months'].value_counts().get('Yes', 0)
        st.markdown(f"""
<div class="danger-box">
<b>⚠️ {at_risk_37} of 37</b> selected candidates are flagged as attrition risk! Classification analysis needed.
</div>""", unsafe_allow_html=True)

# ============================================================
# TAB 4: CLASSIFICATION
# ============================================================
with tabs[4]:
    st.markdown("### 🤖 Attrition Prediction — Will They Stay for 3 Months?")
    
    st.markdown("""
<div class="story-box">
<b>Objective:</b> Train classification models to predict which employees are likely to leave within 3 months. 
This is critical for Nikhil — if a trainee leaves mid-project, the entire timeline collapses.
</div>""", unsafe_allow_html=True)
    
    @st.cache_data
    def train_models():
        feature_cols = ['Age','MonthlySalary_INR','ManagerFeedback','PerformanceRating','LearningAttitude',
                       'SkillsUpgradedLastYear','TrainingHoursLastYear','ProjectsCompleted','Certifications',
                       'LastPromotionMonthsAgo','WorkLifeBalance','JobSatisfaction','EnvironmentSatisfaction',
                       'AbsenteeismDays','TeamCollaborationScore','TotalTechSkills','RequiredSkillsCount']
        # Encode OverTime
        df_model = df.copy()
        df_model['OverTime_enc'] = (df_model['OverTime'] == 'Yes').astype(int)
        feature_cols.append('OverTime_enc')
        
        X = df_model[feature_cols]
        y = df_model['Attrition_Flag']
        
        scaler_m = StandardScaler()
        X_scaled = scaler_m.fit_transform(X)
        
        models = {
            'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, random_state=42)
        }
        
        results = {}
        for name, model in models.items():
            cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='roc_auc')
            y_proba = cross_val_predict(model, X_scaled, y, cv=5, method='predict_proba')[:, 1]
            model.fit(X_scaled, y)
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            else:
                importances = np.abs(model.coef_[0])
            
            results[name] = {
                'model': model,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_proba': y_proba,
                'importances': importances,
                'feature_names': feature_cols,
            }
        return results, X_scaled, y, scaler_m, feature_cols
    
    results, X_model, y_model, scaler_model, feature_cols_model = train_models()
    
    # Model comparison
    c1, c2 = st.columns(2)
    with c1:
        model_names = list(results.keys())
        aucs = [results[m]['cv_mean'] for m in model_names]
        stds = [results[m]['cv_std'] for m in model_names]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=model_names, y=aucs, error_y=dict(type='data', array=stds),
            marker=dict(color=['#6ee7b7','#a78bfa','#fbbf24']),
            text=[f'{a:.3f}' for a in aucs], textposition='outside'
        ))
        fig.update_layout(title='Model Comparison — 5-Fold CV AUC', yaxis_title='AUC-ROC',
                         yaxis=dict(range=[0, 1.1]))
        apply_theme(fig, 400)
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        fig = go.Figure()
        colors_roc = ['#6ee7b7','#a78bfa','#fbbf24']
        for i, (name, res) in enumerate(results.items()):
            fpr, tpr, _ = roc_curve(y_model, res['y_proba'])
            auc_val = auc(fpr, tpr)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{name} (AUC={auc_val:.3f})',
                                    line=dict(color=colors_roc[i], width=2.5)))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random',
                                line=dict(color='#64748b', dash='dash')))
        fig.update_layout(title='ROC Curves', xaxis_title='FPR', yaxis_title='TPR')
        apply_theme(fig, 400)
        st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    st.markdown("### 🏆 Feature Importance Rankings")
    selected_model = st.selectbox("Select Model:", list(results.keys()), index=1)
    
    c1, c2 = st.columns([3, 2])
    with c1:
        res = results[selected_model]
        imp_df = pd.DataFrame({'Feature': res['feature_names'], 'Importance': res['importances']})
        imp_df = imp_df.sort_values('Importance', ascending=True).tail(15)
        fig = go.Figure(go.Bar(
            x=imp_df['Importance'], y=imp_df['Feature'], orientation='h',
            marker=dict(color=imp_df['Importance'], colorscale='Viridis'),
            text=imp_df['Importance'].round(3), textposition='outside'
        ))
        fig.update_layout(title=f'Top 15 Features — {selected_model}', xaxis_title='Importance')
        apply_theme(fig, 500)
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        # Consensus
        all_imp = pd.DataFrame({'Feature': results['Logistic Regression']['feature_names']})
        for name, res in results.items():
            normalized = res['importances'] / res['importances'].max()
            all_imp[name] = normalized
        all_imp['Consensus'] = all_imp[model_names].mean(axis=1)
        all_imp = all_imp.sort_values('Consensus', ascending=False).head(10)
        
        fig = go.Figure()
        for i, m in enumerate(model_names):
            fig.add_trace(go.Bar(name=m, y=all_imp['Feature'], x=all_imp[m], orientation='h',
                                marker_color=colors_roc[i]))
        fig.update_layout(title='Consensus Ranking (Top 10)', barmode='group', xaxis_title='Normalized Importance',
                         yaxis=dict(autorange='reversed'))
        apply_theme(fig, 500)
        st.plotly_chart(fig, use_container_width=True)

    # Confusion matrix for best model
    st.markdown("### 📊 Detailed Model Performance")
    best_model_name = max(results, key=lambda x: results[x]['cv_mean'])
    best_model = results[best_model_name]['model']
    y_pred = best_model.predict(X_model)
    cm = confusion_matrix(y_model, y_pred)
    
    c1, c2 = st.columns(2)
    with c1:
        fig = px.imshow(cm, text_auto=True, labels=dict(x='Predicted', y='Actual'),
                       x=['Stay', 'Leave'], y=['Stay', 'Leave'],
                       color_continuous_scale='Purp', title=f'Confusion Matrix — {best_model_name}')
        apply_theme(fig, 380)
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        report = classification_report(y_model, y_pred, target_names=['Stay','Leave'], output_dict=True)
        report_df = pd.DataFrame(report).T.round(3)
        st.markdown(f"#### Classification Report — {best_model_name}")
        st.dataframe(report_df, height=220)
        st.markdown(f"""
<div class="insight-box">
<b>Best Model:</b> {best_model_name} with AUC = {results[best_model_name]['cv_mean']:.3f}<br>
Recall for 'Leave' class is critical — we need to catch potential leavers, even at the cost of some false positives.
</div>""", unsafe_allow_html=True)

    # Risk scores for the 37 selected
    st.markdown("### 🚨 Attrition Risk — The Selected 37")
    df_37 = top_37.copy()
    df_37['OverTime_enc'] = (df_37['OverTime'] == 'Yes').astype(int)
    cols_for_pred = [c for c in feature_cols_model if c in df_37.columns]
    X_37 = scaler_model.transform(df_37[cols_for_pred])
    df_37['Attrition_Probability'] = best_model.predict_proba(X_37)[:, 1]
    df_37['Risk_Level'] = pd.cut(df_37['Attrition_Probability'], bins=[0, 0.3, 0.6, 1.0],
                                  labels=['🟢 Low', '🟡 Medium', '🔴 High'])
    
    fig = go.Figure()
    df_37_sorted = df_37.sort_values('Attrition_Probability', ascending=True)
    colors_bar = ['#34d399' if p < 0.3 else '#fbbf24' if p < 0.6 else '#f87171' for p in df_37_sorted['Attrition_Probability']]
    fig.add_trace(go.Bar(
        x=df_37_sorted['Attrition_Probability'], y=df_37_sorted['Name'], orientation='h',
        marker=dict(color=colors_bar),
        text=[f'{p:.0%}' for p in df_37_sorted['Attrition_Probability']], textposition='outside',
        hovertemplate='<b>%{y}</b><br>Risk: %{x:.1%}<br>Role: %{customdata[0]}<br>Salary: ₹%{customdata[1]:,}<extra></extra>',
        customdata=list(zip(df_37_sorted['Role'], df_37_sorted['MonthlySalary_INR']))
    ))
    fig.update_layout(title='Attrition Probability — Selected 37 Candidates',
                     xaxis_title='Attrition Probability', yaxis=dict(autorange='reversed'),
                     xaxis=dict(tickformat='.0%', range=[0, 1.15]))
    apply_theme(fig, max(500, len(df_37) * 18))
    st.plotly_chart(fig, use_container_width=True)
    
    risk_summary = df_37['Risk_Level'].value_counts()
    c1, c2, c3 = st.columns(3)
    for col, level, color in [(c1, '🟢 Low', '#34d399'), (c2, '🟡 Medium', '#fbbf24'), (c3, '🔴 High', '#f87171')]:
        count = risk_summary.get(level, 0)
        with col:
            st.markdown(f'<div class="kpi-card"><div class="value" style="color:{color}">{count}</div><div class="label">{level} Risk</div></div>', unsafe_allow_html=True)

# ============================================================
# TAB 5: ASSOCIATION RULES
# ============================================================
with tabs[5]:
    st.markdown("### 🔗 Association Rule Mining — Hidden Patterns")
    
    st.markdown("""
<div class="story-box">
<b>Objective:</b> Apply the Apriori algorithm to discover frequent itemsets and association rules in employee 
attributes. This reveals co-occurrence patterns like "If an employee has Python AND high learning attitude, 
they are likely to have low attrition risk."
</div>""", unsafe_allow_html=True)
    
    # Prepare binary transaction data
    @st.cache_data
    def mine_rules():
        from mlxtend.frequent_patterns import apriori, association_rules as ar_func
        
        txn = pd.DataFrame()
        txn['Has_Python'] = df['Skill_Python']
        txn['Has_Java'] = df['Skill_Java']
        txn['Has_SQL'] = df['Skill_SQL']
        txn['Has_JavaScript'] = df['Skill_JavaScript']
        txn['Has_English'] = df['Lang_English']
        txn['Has_French'] = df['Lang_French']
        txn['High_Learning'] = (df['LearningAttitude'] >= 7).astype(int)
        txn['Low_Learning'] = (df['LearningAttitude'] < 4).astype(int)
        txn['High_Performance'] = (df['PerformanceRating'] >= 4).astype(int)
        txn['Low_Performance'] = (df['PerformanceRating'] < 2.5).astype(int)
        txn['Good_Manager_FB'] = (df['ManagerFeedback'] >= 7).astype(int)
        txn['Poor_Manager_FB'] = (df['ManagerFeedback'] < 4).astype(int)
        txn['OverTime_Yes'] = (df['OverTime'] == 'Yes').astype(int)
        txn['High_Salary'] = (df['MonthlySalary_INR'] >= df['MonthlySalary_INR'].quantile(0.75)).astype(int)
        txn['Low_Salary'] = (df['MonthlySalary_INR'] <= df['MonthlySalary_INR'].quantile(0.25)).astype(int)
        txn['Skilled_Up'] = (df['SkillsUpgradedLastYear'] >= 2).astype(int)
        txn['High_Collab'] = (df['TeamCollaborationScore'] >= 7).astype(int)
        txn['Attrition_Risk'] = df['Attrition_Flag']
        txn['No_Attrition_Risk'] = 1 - df['Attrition_Flag']
        txn['Developer'] = (df['Department'] == 'Development').astype(int)
        txn['Has_Certs'] = (df['Certifications'] >= 2).astype(int)
        txn['Low_Satisfaction'] = (df['JobSatisfaction'] <= 2).astype(int)
        txn['High_Absence'] = (df['AbsenteeismDays'] >= 8).astype(int)
        
        txn_bool = txn.astype(bool)
        freq_items = apriori(txn_bool, min_support=0.08, use_colnames=True, max_len=4)
        rules = ar_func(freq_items, metric='confidence', min_threshold=0.5)
        rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(sorted(x)))
        rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(sorted(x)))
        rules['rule'] = rules['antecedents_str'] + ' → ' + rules['consequents_str']
        return rules, freq_items, txn
    
    try:
        rules, freq_items, txn_data = mine_rules()
        
        st.markdown(f"**Discovered {len(freq_items)} frequent itemsets and {len(rules)} association rules**")
        
        # Top rules by lift
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🏆 Top Rules by Lift")
            top_lift = rules.nlargest(15, 'lift')
            fig = go.Figure(go.Bar(
                x=top_lift['lift'].round(2), y=top_lift['rule'], orientation='h',
                marker=dict(color=top_lift['lift'], colorscale='Plasma'),
                text=top_lift['lift'].round(2), textposition='outside'
            ))
            fig.update_layout(title='Top 15 Association Rules (by Lift)', xaxis_title='Lift',
                             yaxis=dict(autorange='reversed'))
            apply_theme(fig, 550)
            st.plotly_chart(fig, use_container_width=True)
        
        with c2:
            st.markdown("#### 📊 Support vs Confidence")
            fig = px.scatter(rules, x='support', y='confidence', size='lift', color='lift',
                            hover_data=['rule'], color_continuous_scale='Plasma',
                            title='Rules: Support vs Confidence (size = Lift)')
            fig.update_layout(xaxis_title='Support', yaxis_title='Confidence')
            apply_theme(fig, 550)
            st.plotly_chart(fig, use_container_width=True)

        # Rules related to attrition
        st.markdown("### 🚨 Rules Involving Attrition Risk")
        attrition_rules = rules[
            rules['consequents_str'].str.contains('Attrition_Risk') | 
            rules['antecedents_str'].str.contains('Attrition_Risk')
        ].sort_values('lift', ascending=False).head(15)
        
        if len(attrition_rules) > 0:
            c1, c2 = st.columns([3, 2])
            with c1:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=attrition_rules['confidence'].round(2), y=attrition_rules['rule'], orientation='h',
                    marker=dict(color=['#f87171' if 'Attrition_Risk' in c else '#34d399' for c in attrition_rules['consequents_str']]),
                    text=[f"Conf: {c:.0%} | Lift: {l:.1f}" for c, l in zip(attrition_rules['confidence'], attrition_rules['lift'])],
                    textposition='outside'
                ))
                fig.update_layout(title='Attrition-Related Rules', xaxis_title='Confidence',
                                 yaxis=dict(autorange='reversed'))
                apply_theme(fig, 500)
                st.plotly_chart(fig, use_container_width=True)
            
            with c2:
                st.markdown("#### Key Insights from Rules")
                danger_rules = attrition_rules[attrition_rules['consequents_str'].str.contains('Attrition_Risk')]
                safe_rules = attrition_rules[attrition_rules['consequents_str'].str.contains('No_Attrition')]
                
                if len(danger_rules) > 0:
                    st.markdown(f"""
<div class="danger-box">
<b>🔴 Danger Patterns ({len(danger_rules)} rules):</b><br>
These combinations strongly predict attrition risk. Nikhil must avoid selecting employees matching these patterns, 
or implement retention measures first.
</div>""", unsafe_allow_html=True)
                
                if len(safe_rules) > 0:
                    st.markdown(f"""
<div class="story-box">
<b>🟢 Safe Patterns ({len(safe_rules)} rules):</b><br>
These combinations predict employee retention. Prioritise candidates matching these patterns for the project.
</div>""", unsafe_allow_html=True)

        # Rules related to required skills
        st.markdown("### 🎯 Rules Involving Required Skills (Python / English / French)")
        skill_rules = rules[
            rules['antecedents_str'].str.contains('Python|English|French') |
            rules['consequents_str'].str.contains('Python|English|French')
        ].sort_values('lift', ascending=False).head(12)
        
        if len(skill_rules) > 0:
            fig = go.Figure(go.Bar(
                x=skill_rules['lift'].round(2), y=skill_rules['rule'], orientation='h',
                marker=dict(color=skill_rules['confidence'], colorscale='Tealgrn',
                           colorbar=dict(title='Confidence')),
                text=[f"Sup: {s:.0%} | Conf: {c:.0%}" for s, c in zip(skill_rules['support'], skill_rules['confidence'])],
                textposition='outside'
            ))
            fig.update_layout(title='Skill-Related Association Rules (by Lift)', xaxis_title='Lift',
                             yaxis=dict(autorange='reversed'))
            apply_theme(fig, 450)
            st.plotly_chart(fig, use_container_width=True)
        
        # Full rules table
        with st.expander("📋 View All Rules (sortable)"):
            display_cols = ['antecedents_str','consequents_str','support','confidence','lift']
            rules_display = rules[display_cols].rename(columns={
                'antecedents_str':'Antecedents','consequents_str':'Consequents',
                'support':'Support','confidence':'Confidence','lift':'Lift'
            }).sort_values('Lift', ascending=False)
            st.dataframe(rules_display.style.format({'Support':'{:.3f}','Confidence':'{:.3f}','Lift':'{:.2f}'}),
                        hide_index=True, height=400)
    
    except ImportError:
        st.error("mlxtend library is required for association rule mining. Install with: `pip install mlxtend`")
    except Exception as e:
        st.error(f"Error in association rule mining: {e}")

# ============================================================
# TAB 6: PRESCRIPTIVE
# ============================================================
with tabs[6]:
    st.markdown("### 💊 Final Recommendations — Nikhil's Action Plan")
    
    # Prepare final 37 with all analysis
    final_37 = top_37.copy()
    final_37['OverTime_enc'] = (final_37['OverTime'] == 'Yes').astype(int)
    cols_pred = [c for c in feature_cols_model if c in final_37.columns]
    X_final = scaler_model.transform(final_37[cols_pred])
    best_model_name = max(results, key=lambda x: results[x]['cv_mean'])
    best_model = results[best_model_name]['model']
    final_37['Attrition_Prob'] = best_model.predict_proba(X_final)[:, 1]
    final_37['Risk'] = pd.cut(final_37['Attrition_Prob'], bins=[0, 0.3, 0.6, 1.0],
                               labels=['Low', 'Medium', 'High'])
    
    # Skills gap for each employee
    final_37['Needs_Python'] = 1 - final_37['Skill_Python']
    final_37['Needs_English'] = 1 - final_37['Lang_English']
    final_37['Needs_French'] = 1 - final_37['Lang_French']
    final_37['Skills_To_Train'] = final_37['Needs_Python'] + final_37['Needs_English'] + final_37['Needs_French']
    
    # Summary KPIs
    st.markdown("### 📟 Project Readiness Dashboard")
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(f'<div class="kpi-card"><div class="value" style="color:#6ee7b7">3 + 37</div><div class="label">Total Team (3 Ready + 37 Training)</div></div>', unsafe_allow_html=True)
    with k2:
        high_risk = len(final_37[final_37['Risk']=='High'])
        st.markdown(f'<div class="kpi-card"><div class="value" style="color:#f87171">{high_risk}</div><div class="label">High Attrition Risk</div></div>', unsafe_allow_html=True)
    with k3:
        avg_skills_gap = final_37['Skills_To_Train'].mean()
        st.markdown(f'<div class="kpi-card"><div class="value" style="color:#fbbf24">{avg_skills_gap:.1f}</div><div class="label">Avg Skills to Train</div></div>', unsafe_allow_html=True)
    with k4:
        need_french = int(final_37['Needs_French'].sum())
        st.markdown(f'<div class="kpi-card"><div class="value" style="color:#a78bfa">{need_french}</div><div class="label">Need French Training</div></div>', unsafe_allow_html=True)
    with k5:
        ready_in_1 = len(final_37[final_37['Skills_To_Train']==1])
        st.markdown(f'<div class="kpi-card"><div class="value" style="color:#60a5fa">{ready_in_1}</div><div class="label">Need Only 1 Skill</div></div>', unsafe_allow_html=True)

    # Training plan visualization
    st.markdown("### 📚 Training Needs Matrix")
    c1, c2 = st.columns([3, 2])
    with c1:
        training_matrix = final_37[['Name','Skill_Python','Lang_English','Lang_French','Needs_Python','Needs_English','Needs_French','Skills_To_Train','Risk']].copy()
        training_matrix = training_matrix.sort_values('Skills_To_Train')
        
        # Create heatmap of what they NEED
        needs_data = training_matrix[['Needs_Python','Needs_English','Needs_French']].values
        fig = px.imshow(needs_data.T, 
                       y=['Python', 'English', 'French'],
                       x=training_matrix['Name'].values,
                       color_continuous_scale=[[0, '#1e293b'], [1, '#f87171']],
                       title='Training Gap Matrix (Red = Needs Training)')
        fig.update_layout(xaxis_tickangle=-45)
        apply_theme(fig, 350)
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        # Training effort
        training_effort = final_37.groupby('Skills_To_Train').size().reset_index(name='Count')
        training_effort['Label'] = training_effort['Skills_To_Train'].map({1: '1 Skill Gap', 2: '2 Skill Gaps'})
        fig = px.pie(training_effort, values='Count', names='Label', hole=0.55,
                     color_discrete_sequence=['#fbbf24','#f87171'],
                     title='Training Effort Distribution')
        apply_theme(fig, 350)
        st.plotly_chart(fig, use_container_width=True)

    # Risk-adjusted recommendations
    st.markdown("### 🎯 Risk-Adjusted Final Roster")
    
    # Categorize into tiers
    safe_37 = final_37[final_37['Risk'] != 'High']
    risky_37 = final_37[final_37['Risk'] == 'High']
    
    fig = px.scatter(final_37, x='TrainabilityScore', y='Attrition_Prob',
                     color='Risk', size='Skills_To_Train',
                     hover_data=['Name','Role','Department','RequiredSkillsCount'],
                     color_discrete_map={'Low':'#34d399','Medium':'#fbbf24','High':'#f87171'},
                     title='Trainability vs Attrition Risk (size = Training Needed)')
    fig.add_hline(y=0.6, line_dash='dash', line_color='#f87171', annotation_text='High Risk Threshold')
    fig.add_hline(y=0.3, line_dash='dash', line_color='#fbbf24', annotation_text='Medium Risk Threshold')
    fig.update_layout(xaxis_title='Trainability Score', yaxis_title='Attrition Probability',
                     yaxis=dict(tickformat='.0%'))
    apply_theme(fig, 480)
    st.plotly_chart(fig, use_container_width=True)

    # Strategic Recommendations
    st.markdown("### 📋 Strategic Recommendations")
    
    st.markdown(f"""
<div class="rec-card-high">
<h4 style="color:#f87171;margin:0">🔴 CRITICAL: Immediate Retention for {high_risk} High-Risk Candidates</h4>
<p style="margin:0.5rem 0 0">These employees score high on trainability but are likely to leave within 3 months. 
<b>Actions:</b> Immediate 1-on-1 with manager, salary review, reduce overtime, assign dedicated mentor, 
offer project completion bonus tied to 3-month retention.</p>
</div>
""", unsafe_allow_html=True)
    
    st.markdown(f"""
<div class="rec-card-med">
<h4 style="color:#fbbf24;margin:0">🟡 PRIORITY: French Language Bootcamp for {need_french} Employees</h4>
<p style="margin:0.5rem 0 0">French is the biggest bottleneck — {need_french} of 37 selected candidates lack French proficiency.
<b>Actions:</b> Launch intensive 4-week French language bootcamp with daily sessions. 
Partner with Alliance Française for accelerated business French certification. Budget: ₹2-3L.</p>
</div>
""", unsafe_allow_html=True)
    
    python_need = int(final_37['Needs_Python'].sum())
    english_need = int(final_37['Needs_English'].sum())
    st.markdown(f"""
<div class="rec-card-med">
<h4 style="color:#fbbf24;margin:0">🟡 TRAINING: Python ({python_need}) and English ({english_need}) Upskilling</h4>
<p style="margin:0.5rem 0 0"><b>Python:</b> 2-week intensive Python for software development bootcamp. Pair learners with existing Python developers.
<b>English:</b> Business English communication workshops — these employees likely have passive knowledge, focus on active usage.</p>
</div>
""", unsafe_allow_html=True)
    
    buffer_count = max(0, len(safe_37) - 27)
    st.markdown(f"""
<div class="rec-card-low">
<h4 style="color:#6ee7b7;margin:0">🟢 BUFFER: {buffer_count} Low-Risk Employees as Safety Net</h4>
<p style="margin:0.5rem 0 0">After removing high-risk candidates, {len(safe_37)} employees remain — giving a buffer of 
{buffer_count} beyond the 27 needed. These buffers should still receive training but can serve as replacements if any 
primary assignee leaves. Rotate them on non-critical modules.</p>
</div>
""", unsafe_allow_html=True)
    
    st.markdown("""
<div class="rec-card-low">
<h4 style="color:#6ee7b7;margin:0">🟢 MONITORING: Weekly Pulse Checks During Project</h4>
<p style="margin:0.5rem 0 0">Deploy the classification model as a bi-weekly early warning system. 
Track changes in satisfaction, absenteeism, and overtime for all 37 employees. 
Flag anyone whose predicted attrition probability crosses 0.5 for immediate HR intervention.</p>
</div>
""", unsafe_allow_html=True)

    # Timeline
    st.markdown("### 📅 Recommended Project Timeline")
    timeline_data = pd.DataFrame({
        'Phase': ['Week 1-2: Assessment & Selection', 'Week 2-4: French Bootcamp Phase 1', 
                  'Week 2-3: Python Bootcamp', 'Week 3-4: English Workshop',
                  'Week 3-6: French Bootcamp Phase 2', 'Week 4-8: Software Dev Sprint 1',
                  'Week 8-12: Software Dev Sprint 2', 'Week 12-14: Testing & Delivery (Buffer)'],
        'Start': [1, 2, 2, 3, 3, 4, 8, 12],
        'End': [2, 4, 3, 4, 6, 8, 12, 14],
        'Category': ['HR', 'Training', 'Training', 'Training', 'Training', 'Development', 'Development', 'Delivery']
    })
    
    fig = go.Figure()
    cat_colors = {'HR': '#a78bfa', 'Training': '#fbbf24', 'Development': '#6ee7b7', 'Delivery': '#60a5fa'}
    for _, row in timeline_data.iterrows():
        fig.add_trace(go.Bar(
            x=[row['End'] - row['Start']], y=[row['Phase']], orientation='h',
            base=row['Start'], marker_color=cat_colors[row['Category']],
            name=row['Category'], showlegend=row['Phase']==timeline_data[timeline_data['Category']==row['Category']].iloc[0]['Phase'],
            text=f"Week {row['Start']}-{row['End']}", textposition='inside',
            hovertemplate=f"<b>{row['Phase']}</b><br>Week {row['Start']} to {row['End']}<extra></extra>"
        ))
    fig.update_layout(title='Project Gantt Chart (14 weeks = 3 months + 15 days buffer)',
                     xaxis_title='Week', barmode='stack', yaxis=dict(autorange='reversed'))
    apply_theme(fig, 400)
    st.plotly_chart(fig, use_container_width=True)

    # Final employee table
    st.markdown("### 📋 Complete Candidate Roster")
    display_37 = final_37[['EmployeeID','Name','Role','Department','RequiredSkillsCount',
                           'Skill_Python','Lang_English','Lang_French','TrainabilityScore',
                           'Attrition_Prob','Risk','Skills_To_Train','ClusterName']].copy()
    display_37['Attrition_Prob'] = display_37['Attrition_Prob'].apply(lambda x: f'{x:.1%}')
    display_37 = display_37.sort_values('Risk', ascending=True)
    display_37.columns = ['ID','Name','Role','Dept','Req Skills','Python','English','French',
                          'Trainability','Attrition %','Risk','Gap','Cluster']
    st.dataframe(display_37, hide_index=True, height=500, use_container_width=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#64748b;font-size:0.85rem;padding:1rem;">
<b>Nikhil's Workforce Intelligence Suite</b> · Built with Streamlit + Plotly + scikit-learn + mlxtend<br>
Classification · Clustering · Association Rule Mining · Descriptive to Prescriptive Analytics<br>
Dataset: 100 employees · 31 features · COVID-era workforce planning scenario
</div>
""", unsafe_allow_html=True)
