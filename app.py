import streamlit as st
import pandas as pd
import plotly.express as px
import io
import os
import numpy as np
from reportlab.lib.utils import ImageReader
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI
import plotly.graph_objs as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Aura Analytics", layout="wide", page_icon="🌌")

# ---------------- THEME & CSS ENGINE ----------------
def apply_theme(theme_choice):
    if theme_choice == "Dark":
        bg = "#0e1117"; txt = "#ffffff"; card = "rgba(255,255,255,0.05)"; side = "#161b22"
        chart_tpl = "plotly_dark"
    else:
        bg = "#ffffff"; txt = "#1f1f1f"; card = "rgba(0,0,0,0.05)"; side = "#f0f2f6"
        chart_tpl = "plotly_white"
    
    st.markdown(f"""
        <style>
        .stApp {{ background-color: {bg}; color: {txt}; }}
        [data-testid="stSidebar"] {{ background-color: {side} !important; }}
        [data-testid="stSidebar"] * {{ color: {txt} !important; }}
        .kpi-box {{
            background: {card};
            padding: 20px; border-radius: 12px;
            border-left: 5px solid #00d4ff;
            text-align: center; margin-bottom: 10px;
        }}
        .stButton>button {{
            width: 100%; border-radius: 8px; height: 3em;
            background: linear-gradient(45deg, #00d4ff, #6a11cb); color: white; border: none;
        }}
        </style>
    """, unsafe_allow_html=True)
    return chart_tpl

# -----------Ask Ai------------
def local_ai_chat(df, question):
    question = question.lower()

    try:
        if "mean" in question or "average" in question:
            return df.mean(numeric_only=True)

        elif "max" in question:
            return df.max(numeric_only=True)

        elif "min" in question:
            return df.min(numeric_only=True)

        elif "summary" in question or "describe" in question:
            return df.describe()

        elif "columns" in question:
            return df.columns.tolist()

        elif "correlation" in question:
            return df.corr(numeric_only=True)

        elif "plot" in question or "chart" in question:
            num_cols = df.select_dtypes(include='number').columns
            if len(num_cols) >= 2:
                fig = px.scatter(df, x=num_cols[0], y=num_cols[1])
                return fig
            else:
                return "Not enough numeric columns for plotting"

        else:
            return "🤖 Try: average, max, min, summary, correlation, plot"

    except Exception as e:
        return str(e)
    
# ---------------Ai insights ---------------------
def generate_ai_insights(df, num_cols, cat_cols):
    total_rows, total_cols = df.shape
    missing = df.isnull().sum().sum()
    missing_pct = (missing / (total_rows * total_cols)) * 100

    num_col = num_cols[0] if num_cols else None
    cat_col = cat_cols[0] if cat_cols else None

    if num_col:
        avg = df[num_col].mean()
        median = df[num_col].median()
        std = df[num_col].std()
        min_val = df[num_col].min()
        max_val = df[num_col].max()
    else:
        avg = median = std = min_val = max_val = 0

    if cat_col:
        top_cat = df[cat_col].mode()[0]
        unique_vals = df[cat_col].nunique()
    else:
        top_cat = "N/A"
        unique_vals = 0

    # Data Quality Score
    data_quality = 100 - missing_pct

    # Simple insights logic
    if std > avg:
        spread_comment = "high variability detected (data is widely spread)"
    else:
        spread_comment = "data is relatively stable with low variability"

    if missing_pct > 20:
        missing_comment = "significant missing data present (needs cleaning)"
    else:
        missing_comment = "data completeness is good"

    return f"""
🔍 DATASET OVERVIEW
The dataset contains {total_rows} rows and {total_cols} columns, with a total of {missing} missing values ({missing_pct:.2f}%).
Overall data quality score is approximately {data_quality:.1f}%.

📊 NUMERICAL ANALYSIS
The primary metric '{num_col}' shows:
• Average: {avg:.2f}
• Median: {median:.2f}
• Standard Deviation: {std:.2f}
• Range: {min_val:.2f} to {max_val:.2f}

This indicates {spread_comment}.

📁 CATEGORICAL INSIGHTS
The key category '{cat_col}' contains {unique_vals} unique values.
The most frequent category is '{top_cat}', suggesting dominant patterns in this segment.

⚠️ DATA QUALITY
{missing_comment}. Missing values should be handled for better model accuracy and analysis.

📈 BUSINESS INSIGHTS
• The dataset is suitable for trend analysis and predictive modeling.
• Dominant categories can be targeted for business optimization.
• Variability in data suggests potential outliers or diverse behavior patterns.

🚀 RECOMMENDATIONS
• Perform deeper correlation analysis between variables.
• Handle missing values using imputation techniques.
• Apply machine learning models for forecasting and segmentation.
"""
# ------------------------------
def auto_generate_charts(df):
    charts = []
    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    # 🔥 AI SMART DATE DETECTION
    date_cols = []
    for col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col])
            date_cols.append(col)
            break
        except:
            continue

    # 0️⃣ Time Series (if date found)
    if len(date_cols) >= 1 and len(num_cols) >= 1:
        fig = px.line(df, x=date_cols[0], y=num_cols[0],
                      title=f"Time Trend: {num_cols[0]} over {date_cols[0]}")
        fig.write_image("chart_time.png", width=1200, height=700)
        charts.append("chart_time.png")

    # 1️⃣ Histogram
    if len(num_cols) >= 1:
        fig = px.histogram(df, x=num_cols[0],
                   title=f"Distribution of {num_cols[0]}",
                   color_discrete_sequence=["#00d4ff"])
        fig.write_image("chart_hist.png", width=1200, height=700)
        charts.append("chart_hist.png")

    # 2️⃣ Bar (Top categories)
    if len(cat_cols) >= 1:
        top_data = df[cat_cols[0]].value_counts().head(10).reset_index()
        top_data.columns = ["Category", "Count"]
        fig = px.bar(top_data, x="Category", y="Count",
             title=f"Top Categories in {cat_cols[0]}",
             color="Count",
             color_continuous_scale="Blues")
        fig.write_image("chart_bar.png", width=1200, height=700)
        charts.append("chart_bar.png")

    # 3️⃣ Pie
    if len(cat_cols) >= 1:
        fig = px.pie(df, names=cat_cols[0],
             title=f"{cat_cols[0]} Distribution",
             color_discrete_sequence=px.colors.sequential.RdBu)
        fig.write_image("chart_pie.png", width=1200, height=700)
        charts.append("chart_pie.png")

    # 4️⃣ Line (trend)
    if len(num_cols) >= 1:
        fig = px.line(df, y=num_cols[0],
              title=f"Trend of {num_cols[0]}",
              color_discrete_sequence=["#00ff88"])
        fig.write_image("chart_line.png", width=1200, height=700)
        charts.append("chart_line.png")

    # 5️⃣ Heatmap
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        fig = px.imshow(corr,
                text_auto=True,
                color_continuous_scale="RdBu",
                title="Correlation Heatmap")
        fig.write_image("chart_heatmap.png", width=1200, height=700)
        charts.append("chart_heatmap.png")

    return charts

# ---------------- PDF GENERATOR ----------------
def create_pdf(df, insights, charts):
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # --- PAGE 1: COVER DESIGN ---
    p.setFillColorRGB(0.05, 0.15, 0.3)
    p.rect(0, height-250, width, 250, fill=1, stroke=0)
    
    p.setFont("Helvetica-Bold", 40)
    p.setFillColorRGB(1, 1, 1)
    p.drawString(60, height-100, "AURA ANALYTICS")
    
    p.setFont("Helvetica", 16)
    p.drawString(60, height-140, "Automated Data Intelligence & Insights Report")
    
    p.setFillColorRGB(0.96, 0.97, 0.98)
    p.roundRect(50, height-480, width-100, 180, 20, fill=1, stroke=0)
    
    p.setFillColorRGB(0, 0, 0)
    p.setFont("Helvetica-Bold", 20)
    p.drawString(80, height-350, "REPORT SUMMARY")
    
    p.setFont("Helvetica", 12)
    p.drawString(80, height-390, f"• Date Generated: {pd.Timestamp.now().strftime('%d %B, %Y')}")
    p.drawString(80, height-415, f"• Data Structure: {df.shape[0]} Rows and {df.shape[1]} Columns")
    p.drawString(80, height-440, f"• Health Score: {100 - (df.isnull().sum().sum()/df.size*100):.1f}% Data Density")

    p.setFont("Helvetica-Bold", 14)
    p.drawString(80, height-530, "TOP METRICS DETECTED:")
    
    num_cols_pdf = df.select_dtypes(include='number').columns
    m_labels = ["AVERAGE VALUE", "PEAK RECORD", "DATA RANGE"]
    m_values = [
        f"{df[num_cols_pdf[0]].mean():.2f}" if len(num_cols_pdf)>0 else "0.00",
        f"{df[num_cols_pdf[0]].max():.2f}" if len(num_cols_pdf)>0 else "0.00",
        f"{len(num_cols_pdf)} Metrics"
    ]

    x_pos = 80
    for label, val in zip(m_labels, m_values):
        p.setFillColorRGB(0.88, 0.93, 1.0)
        p.roundRect(x_pos, height-600, 145, 55, 10, fill=1, stroke=0)
        p.setFillColorRGB(0.05, 0.2, 0.5)
        p.setFont("Helvetica-Bold", 9)
        p.drawCentredString(x_pos + 72, height-565, label)
        p.setFont("Helvetica", 12)
        p.drawCentredString(x_pos + 72, height-585, val)
        x_pos += 165

    p.showPage() # Cover page khatam

    # --- PAGE 2+: CHARTS (MODIFIED LOOP) ---
    for img_path in charts:
     try:
        p.setFillColorRGB(0.05, 0.15, 0.3)
        p.rect(0, height-60, width, 60, fill=1, stroke=0)

        p.setFont("Helvetica-Bold", 14)
        p.setFillColorRGB(1, 1, 1)
        p.drawString(50, height-37, "VISUAL ANALYSIS")

        # ✅ FIXED LINE
        img = ImageReader(img_path)
        p.drawImage(img, 50, height-500, width=500, height=350)

        p.showPage()

     except Exception as e:
        p.setFont("Helvetica", 10)
        p.setFillColorRGB(1, 0, 0)
        p.drawString(50, height-300, f"Error rendering {img_path}: {str(e)}")
        p.showPage()

    p.save()
    buffer.seek(0)
    return buffer

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>🌌 AURA</h1>", unsafe_allow_html=True)
    theme_choice = st.radio("🎨 Theme", ["Dark", "Light"], horizontal=True)
    current_tpl = apply_theme(theme_choice)
    st.markdown("---")
    menu = st.radio("🚀 Menu", ["Dashboard", "Cleaning", "Visualizer", "Reports"])
    st.markdown("---")
    st.success("Aura AI Engine Active")

# ---------------- MAIN APP ----------------
st.title("🌌 Aura Analytics")
file = st.file_uploader("📂 Upload your Dataset", type=["csv", "xlsx"])

if file:
    if 'df' not in st.session_state:
        if file.name.endswith(".csv"):
            st.session_state.df = pd.read_csv(file)
        else:
            st.session_state.df = pd.read_excel(file)
    
    df = st.session_state.df
    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    
    st.success("✅ File Loaded Successfully!")

    if menu == "Dashboard":
        st.subheader("📊 Executive Overview")
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(f"<div class='kpi-box'>📝 Rows<br><h2>{df.shape[0]}</h2></div>", unsafe_allow_html=True)
        k2.markdown(f"<div class='kpi-box'>📐 Columns<br><h2>{df.shape[1]}</h2></div>", unsafe_allow_html=True)
        k3.markdown(f"<div class='kpi-box'>⚠️ Nulls<br><h2>{df.isnull().sum().sum()}</h2></div>", unsafe_allow_html=True)
        k4.markdown(f"<div class='kpi-box'>🔢 Numeric<br><h2>{len(num_cols)}</h2></div>", unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)

    elif menu == "Cleaning":
        st.subheader("🛠 Data Refinery (AI Powered)")
        col1, col2 = st.columns(2)
        with col1:
            dup_count = df.duplicated().sum()
            st.markdown(f"🔁 Duplicate Records: {dup_count}")
            if st.button("✨ Remove Duplicates"):
                st.session_state.df = df.drop_duplicates()
                st.success(f"{dup_count} rows removed")
                st.rerun()
        with col2:
            missing_total = df.isnull().sum().sum()
            st.markdown(f"⚠️ Missing Values: {missing_total}")
            if st.button("🩹 Fill Missing Values"):
                new_df = df.copy()
                new_df[num_cols] = new_df[num_cols].fillna(new_df[num_cols].mean())
                for col in cat_cols:
                    new_df[col] = new_df[col].fillna(new_df[col].mode()[0])
                st.session_state.df = new_df
                st.success("Missing values fixed")
                st.rerun()
        st.markdown("### 📊 Missing Data Breakdown")
        st.write(df.isnull().sum())

    elif menu == "Visualizer":
        st.subheader("🎨 Data Storytelling (Charts)")
        if num_cols:
            t1, t2, t3, t4 = st.tabs(["📊 Bar & Line", "🥧 Pie Chart", "📉 Scatter", "📦 Box Plot"])
            with t1:
                col_x = st.selectbox("Select X Axis", df.columns, key="vis_x")
                col_y = st.selectbox("Select Y Axis", num_cols, key="vis_y")
                c_mode = st.radio("Chart Type", ["Bar", "Line"], horizontal=True)
                if c_mode == "Bar":
                    fig = px.bar(df, x=col_x, y=col_y, template=current_tpl, color=col_x)
                else:
                    fig = px.line(df, x=col_x, y=col_y, template=current_tpl)
                st.plotly_chart(fig, use_container_width=True)

            with t2:
                pie_col = st.selectbox("Select Category Column", cat_cols if cat_cols else df.columns, key="vis_pie")
                if pie_col:
                    fig_pie = px.pie(df, names=pie_col, hole=0.4, template=current_tpl)
                    st.plotly_chart(fig_pie, use_container_width=True)

            with t3:
                if len(num_cols) >= 2:
                    st.plotly_chart(px.scatter(df, x=num_cols[0], y=num_cols[1], template=current_tpl), use_container_width=True)

            with t4:
                st.plotly_chart(px.box(df, y=num_cols[0], template=current_tpl), use_container_width=True)
        else:
            st.error("❌ Numeric data not found!")

        # ✅ FREE CHATBOT
        st.markdown("## 🤖 Ask Your Data (AI Chatbot)")
        user_q = st.text_input("💬 Ask anything about your data:")

        if user_q:
            with st.spinner("Thinking..."):
                answer = local_ai_chat(df, user_q)

                if "plot" in user_q.lower() or "chart" in user_q.lower():
                    st.plotly_chart(answer)
                else:
                    st.write(answer)

    elif menu == "Reports":
        st.subheader("📄 AI Strategic Deep-Dive")

        full_insights = generate_ai_insights(df, num_cols, cat_cols)
        st.info(f"✨ AI Detection: {full_insights}")

        if st.button("🚀 Generate PDF Report"):
            with st.spinner("Analyzing..."):
                paths = auto_generate_charts(df)
                pdf = create_pdf(df, full_insights, paths)

                st.download_button(
                    "📥 Download Report",
                    pdf,
                    "Aura_Executive_Report.pdf",
                    "application/pdf"
                )

else:
    st.info("👋 Welcome! Please upload a CSV or Excel file to start your analysis.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("<center>✨ Aura Analytics | Built by <b>Aparna Patel</b></center>", unsafe_allow_html=True)