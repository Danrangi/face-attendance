import streamlit as st
import pandas as pd
from helpers.db_utils import load_attendance

st.set_page_config(page_title="Attendance History")

st.title("📋 Attendance History & Logs")

# Load attendance records
df = load_attendance()

if df.empty:
    st.warning("🚫 No attendance records yet.")
    st.stop()

# Sidebar filters
st.sidebar.header("🔎 Filter Options")

students = ["All"] + sorted(df["Name"].unique().tolist())
selected_student = st.sidebar.selectbox("Select Student", students)

dates = ["All"] + sorted(df["Date"].unique(), reverse=True)
selected_date = st.sidebar.selectbox("Select Date", dates)

# Filter logic
if selected_student != "All":
    df = df[df["Name"] == selected_student]

if selected_date != "All":
    df = df[df["Date"] == selected_date]

# Show filtered data
st.subheader("🧾 Attendance Records")
st.dataframe(df, use_container_width=True)

# Stats
st.markdown("### 📊 Attendance Summary")
summary = df.groupby("Name").size().reset_index(name="Total Scans")
st.dataframe(summary)

# Export option
csv = df.to_csv(index=False)
st.download_button(
    label="⬇️ Download Attendance CSV",
    data=csv,
    file_name="attendance_records.csv",
    mime="text/csv"
)
