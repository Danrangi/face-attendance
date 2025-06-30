import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import logging
from helpers.db_utils import load_attendance

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_prepare_data():
    """Load and prepare attendance data"""
    try:
        df = load_attendance()
        if df.empty:
            return False, "No attendance records available.", None
        
        # Convert date and time columns
        df['Date'] = pd.to_datetime(df['Date'])
        df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.time
        df['Day'] = df['Date'].dt.day_name()
        
        return True, "", df
    except Exception as e:
        logger.error(f"Error loading attendance data: {str(e)}")
        return False, f"Error loading attendance data: {str(e)}", None

def create_attendance_visualizations(df: pd.DataFrame):
    """Create attendance visualizations"""
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            # Daily attendance trend
            daily_counts = df.groupby('Date').size().reset_index(name='Count')
            fig_daily = px.line(daily_counts, x='Date', y='Count',
                              title='Daily Attendance Trend',
                              labels={'Count': 'Number of Check-ins'})
            st.plotly_chart(fig_daily, use_container_width=True)

        with col2:
            # Weekly pattern
            weekly_pattern = df.groupby('Day').size()
            weekly_pattern = weekly_pattern.reindex(['Monday', 'Tuesday', 'Wednesday', 
                                                   'Thursday', 'Friday', 'Saturday', 'Sunday'])
            fig_weekly = px.bar(weekly_pattern, title='Weekly Attendance Pattern',
                              labels={'value': 'Number of Check-ins', 'Day': 'Day of Week'})
            st.plotly_chart(fig_weekly, use_container_width=True)

    except Exception as e:
        logger.error(f"Error creating visualizations: {str(e)}")
        st.error("Failed to create visualizations")

def main():
    st.title("📋 Attendance History & Analytics")
    
    # Load data
    success, message, df = load_and_prepare_data()
    if not success:
        st.warning(message)
        return

    # Sidebar filters
    st.sidebar.header("🔎 Filter Options")

    # Date range filter
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(df['Date'].min(), df['Date'].max()),
        min_value=df['Date'].min(),
        max_value=df['Date'].max()
    )

    # Student filter
    students = ["All"] + sorted(df["Name"].unique().tolist())
    selected_student = st.sidebar.selectbox("Select Student", students)

    # Apply filters
    filtered_df = df.copy()
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['Date'].dt.date >= start_date) & 
            (filtered_df['Date'].dt.date <= end_date)
        ]
    
    if selected_student != "All":
        filtered_df = filtered_df[filtered_df["Name"] == selected_student]

    # Display statistics
    st.header("📊 Attendance Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(filtered_df))
    with col2:
        st.metric("Unique Students", filtered_df["Name"].nunique())
    with col3:
        st.metric("Date Range", f"{filtered_df['Date'].min().strftime('%Y-%m-%d')} to {filtered_df['Date'].max().strftime('%Y-%m-%d')}")

    # Visualizations
    st.header("📈 Attendance Analytics")
    create_attendance_visualizations(filtered_df)

    # Detailed Records
    st.header("🧾 Detailed Records")
    st.dataframe(filtered_df[["Name", "Matric No", "Date", "Time", "Status"]], use_container_width=True)

    # Export options
    st.download_button(
        label="⬇️ Download Records (CSV)",
        data=filtered_df.to_csv(index=False),
        file_name=f"attendance_records_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main() 