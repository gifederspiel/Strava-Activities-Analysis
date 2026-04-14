import streamlit as st
import pandas as pd
import plotly.express as px
from scipy import stats

import data_processor as dp


st.set_page_config(page_title="Running Stats", layout="wide", menu_items={})


def format_pace(decimal_minutes):
    if pd.isna(decimal_minutes):
        return "–"
    minutes = int(decimal_minutes)
    seconds = int(round((decimal_minutes - minutes) * 60))
    return f"{minutes}:{seconds:02d} /km"


def chart_pace_trend(df):
    df = df.dropna(subset=["pace_min_per_km"]).copy()
    df["rolling_avg"] = df["pace_min_per_km"].rolling(window=10, min_periods=3).mean()

    fig = px.scatter(df, x="date", y="pace_min_per_km",
                     title="Pace over Time",
                     labels={"pace_min_per_km": "min / km", "date": ""},
                     hover_data={"pace_label": True, "pace_min_per_km": False})
    fig.add_scatter(x=df["date"], y=df["rolling_avg"],
                    mode="lines", name="10-run average", line=dict(width=2))
    fig.update_yaxes(autorange="reversed")
    return fig


def chart_weekly_volume(df):
    weekly = df.groupby("week_start")["distance_km"].sum().reset_index()
    weekly.columns = ["week", "km"]

    fig = px.bar(weekly, x="week", y="km",
                 title="Weekly Volume (km)",
                 labels={"week": "", "km": "km"})
    return fig


def chart_hr_distribution(df):
    df = df.dropna(subset=["average_heartrate"])

    fig = px.histogram(df, x="average_heartrate", nbins=25,
                       title="Heart Rate Distribution",
                       labels={"average_heartrate": "avg HR (bpm)"})
    return fig


def chart_hr_over_time(df):
    df = df.dropna(subset=["average_heartrate"]).copy()
    df["rolling_avg"] = df["average_heartrate"].rolling(window=8, min_periods=3).mean()

    fig = px.scatter(df, x="date", y="average_heartrate",
                     title="Heart Rate over Time",
                     labels={"average_heartrate": "avg HR (bpm)", "date": ""})
    fig.add_scatter(x=df["date"], y=df["rolling_avg"],
                    mode="lines", name="8-run average", line=dict(width=2))
    return fig


def chart_pace_vs_hr(df):
    df = df.dropna(subset=["pace_min_per_km", "average_heartrate"])
    if len(df) < 5:
        return None, None, None

    _, _, r, p_value, _ = stats.linregress(
        df["average_heartrate"], df["pace_min_per_km"]
    )

    fig = px.scatter(df, x="average_heartrate", y="pace_min_per_km",
                     trendline="ols",
                     title=f"Pace vs Heart Rate  (r = {r:.2f})",
                     labels={"average_heartrate": "avg HR (bpm)",
                             "pace_min_per_km": "min / km"})
    fig.update_yaxes(autorange="reversed")
    return fig, r, p_value


def chart_pace_regression(df):
    df = df.dropna(subset=["pace_min_per_km"]).copy()
    if len(df) < 5:
        return None, {}

    x = df["run_number"].values.astype(float)
    y = df["pace_min_per_km"].values

    slope, intercept, r, p_value, _ = stats.linregress(x, y)
    trend_line = slope * x + intercept

    days_total = (df["date"].max() - df["date"].min()).days or 1
    runs_per_month = (len(df) / days_total) * 30.44
    monthly_change = slope * runs_per_month

    fig = px.scatter(df, x="date", y="pace_min_per_km",
                     title="Pace Improvement — Linear Regression",
                     labels={"pace_min_per_km": "min / km", "date": ""},
                     hover_data={"pace_label": True, "pace_min_per_km": False})
    fig.add_scatter(x=df["date"], y=trend_line,
                    mode="lines", name="trend", line=dict(width=2, color="red"))
    fig.update_yaxes(autorange="reversed")

    result = {
        "r_squared":      r ** 2,
        "p_value":        p_value,
        "monthly_change": monthly_change,
        "first_pace":     intercept + slope * 1,
        "last_pace":      intercept + slope * len(df),
    }
    return fig, result


def show_upload_screen():
    st.title("Running Stats")
    st.write("Upload your Strava export to see your running statistics.")

    st.info(
        "**How to export from Strava:** "
        "Settings → My Account → Download or Delete Your Account → Request Your Archive. "
        "Open the downloaded ZIP and upload the `activities.csv` file inside it."
    )

    uploaded_file = st.file_uploader("Choose a file", type=["csv"])
    return uploaded_file


def show_dashboard(df):
    st.title("Running Stats")
    if st.button("Remove data"):
        del st.session_state["df"]
        st.rerun()

    hr_df = df.dropna(subset=["average_heartrate"])

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Runs",           f"{len(df):,}")
    col2.metric("Total distance", f"{df['distance_km'].sum():,.0f} km")
    col3.metric("Avg pace",       format_pace(df["pace_min_per_km"].mean()))
    col4.metric("Best pace",      format_pace(df["pace_min_per_km"].min()))
    col5.metric("Longest run",    f"{df['distance_km'].max():.1f} km")
    col6.metric("Avg HR",         f"{hr_df['average_heartrate'].mean():.0f} bpm" if len(hr_df) else "–")

    st.divider()

    st.subheader("Pace over Time")
    st.plotly_chart(chart_pace_trend(df), use_container_width=True)

    st.divider()

    st.subheader("Weekly Volume")
    st.plotly_chart(chart_weekly_volume(df), use_container_width=True)

    st.divider()

    st.subheader("Heart Rate Analysis")
    if len(hr_df) < 3:
        st.info("Not enough runs with heart rate data.")
    else:
        left, right = st.columns(2)
        with left:
            st.plotly_chart(chart_hr_distribution(df), use_container_width=True)
        with right:
            st.plotly_chart(chart_hr_over_time(df), use_container_width=True)

        st.write(f"Mean: **{hr_df['average_heartrate'].mean():.0f} bpm** · "
                 f"Min: **{hr_df['average_heartrate'].min():.0f} bpm** · "
                 f"Max: **{hr_df['average_heartrate'].max():.0f} bpm** · "
                 f"Std: **{hr_df['average_heartrate'].std():.1f} bpm**")

    st.divider()

    st.subheader("Pace vs Heart Rate")
    fig_corr, r, p = chart_pace_vs_hr(df)
    if fig_corr:
        st.plotly_chart(fig_corr, use_container_width=True)
        significant = "yes" if p < 0.05 else "no"
        direction = "higher HR → faster pace" if r < 0 else "higher HR → slower pace"
        st.write(f"r = **{r:.3f}** · R² = **{r**2:.3f}** · p = **{p:.4f}** · "
                 f"significant: **{significant}** · {direction}")
    else:
        st.info("Need at least 5 runs with HR data.")

    st.divider()

    st.subheader("Pace Improvement (Linear Regression)")
    fig_reg, reg = chart_pace_regression(df)
    if fig_reg:
        st.plotly_chart(fig_reg, use_container_width=True)

        monthly = reg["monthly_change"]
        direction = "faster" if monthly < 0 else "slower"
        net = abs(reg["last_pace"] - reg["first_pace"])
        significant = "yes" if reg["p_value"] < 0.05 else "no"

        st.write(f"R² = **{reg['r_squared']:.4f}** · "
                 f"p = **{reg['p_value']:.4f}** · "
                 f"significant: **{significant}**")
        st.write(f"Monthly change: **{abs(monthly):.3f} min/km {direction}** · "
                 f"Net improvement over all runs: **{format_pace(net)}** ({direction})")
    else:
        st.info("Need at least 5 runs.")

    st.divider()

    with st.expander("Show all runs"):
        display = df[["date", "name", "distance_km", "pace_label",
                      "duration_min", "average_heartrate"]].copy()
        display.columns = ["Date", "Name", "Distance (km)", "Pace",
                           "Duration (min)", "Avg HR"]
        display["Date"] = display["Date"].dt.strftime("%Y-%m-%d")
        st.dataframe(display, use_container_width=True, hide_index=True)


if "df" not in st.session_state:
    uploaded_file = show_upload_screen()

    if uploaded_file is not None:
        with st.spinner("Loading…"):
            try:
                df = dp.load_csv(uploaded_file.read())
                st.session_state["df"] = df
                st.rerun()
            except Exception as error:
                st.error(f"Could not read file: {error}")
else:
    show_dashboard(st.session_state["df"])
