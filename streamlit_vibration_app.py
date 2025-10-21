
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.integrate import cumulative_trapezoid

st.set_page_config(page_title="Vibration Analysis", layout="wide")
st.title("Vibration Velocity Analysis App")
st.markdown("Upload an acceleration CSV file to convert to filtered velocity (in/s).")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, header=None)
    df.columns = ['Time_s', 'Accel_X_g', 'Accel_Y_g', 'Accel_Z_g']
    st.success("File loaded successfully.")

    st.sidebar.header("Filter Settings")
    lowcut = st.sidebar.slider("Low Cut Frequency (Hz)", 0.1, 20.0, 2.0)
    highcut = st.sidebar.slider("High Cut Frequency (Hz)", 20.0, 200.0, 99.0)
    order = st.sidebar.selectbox("Filter Order", [2, 4, 6], index=1)

    st.sidebar.header("Time Range")
    time_start = st.sidebar.number_input("Start Time (s)", value=float(df['Time_s'].min()))
    time_end = st.sidebar.number_input("End Time (s)", value=float(df['Time_s'].max()))

    df = df[(df['Time_s'] >= time_start) & (df['Time_s'] <= time_end)]

    g_to_in_s2 = 386.09
    for axis in ['X', 'Y', 'Z']:
        df[f'Accel_{axis}_in_s2'] = df[f'Accel_{axis}_g'] * g_to_in_s2

    fs = 1 / df['Time_s'].diff().mean()

    def butter_bandpass_filter(data, lowcut, highcut, fs, order):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)

    for axis in ['X', 'Y', 'Z']:
        df[f'Filtered_Accel_{axis}'] = butter_bandpass_filter(df[f'Accel_{axis}_in_s2'], lowcut, highcut, fs, order)
        df[f'Velocity_{axis}'] = cumulative_trapezoid(df[f'Filtered_Accel_{axis}'], df['Time_s'], initial=0)
        df[f'DC_Free_Velocity_{axis}'] = df[f'Velocity_{axis}'] - df[f'Velocity_{axis}'].mean()

    st.subheader("Filtered Velocity (DC Offset Removed)")
    fig, ax = plt.subplots(figsize=(10, 5))
    for axis in ['X', 'Y', 'Z']:
        ax.plot(df['Time_s'], df[f'DC_Free_Velocity_{axis}'], label=f'Velocity {axis} (in/s)')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (in/s)")
    ax.set_title("Filtered Velocity (Bandpass + DC Removal)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    peak_velocity = df[[f'DC_Free_Velocity_{axis}' for axis in ['X', 'Y', 'Z']]].abs().max().max()
    st.metric("Max Peak Velocity", f"{peak_velocity:.3f} in/s")

    with st.expander("Download Processed Data"):
        csv = df.to_csv(index=False)
        st.download_button("Download CSV", csv, file_name="processed_vibration_data.csv", mime="text/csv")
