"""
Streamlit Web Demo for Bat Swing Analysis
"""
import streamlit as st
import tempfile
import time
from pathlib import Path
import pandas as pd
import sys
import os

# --- NEW FIX: Prevent ML library conflict ---
os.environ["OMP_NUM_THREADS"] = "1"
# --- END OF FIX ---


# Add project root to sys.path to allow importing modules
sys.path.append(str(Path(__file__).parent))
try:
    from test_run import run_pipeline
except ImportError:
    st.error("Could not import the main 'test_run' pipeline. "
             "Make sure all project files are in the same directory.")
    st.stop()


st.set_page_config(layout="wide", page_title="Bat Swing Analysis")
st.title("⚾ Bat Swing Analysis Demo")

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("1. Upload Video")
    uploaded_file = st.file_uploader(
        "Upload a bat swing video", type=["mp4", "avi", "mov", "mkv"]
    )

    st.header("2. Set Parameters")
    
    # Help text for calibration
    st.info("""
    **How to find 'Scale' (meters/pixel):**
    1.  Measure an object of known length in your video (e..g., a bat $\approx$ 1.0m).
    2.  Find its length in pixels (use any image editor).
    3.  `Scale = Known Length (m) / Pixel Length`
    *Example: 1.0m / 250px = 0.004*
    """)
    
    scale = st.number_input(
        "Pixel-to-Meter Scale (meters/pixel)", 
        min_value=0.0001, 
        max_value=0.1, 
        value=0.003, # Updated default scale for cricket
        step=0.0001,
        format="%.4f",
        help="The conversion factor from pixels to meters."
    )
    
    yolo_model = st.text_input(
        "YOLO Model Path", 
        value="runs/detect/train3/weights/best.pt",
        help="Path to your custom .pt model file."
    )
    
    fps_override = st.number_input(
        "Override Video FPS (0 = auto)",
        min_value=0,
        max_value=300,
        value=0,
        help="Set to 0 to auto-detect FPS from the video file."
    )

    run_button = st.button("🚀 Run Analysis", type="primary")

# --- Main Area for Outputs ---
if run_button and uploaded_file is not None:
    with st.spinner("Analysis in progress... This may take a few minutes. ⏳"):
        # Create a temporary directory for this run
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            
            # Save uploaded file temporarily in chunks to save memory
            video_path = temp_dir / uploaded_file.name
            with open(video_path, "wb") as f:
                chunk_size = 1024 * 1024
                while True:
                    chunk = uploaded_file.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                
            # Define output directory
            output_dir = temp_dir / "analysis_results"
            
            start_time = time.time()
            
            try:
                # Run the main pipeline
                run_pipeline(
                    video_path=str(video_path),
                    yolo_model_path=yolo_model,
                    scale_mps=scale,
                    output_dir=output_dir,
                    fps_override=float(fps_override) if fps_override > 0 else None,
                    disable_progress_bar=True
                )
                
                end_time = time.time()
                st.success(f"Analysis complete in {end_time - start_time:.2f} seconds! 🎉")

                # --- Display Results ---
                metrics_csv_path = output_dir / "metrics.csv"
                
                if metrics_csv_path.exists():
                    st.header("📊 Swing Metrics")
                    df = pd.read_csv(metrics_csv_path)
                    
                    if 'time_series' in df.columns:
                        df = df.drop(columns=['time_series'])
                    st.dataframe(df.style.format({
                        "duration_s": "{:.2f}",
                        "peak_speed_mps": "{:.2f}",
                        "peak_angular_velocity_rps": "{:.2f}",
                        "mean_angular_velocity_rps": "{:.2f}",
                        "swing_angle_deg": "{:.1f}",
                        "smoothness_score": "{:.3f}",
                    }))
                    
                    st.header("📈 Visualizations")

                    # Show full video first
                    full_video = output_dir / "full_analysis_video.mp4"
                    full_plot = output_dir / "all_swings_plots.png"

                    if full_video.exists():
                        st.subheader("Full Analysis Video")
                        st.video(str(full_video))
                    
                    if full_plot.exists():
                        st.subheader("Combined Swing Plots")
                        st.image(str(full_plot))

                    st.divider()
                    st.subheader("Individual Swing Details")
                    
                    swing_ids = df["swing_id"].unique()
                    
                    for swing_id in swing_ids:
                        st.markdown(f"**Swing {swing_id}**")
                        
                        vid = output_dir / f"swing_{swing_id}_overlay.mp4"
                        gif = output_dir / f"swing_{swing_id}_overlay.gif"
                        plot = output_dir / f"swing_{swing_id}_plots.png"
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if vid.exists():
                                st.caption("Video")
                                st.video(str(vid))
                        with col2:
                            if gif.exists():
                                st.caption("GIF")
                                st.image(str(gif))
                        with col3:
                            if plot.exists():
                                st.caption("Plots")
                                st.image(str(plot))
                        st.divider()

                else:
                    st.error("Analysis finished, but the `metrics.csv` file was not created. No swings may have been detected.")

            except Exception as e:
                st.error(f"An error occurred during analysis:")
                st.exception(e)

elif run_button:
    st.warning("Please upload a video file first.")
else:
    st.info("Upload a video and set the parameters in the sidebar, then click 'Run Analysis'.")
