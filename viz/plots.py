"""
Functions for plotting swing metrics using Matplotlib.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional

def plot_swing_analytics(
    time_series_data: Dict[str, np.ndarray],
    output_path: str,
    title: str = "Swing Analytics"
):
    """
    Generates and saves a 2-panel plot:
    1. Bat-tip linear speed vs. time.
    2. Bat-tip angular velocity and angle vs. time.

    Args:
        time_series_data (Dict[str, np.ndarray]): Dictionary containing
            'time_s', 'speed_mps', 'angular_velocity_rps', 'angle_deg'.
        output_path (str): Path to save the .png file.
        title (str): Title for the plot.
    """
    time_s = time_series_data["time_s"]
    speed_mps = time_series_data["speed_mps"]
    ang_vel_rps = time_series_data["angular_velocity_rps"]
    angle_deg = time_series_data["angle_deg"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(title, fontsize=16)

    # --- Plot 1: Linear Speed ---
    color1 = 'tab:blue'
    ax1.plot(time_s, speed_mps, color=color1, lw=2)
    ax1.set_ylabel('Bat-Tip Speed (m/s)', color=color1, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, linestyle=':')
    ax1.set_title('Linear Speed', loc='left', fontsize=14)
    
    # Highlight peak speed
    peak_idx = np.argmax(speed_mps)
    peak_t = time_s[peak_idx]
    peak_v = speed_mps[peak_idx]
    ax1.plot(peak_t, peak_v, 'o', color='tab:red', markersize=8, label=f'Peak: {peak_v:.2f} m/s')
    ax1.legend()


    # --- Plot 2: Angular Velocity and Angle ---
    color2 = 'tab:green'
    ax2.plot(time_s, ang_vel_rps, color=color2, lw=2, label='Ang. Velocity (rad/s)')
    ax2.set_ylabel('Angular Velocity (rad/s)', color=color2, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.grid(True, linestyle=':')
    ax2.set_title('Angular Velocity & Angle', loc='left', fontsize=14)

    # Create a twin axis for the angle
    ax3 = ax2.twinx()
    color3 = 'tab:purple'
    ax3.plot(time_s, angle_deg, color=color3, lw=2, linestyle='--', label='Angle (deg)')
    ax3.set_ylabel('Shoulder-Tip Angle (deg)', color=color3, fontsize=12)
    ax3.tick_params(axis='y', labelcolor=color3)

    # Add legend for ax2 and ax3
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax3.legend(lines + lines2, labels + labels2, loc='upper left')

    
    ax2.set_xlabel('Time (s)', fontsize=12)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
    
    try:
        plt.savefig(output_path, dpi=150)
    except Exception as e:
        print(f"Error saving plot: {e}")
    
    plt.close(fig) # Close the figure to free up memory