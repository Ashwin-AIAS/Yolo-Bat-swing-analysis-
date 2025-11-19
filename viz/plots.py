"""
Functions for plotting swing metrics using Matplotlib.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List
import math

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

# --- THIS IS THE FUNCTION YOU WERE MISSING ---
def plot_all_swing_analytics(
    all_swing_metrics: List[Dict[str, Any]],
    output_path: str,
    title: str = "Full Swing Analysis Report"
):
    """
    Generates and saves one single image with a grid of plots for all swings.
    """
    num_swings = len(all_swing_metrics)
    if num_swings == 0:
        return

    # Create a grid of plots: 2 plots per swing (speed, angle)
    # We'll make it 2 columns wide
    num_rows = num_swings
    num_cols = 2
    
    # Calculate figure height based on number of swings
    fig_height = 5 * num_rows
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, fig_height), squeeze=False)
    fig.suptitle(title, fontsize=20, y=0.98) # Adjusted y position

    for i, metrics in enumerate(all_swing_metrics):
        time_series_data = metrics["time_series"]
        time_s = time_series_data["time_s"]
        speed_mps = time_series_data["speed_mps"]
        ang_vel_rps = time_series_data["angular_velocity_rps"]
        
        swing_title = f"Swing {metrics['swing_id']} (Frames {metrics['start_frame']}-{metrics['end_frame']})"

        # --- Plot 1: Linear Speed (Left Column) ---
        ax1 = axes[i, 0]
        color1 = 'tab:blue'
        ax1.plot(time_s, speed_mps, color=color1, lw=2)
        ax1.set_ylabel('Bat-Tip Speed (m/s)', color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, linestyle=':')
        ax1.set_title(swing_title + " - Linear Speed", loc='left', fontsize=14)
        
        peak_idx = np.argmax(speed_mps)
        peak_t = time_s[peak_idx]
        peak_v = speed_mps[peak_idx]
        ax1.plot(peak_t, peak_v, 'o', color='tab:red', markersize=8, label=f'Peak: {peak_v:.2f} m/s')
        ax1.legend()
        if i == num_rows - 1: # Only add x-label to bottom plot
            ax1.set_xlabel('Time (s)')

        # --- Plot 2: Angular Velocity (Right Column) ---
        ax2 = axes[i, 1]
        color2 = 'tab:green'
        ax2.plot(time_s, ang_vel_rps, color=color2, lw=2, label='Ang. Velocity (rad/s)')
        ax2.set_ylabel('Angular Velocity (rad/s)', color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.grid(True, linestyle=':')
        ax2.set_title(swing_title + " - Angular Velocity", loc='left', fontsize=14)
        ax2.legend()
        if i == num_rows - 1: # Only add x-label to bottom plot
            ax2.set_xlabel('Time (s)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust for suptitle
    
    try:
        plt.savefig(output_path, dpi=150)
    except Exception as e:
        print(f"Error saving all-swings plot: {e}")
    
    plt.close(fig) # Close the figure to free up memory