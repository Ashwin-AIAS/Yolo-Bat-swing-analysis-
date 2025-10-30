"""
Unit tests for the core metrics calculations.
Run with: pytest
"""
import numpy as np
import pytest
from metrics.swing_metrics import (
    PixelScaler,
    compute_derivatives,
    compute_angular_velocity,
    detect_swings,
    compute_smoothness_score,
)
from utils.geometry import interpolate_points

@pytest.fixture
def basic_scaler():
    """A simple scaler: 200 pixels = 1.0 meter."""
    return PixelScaler(pixel_ref=200.0, meter_ref=1.0)

def test_pixel_scaler(basic_scaler):
    assert basic_scaler.scale == 0.005  # 1.0 / 200.0
    assert basic_scaler.to_meters(100) == 0.5
    assert basic_scaler.to_mps(pixels_per_frame=10, fps=30) == (10 * 0.005 * 30) # 1.5

def test_pixel_scaler_errors():
    with pytest.raises(ValueError):
        PixelScaler(pixel_ref=0, meter_ref=1.0)
    with pytest.raises(ValueError):
        PixelScaler(pixel_ref=-100, meter_ref=1.0)

def test_interpolate_points():
    points = np.array([
        [1.0, 1.0],
        [np.nan, np.nan],
        [3.0, 5.0]
    ])
    expected = np.array([
        [1.0, 1.0],
        [2.0, 3.0],
        [3.0, 5.0]
    ])
    result = interpolate_points(points)
    np.testing.assert_allclose(result, expected)

def test_interpolate_at_ends():
    points = np.array([
        [np.nan, np.nan],
        [1.0, 10.0],
        [np.nan, np.nan]
    ])
    expected = np.array([
        [1.0, 10.0],
        [1.0, 10.0],
        [1.0, 10.0]
    ])
    result = interpolate_points(points)
    np.testing.assert_allclose(result, expected)

def test_compute_derivatives_constant_velocity():
    # Constant velocity: 10 pixels/frame in x, 0 in y
    # 5 frames, fps=1
    pos = np.array([
        [0.0, 0.0],
        [10.0, 0.0],
        [20.0, 0.0],
        [30.0, 0.0],
        [40.0, 0.0]
    ])
    fps = 1.0
    
    pos_smooth, speed_px, accel_px = compute_derivatives(pos, fps)
    
    # Speed should be ~10 pixels/frame
    # np.gradient will have edge effects, so we check the middle
    np.testing.assert_allclose(speed_px[1:-1], 10.0, rtol=0.1)
    
    # Acceleration should be ~0
    np.testing.assert_allclose(accel_px[1:-1], 0.0, atol=0.1)

def test_compute_derivatives_with_nans():
    pos = np.array([
        [0.0, 0.0],
        [np.nan, np.nan],
        [20.0, 0.0],
        [np.nan, np.nan],
        [40.0, 0.0]
    ])
    fps = 1.0
    
    pos_smooth, speed_px, accel_px = compute_derivatives(pos, fps)
    
    # After interpolation, trajectory should be [0, 10, 20, 30, 40]
    # Speed should be ~10
    np.testing.assert_allclose(speed_px[1:-1], 10.0, rtol=0.1)
    np.testing.assert_allclose(accel_px[1:-1], 0.0, atol=0.1)

def test_compute_angular_velocity():
    # Simple 90-degree rotation in 2 steps (3 frames)
    # Shoulder is static at (0, 0)
    # Tip moves from (10, 0) -> (0, 10) -> (-10, 0)
    # Angles: 0 rad -> pi/2 rad -> pi rad
    fps = 1.0
    shoulder = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    tip = np.array([[10.0, 0.0], [0.0, 10.0], [-10.0, 0.0]])
    
    angles_rad, ang_vel_rps = compute_angular_velocity(shoulder, tip, fps)
    
    expected_angles = np.array([0.0, np.pi/2, np.pi])
    np.testing.assert_allclose(angles_rad, expected_angles, atol=1e-6)
    
    # Angular velocity should be (pi/2) rad/frame = (pi/2) rad/s
    # Gradient will give [pi/2, pi/2, pi/2]
    expected_vel = np.array([np.pi/2, np.pi/2, np.pi/2])
    np.testing.assert_allclose(ang_vel_rps, expected_vel, atol=1e-6)

def test_detect_swings():
    speeds = np.array([0.1, 0.1, 0.8, 0.9, 1.0, 0.8, 0.2, 0.1, 0.6, 0.7, 0.1])
    thresh = 0.5
    min_frames = 2
    
    swings = detect_swings(speeds, thresh, min_frames)
    
    # Note: scipy.ndimage.label finds inclusive start/end indices
    assert swings == [(2, 5), (8, 9)]

def test_detect_swings_no_swings():
    speeds = np.array([0.1, 0.2, 0.3, 0.4])
    swings = detect_swings(speeds, speed_thresh_mps=0.5, min_swing_frames=2)
    assert swings == []

def test_detect_swings_too_short():
    speeds = np.array([0.1, 0.6, 0.2, 0.8, 0.9, 0.1])
    swings = detect_swings(speeds, speed_thresh_mps=0.5, min_swing_frames=2)
    assert swings == [(3, 4)] # Only the 2-frame swing is detected

def test_compute_smoothness_score():
    # A perfect sine wave (smooth) should have low entropy -> high score
    t = np.linspace(0, 1, 1000)
    smooth_signal = np.sin(2 * np.pi * 5 * t)
    score_smooth = compute_smoothness_score(smooth_signal)
    
    # Random noise (jerky) should have high entropy -> low score
    jerky_signal = np.random.rand(1000)
    score_jerky = compute_smoothness_score(jerky_signal)
    
    # A constant signal (perfectly smooth)
    constant_signal = np.ones(1000)
    score_constant = compute_smoothness_score(constant_signal)
    
    assert score_constant == 1.0
    assert score_smooth > 0.8  # Should be very high
    assert score_jerky < 0.2  # Should be very low
    assert score_smooth > score_jerky