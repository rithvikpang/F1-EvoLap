"""Module for visualizing lap simulation results and genetic algorithm progress."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

def plot_lap_simulation_results(sim, track, racing_line):
    """
    Comprehensive visualization of lap simulation results.
    Shows speed maps, limiting factors, and physics breakdown.
    
    Args:
        sim: LapSimulator object with results
        track: Track object
        racing_line: Racing line coordinates
    """
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig)
    
    # Calculate cumulative distance
    distances = np.zeros(len(racing_line))
    for i in range(1, len(racing_line)):
        distances[i] = distances[i-1] + np.linalg.norm(racing_line[i] - racing_line[i-1])
    
    # Plot 1: Speed-colored track map (left side)
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.plot(track.x_left, track.y_left, 'gray', linewidth=1, alpha=0.5)
    ax1.plot(track.x_right, track.y_right, 'gray', linewidth=1, alpha=0.5)
    scatter = ax1.scatter(racing_line[:,0], racing_line[:,1], 
                         c=sim.v_final*3.6, cmap='jet', s=20, zorder=5)
    plt.colorbar(scatter, ax=ax1, label='Speed (km/h)')
    ax1.axis('equal')
    ax1.set_title('Speed Map', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    
    # Plot 2: Speed trace
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(distances, sim.v_final * 3.6, 'b-', linewidth=2)
    ax2.fill_between(distances, 0, sim.v_final * 3.6, alpha=0.3)
    ax2.set_xlabel('Distance (m)')
    ax2.set_ylabel('Speed (km/h)')
    ax2.set_title('Speed Trace', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Speed limit breakdown
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(distances, sim.v_corner * 3.6, 'r--', linewidth=1.5, 
             label='Cornering', alpha=0.7)
    ax3.plot(distances, sim.v_accel * 3.6, 'g--', linewidth=1.5, 
             label='Acceleration', alpha=0.7)
    ax3.plot(distances, sim.v_brake * 3.6, 'orange', linewidth=1.5, 
             label='Braking', alpha=0.7)
    ax3.plot(distances, sim.v_final * 3.6, 'b-', linewidth=2, 
             label='Final', zorder=5)
    ax3.set_xlabel('Distance (m)')
    ax3.set_ylabel('Speed (km/h)')
    ax3.set_title('Speed Limits', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Limiting factor map
    ax4 = fig.add_subplot(gs[1, 1])
    limiting_factor = np.zeros(len(sim.v_final))
    for i in range(len(sim.v_final)):
        diffs = [
            abs(sim.v_final[i] - sim.v_corner[i]),
            abs(sim.v_final[i] - sim.v_accel[i]),
            abs(sim.v_final[i] - sim.v_brake[i])
        ]
        limiting_factor[i] = np.argmin(diffs)
    
    colors = ['red', 'green', 'orange']
    labels = ['Cornering', 'Acceleration', 'Braking']
    for i, (color, label) in enumerate(zip(colors, labels)):
        mask = limiting_factor == i
        if np.any(mask):
            ax4.scatter(distances[mask], sim.v_final[mask] * 3.6, 
                       c=color, s=10, label=label, alpha=0.7)
    
    ax4.set_xlabel('Distance (m)')
    ax4.set_ylabel('Speed (km/h)')
    ax4.set_title('Limiting Factor', fontsize=12, fontweight='bold')
    ax4.legend(loc='best', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Curvature
    ax5 = fig.add_subplot(gs[1, 2])
    curvature = 1 / np.maximum(sim.radius, 1)  # Avoid division by large numbers
    ax5.plot(distances, curvature, 'purple', linewidth=1.5)
    ax5.set_xlabel('Distance (m)')
    ax5.set_ylabel('Curvature (1/m)')
    ax5.set_title('Track Curvature', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, max(curvature) * 1.1)
    
    plt.tight_layout()
    plt.show()


def plot_genetic_algorithm_progress(generation_data):
    """
    Plot genetic algorithm optimization progress over generations.
    
    Args:
        generation_data: List of dicts with keys:
            - generation: int
            - best_lap_time: float
            - avg_lap_time: float
            - worst_lap_time: float
            - best_params: dict
    """
    generations = [d['generation'] for d in generation_data]
    best_times = [d['best_lap_time'] for d in generation_data]
    avg_times = [d['avg_lap_time'] for d in generation_data]
    worst_times = [d['worst_lap_time'] for d in generation_data]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Lap time evolution
    ax1.plot(generations, best_times, 'g-', linewidth=2, label='Best', marker='o')
    ax1.plot(generations, avg_times, 'b--', linewidth=1.5, label='Average')
    ax1.plot(generations, worst_times, 'r:', linewidth=1.5, label='Worst')
    ax1.fill_between(generations, best_times, worst_times, alpha=0.2)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Lap Time (s)')
    ax1.set_title('Lap Time Evolution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Improvement per generation
    if len(best_times) > 1:
        improvements = [0] + [best_times[i-1] - best_times[i] 
                             for i in range(1, len(best_times))]
        ax2.bar(generations, improvements, color='green', alpha=0.6)
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Lap Time Improvement (s)')
        ax2.set_title('Per-Generation Improvement', fontsize=12, fontweight='bold')
        ax2.axhline(0, color='black', linewidth=0.5)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_parameter_evolution(generation_data, param_names):
    """
    Plot how specific vehicle parameters evolved over generations.
    
    Args:
        generation_data: List of generation dicts
        param_names: List of parameter names to track (e.g., ['CD', 'CL', 'brake_bias'])
    """
    generations = [d['generation'] for d in generation_data]
    
    n_params = len(param_names)
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_params > 1 else [axes]
    
    for idx, param_name in enumerate(param_names):
        ax = axes[idx]
        values = [d['best_params'].get(param_name, None) for d in generation_data]
        
        if all(v is not None for v in values):
            ax.plot(generations, values, 'b-', linewidth=2, marker='o')
            ax.set_xlabel('Generation')
            ax.set_ylabel(param_name)
            ax.set_title(f'{param_name} Evolution', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'{param_name}\nNo data', 
                   ha='center', va='center', transform=ax.transAxes)
    
    # Hide unused subplots
    for idx in range(n_params, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_speed_comparison(sim1, sim2, racing_line, labels=['Baseline', 'Optimized']):
    """
    Compare two lap simulations side-by-side.
    Useful for comparing before/after GA optimization.
    
    Args:
        sim1: First LapSimulator (baseline)
        sim2: Second LapSimulator (optimized)
        racing_line: Racing line coordinates
        labels: Labels for each simulation
    """
    distances = np.zeros(len(racing_line))
    for i in range(1, len(racing_line)):
        distances[i] = distances[i-1] + np.linalg.norm(racing_line[i] - racing_line[i-1])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Speed comparison
    ax1.plot(distances, sim1.v_final * 3.6, 'b-', linewidth=2, label=labels[0])
    ax1.plot(distances, sim2.v_final * 3.6, 'r-', linewidth=2, label=labels[1])
    ax1.fill_between(distances, sim1.v_final * 3.6, sim2.v_final * 3.6, 
                     where=(sim2.v_final > sim1.v_final), 
                     color='green', alpha=0.3, label='Faster')
    ax1.fill_between(distances, sim1.v_final * 3.6, sim2.v_final * 3.6, 
                     where=(sim2.v_final <= sim1.v_final), 
                     color='red', alpha=0.3, label='Slower')
    ax1.set_xlabel('Distance (m)')
    ax1.set_ylabel('Speed (km/h)')
    ax1.set_title('Speed Comparison', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Speed delta
    speed_delta = (sim2.v_final - sim1.v_final) * 3.6
    ax2.plot(distances, speed_delta, 'purple', linewidth=2)
    ax2.fill_between(distances, 0, speed_delta, 
                     where=(speed_delta > 0), color='green', alpha=0.3)
    ax2.fill_between(distances, 0, speed_delta, 
                     where=(speed_delta <= 0), color='red', alpha=0.3)
    ax2.axhline(0, color='black', linewidth=1, linestyle='--')
    ax2.set_xlabel('Distance (m)')
    ax2.set_ylabel('Speed Difference (km/h)')
    ax2.set_title(f'{labels[1]} vs {labels[0]}', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_vehicle_parameters_radar(vehicle_params, param_ranges=None):
    """
    Radar chart showing vehicle parameter settings.
    
    Args:
        vehicle_params: Dict of vehicle parameters
        param_ranges: Dict of (min, max) for each parameter for normalization
    """
    # Select key parameters to display
    display_params = {
        'CD': vehicle_params.get('CD', 0.85),
        'CL': vehicle_params.get('CL', 3.5),
        'brake_bias': vehicle_params.get('brake_bias', 0.56),
        'front_spring': vehicle_params.get('front_spring_rate', 95000) / 1000,
        'rear_spring': vehicle_params.get('rear_spring_rate', 105000) / 1000,
        'ers_deploy': vehicle_params.get('ers_max_deploy', 120000) / 1000,
    }
    
    categories = list(display_params.keys())
    values = list(display_params.values())
    
    # Normalize if ranges provided
    if param_ranges:
        normalized_values = []
        for cat, val in zip(categories, values):
            if cat in param_ranges:
                min_val, max_val = param_ranges[cat]
                normalized = (val - min_val) / (max_val - min_val)
                normalized_values.append(normalized)
            else:
                normalized_values.append(val)
        values = normalized_values
    
    # Radar chart
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]  # Complete the circle
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    ax.plot(angles, values, 'o-', linewidth=2, color='blue')
    ax.fill(angles, values, alpha=0.25, color='blue')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1 if param_ranges else None)
    ax.set_title('Vehicle Parameter Configuration', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()


def save_results_summary(sim, lap_time, vehicle_params, filename='results_summary.txt'):
    """
    Save a text summary of simulation results.
    
    Args:
        sim: LapSimulator object
        lap_time: Lap time in seconds
        vehicle_params: Dict of vehicle parameters
        filename: Output filename
    """
    with open(filename, 'w') as f:
        f.write("="*60 + "\n")
        f.write("LAP SIMULATION RESULTS SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Lap Time: {lap_time:.3f} seconds ({lap_time/60:.3f} minutes)\n\n")
        
        f.write("Speed Statistics:\n")
        f.write(f"  Max Speed: {sim.v_final.max() * 3.6:.1f} km/h\n")
        f.write(f"  Min Speed: {sim.v_final.min() * 3.6:.1f} km/h\n")
        f.write(f"  Avg Speed: {sim.v_final.mean() * 3.6:.1f} km/h\n\n")
        
        f.write("Vehicle Parameters:\n")
        for key, value in sorted(vehicle_params.items()):
            f.write(f"  {key}: {value}\n")
        
        f.write("\n" + "="*60 + "\n")
    
    print(f"Results summary saved to {filename}")