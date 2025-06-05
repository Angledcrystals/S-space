import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import argparse
from PIL import Image
import os
from scipy.ndimage import gaussian_filter

# --- Mathematical Utility Functions ---

def spherical_to_cartesian(theta_deg, phi_deg, r=1):
    """
    Converts spherical coordinates (longitude, latitude) to 3D Cartesian coordinates.
    theta_deg: longitude in degrees (0 to 360)
    phi_deg: latitude in degrees (-90 to 90)
    r: radius (default 1 for unit sphere)
    """
    theta_rad = np.deg2rad(theta_deg)
    phi_rad = np.deg2rad(phi_deg)
    x = r * np.cos(phi_rad) * np.cos(theta_rad)
    y = r * np.cos(phi_rad) * np.sin(theta_rad)
    z = r * np.sin(phi_rad)
    return np.array([x, y, z])

def stereographic_projection(x_3d, y_3d, z_3d):
    """
    Performs stereographic projection from the North Pole (0,0,1) to the z=0 plane.
    S_x = x / (1 - z)
    S_y = y / (1 - z)
    """
    if np.isclose(1 - z_3d, 0): # Avoid division by zero if point is at North Pole
        return np.array([0.0, 0.0]) # Or handle as infinity, but 0.0 is common for pole.
    S_x = x_3d / (1 - z_3d)
    S_y = y_3d / (1 - z_3d)
    return np.array([S_x, S_y])

def stereographic_inverse(S_2d):
    """
    Recovers 3D Cartesian coordinates from 2D stereographic projection.
    x = 2*S_x / (S_x^2 + S_y^2 + 1)
    y = 2*S_y / (S_x^2 + S_y^2 + 1)
    z = (S_x^2 + S_y^2 - 1) / (S_x^2 + S_y^2 + 1)
    """
    norm_sq = np.dot(S_2d, S_2d) # S_x^2 + S_y^2
    denom = norm_sq + 1
    x_3d = 2 * S_2d[0] / denom
    y_3d = 2 * S_2d[1] / denom
    z_3d = (norm_sq - 1) / denom
    return np.array([x_3d, y_3d, z_3d])

# --- S-Coordinate and Depth Calculation Functions ---

def pixel_to_s_coordinate_mediated(px, py, width, height, s_bounds, lum, sat, lum_map_weight, sat_map_weight):
    """
    Maps pixel coordinates (px, py) to S-coordinates, with optional color mediation.
    s_bounds can be a single float (symmetric) or a tuple (-min_val, max_val).
    """
    # Normalize pixel coordinates to -1 to 1 range, centered
    norm_x = (px / (width - 1)) * 2 - 1
    norm_y = (py / (height - 1)) * 2 - 1 # Y is typically inverted for image coordinates (0 at top)

    # Determine base S-coordinate range
    if isinstance(s_bounds, tuple):
        s_min_x, s_max_x = s_bounds[0], s_bounds[1]
        s_min_y, s_max_y = s_bounds[0], s_bounds[1] # Assuming symmetric for X and Y for s_bounds tuple
    else: # single float means symmetric bounds, e.g., -2.0 to 2.0
        s_min_x, s_max_x = -s_bounds, s_bounds
        s_min_y, s_max_y = -s_bounds, s_bounds

    # Apply color mediation to influence the S-coordinate mapping
    # This dynamically adjusts the 'effective' s_bounds for each pixel
    effective_s_max_x = s_max_x * (1 + lum * lum_map_weight + sat * sat_map_weight)
    effective_s_min_x = s_min_x * (1 + lum * lum_map_weight + sat * sat_map_weight)
    effective_s_max_y = s_max_y * (1 + lum * lum_map_weight + sat * sat_map_weight)
    effective_s_min_y = s_min_y * (1 + lum * lum_map_weight + sat * sat_map_weight)

    S_x = (norm_x + 1) / 2 * (effective_s_max_x - effective_s_min_x) + effective_s_min_x
    S_y = (norm_y + 1) / 2 * (effective_s_max_y - effective_s_min_y) + effective_s_min_y
    
    # Ensure S_x and S_y are clipped to reasonable, though expanded, bounds
    # Clipping against a slightly larger fixed bound for extreme color shifts
    clip_val = max(abs(s_max_x), abs(s_max_y)) * 5 # Allow some extreme expansion for mediation
    S_x = np.clip(S_x, -clip_val, clip_val)
    S_y = np.clip(S_y, -clip_val, clip_val)

    return np.array([S_x, S_y])

def calculate_adaptive_nuit_radius(s_coordinate_map, method='coverage_75', fixed_radius=None):
    """
    Calculates an adaptive Nuit radius based on the distribution of S-coordinates.
    s_coordinate_map: A 2D array of S_x, S_y coordinates (e.g., from generate_s_depth_map)
                      shape (N, 2) where N is number of pixels.
    method: 'coverage_XX' (e.g., 'coverage_75' for 75th percentile of distances)
            'max_extent' (uses a percentile of the max coordinate extent)
            'aspect_aware' (considers distribution's aspect ratio)
            'mean_distance' (mean + one std dev of distances from origin)
    fixed_radius: If provided, this value is used directly, overriding adaptive methods.
    """
    if fixed_radius is not None:
        return fixed_radius

    if not s_coordinate_map.size:
        return 1.0 # Default if no S-coords

    distances = np.linalg.norm(s_coordinate_map, axis=1) # Radial distance from S-origin

    if method.startswith('coverage_'):
        try:
            percentile = float(method.split('_')[1])
            if not (0 <= percentile <= 100):
                raise ValueError
            return np.percentile(distances, percentile)
        except (IndexError, ValueError):
            print(f"Warning: Invalid coverage method '{method}'. Falling back to 75th percentile.")
            return np.percentile(distances, 75)

    elif method == 'max_extent':
        max_abs_s_x = np.max(np.abs(s_coordinate_map[:, 0]))
        max_abs_s_y = np.max(np.abs(s_coordinate_map[:, 1]))
        # Use 95th percentile of the maximum dimension as a heuristic
        return max(max_abs_s_x, max_abs_s_y) * 0.95

    elif method == 'aspect_aware':
        std_x = np.std(s_coordinate_map[:, 0])
        std_y = np.std(s_coordinate_map[:, 1])
        if std_x == 0 and std_y == 0: return 1.0 # Avoid division by zero
        # Combine std devs, perhaps weighted by an overall scale
        return np.sqrt(std_x**2 + std_y**2) * 1.5 # Heuristic factor

    elif method == 'mean_distance':
        return np.mean(distances) + np.std(distances) # Mean plus one standard deviation

    else:
        print(f"Warning: Unknown Nuit radius method '{method}'. Falling back to coverage_75.")
        return np.percentile(distances, 75)


def calculate_s_depth_mediated(S_2d, depth_method, nuit_radius, nuit_align_s_point,
                                 s_bounds, # This is the bounds from the main mapping
                                 lum=0.5, sat=0.5, lum_depth_influence=0.0, sat_depth_influence=0.0,
                                 depth_falloff_factor=5.0, stereographic_mode='closest_center',
                                 blend_method_weights=None,
                                 num_steps=10): # Added for stepped_xy_s_coords

    S_x, S_y = S_2d[0], S_2d[1]
    
    # Helper to encapsulate single method depth calculation logic
    # This avoids repetitive code when blending
    def _calculate_single_method_depth(method_name, S_x, S_y, nuit_radius, nuit_align_s_point, s_bounds, depth_falloff_factor, stereographic_mode, num_steps):
        s_depth_val = 0.5 # Default neutral value

        if method_name == 'nuit_distance':
            # Depth highest near the Nuit boundary circle
            distance_from_origin = np.linalg.norm(np.array([S_x, S_y]))
            distance_from_nuit = abs(distance_from_origin - nuit_radius)
            s_depth_val = 1.0 / (1.0 + distance_from_nuit * depth_falloff_factor) # Inverse relation to distance from boundary

        elif method_name == 'nuit_aligned_point':
            # Depth highest at a specific S-coordinate point
            if nuit_align_s_point is None:
                align_point = np.array([0.0, 0.0])
            else:
                align_point = nuit_align_s_point
            distance_from_align_point = np.linalg.norm(np.array([S_x, S_y]) - align_point)
            s_depth_val = 1.0 / (1.0 + distance_from_align_point * depth_falloff_factor) # Inverse relation to distance from point

        elif method_name == 'stereographic_z':
            # Uses the Z component of the 3D sphere point after inverse projection
            S_3d = stereographic_inverse(np.array([S_x, S_y]))
            # Interpret Z differently based on mode
            if stereographic_mode == 'closest_center': # Z=1 at center (pole), Z=-1 at edges
                s_depth_val = (1.0 - S_3d[2]) / 2.0 # Maps Z from [-1,1] to depth [1,0], then inverted
            elif stereographic_mode == 'furthest_center': # Z=1 at center, Z=-1 at edges
                s_depth_val = (S_3d[2] + 1.0) / 2.0 # Maps Z from [-1,1] to depth [0,1]
            elif stereographic_mode == 'neutral_center_recede':
                s_depth_val = -0.25 * S_3d[2] + 0.25 # Custom mapping
            else: # Default to furthest_center if mode not recognized
                 s_depth_val = (S_3d[2] + 1.0) / 2.0

        elif method_name == 'radial_distance':
            # Depth based on simple radial distance from the S-origin
            distance = np.linalg.norm(np.array([S_x, S_y]))
            s_depth_val = 1.0 / (1.0 + distance * depth_falloff_factor) # Higher depth closer to center

        elif method_name == 'golden_spiral':
            # Generates depth based on a golden spiral pattern
            phi_ratio = (1 + np.sqrt(5)) / 2
            angle = np.arctan2(S_y, S_x)
            radius = np.linalg.norm(np.array([S_x, S_y]))
            # Create a wave along the spiral path
            spiral_factor = np.sin(angle * phi_ratio + radius * phi_ratio * 2.0)
            s_depth_val = 0.5 + 0.5 * spiral_factor # Maps sin output [-1,1] to [0,1]

        elif method_name == 'intensity_weighted':
            # Depth based on coordinate magnitude with tanh falloff, creating a center spike
            coord_magnitude = np.linalg.norm(np.array([S_x, S_y]))
            val = np.tanh(coord_magnitude * depth_falloff_factor)
            s_depth_val = 1.0 - val # Invert to make center high depth

        elif method_name == 's_x_as_depth':
            # Uses S_x directly as depth. Normalize to 0-1 range.
            s_val_max = s_bounds[1] if isinstance(s_bounds, tuple) else s_bounds
            s_val_min = s_bounds[0] if isinstance(s_bounds, tuple) else -s_bounds
            range_x = s_val_max - s_val_min
            if range_x == 0: s_depth_val = 0.5
            else: s_depth_val = (S_x - s_val_min) / range_x

        elif method_name == 's_y_as_depth':
            # Uses S_y directly as depth. Normalize to 0-1 range.
            s_val_max = s_bounds[1] if isinstance(s_bounds, tuple) else s_bounds
            s_val_min = s_bounds[0] if isinstance(s_bounds, tuple) else -s_bounds
            range_y = s_val_max - s_val_min
            if range_y == 0: s_depth_val = 0.5
            else: s_depth_val = (S_y - s_val_min) / range_y

        elif method_name == 'sum_xy_s_coords':
            # Depth based on the linear sum of S_x and S_y, creating a diagonal gradient
            s_val_max = s_bounds[1] if isinstance(s_bounds, tuple) else s_bounds
            s_val_min = s_bounds[0] if isinstance(s_bounds, tuple) else -s_bounds
            
            min_possible_sum = s_val_min + s_val_min
            max_possible_sum = s_val_max + s_val_max
            raw_sum = S_x + S_y
            range_of_sum = max_possible_sum - min_possible_sum
            
            if range_of_sum == 0:
                s_depth_val = 0.5
            else:
                s_depth_val = (raw_sum - min_possible_sum) / range_of_sum
        
        elif method_name == 'stepped_xy_s_coords': # <-- NEW GRADIENTLESS METHOD
            s_val_max = s_bounds[1] if isinstance(s_bounds, tuple) else s_bounds
            s_val_min = s_bounds[0] if isinstance(s_bounds, tuple) else -s_bounds
            
            # Normalize S_x and S_y to 0-1 range within their bounds
            range_x = s_val_max - s_val_min
            range_y = s_val_max - s_val_min
            
            if range_x == 0 or range_y == 0:
                s_depth_val = 0.5
            else:
                norm_S_x = (S_x - s_val_min) / range_x
                norm_S_y = (S_y - s_val_min) / range_y

                # Quantize values into 'num_steps' discrete levels
                # Use max(1, num_steps) to prevent issues if num_steps is 0 or less
                steps_x = np.floor(norm_S_x * max(1, num_steps))
                steps_y = np.floor(norm_S_y * max(1, num_steps))

                # Combine quantized steps (e.g., sum them)
                combined_steps = steps_x + steps_y
                
                # Normalize the combined steps to a 0-1 depth range
                max_combined_steps = 2 * (max(1, num_steps) - 1)
                
                if max_combined_steps <= 0:
                    s_depth_val = 0.5
                else:
                    s_depth_val = combined_steps / max_combined_steps
                    
        return s_depth_val # Return calculated depth for this single method


    # --- Blending Logic ---
    if blend_method_weights:
        final_s_depth = 0.0
        total_weight = 0.0
        
        for method, weight in blend_method_weights.items():
            if weight > 0: # Only calculate if weight is positive
                method_depth = _calculate_single_method_depth(method, S_x, S_y, nuit_radius, nuit_align_s_point, s_bounds, depth_falloff_factor, stereographic_mode, num_steps)
                final_s_depth += method_depth * weight
                total_weight += weight
        
        if total_weight > 0:
            s_depth = final_s_depth / total_weight # Normalize by total weight
        else:
            s_depth = 0.5 # Fallback if no valid methods or weights given
            
    else: # If no blend_method_weights, use the single depth_method as before
        s_depth = _calculate_single_method_depth(depth_method, S_x, S_y, nuit_radius, nuit_align_s_point, s_bounds, depth_falloff_factor, stereographic_mode, num_steps)


    # --- Mediation based on Luminosity and Saturation ---
    if lum_depth_influence > 0 or sat_depth_influence > 0:
        # A more robust way to blend color mediation
        if lum_depth_influence > 0:
            s_depth = (1.0 - lum_depth_influence) * s_depth + lum_depth_influence * lum
        if sat_depth_influence > 0:
            s_depth = (1.0 - sat_depth_influence) * s_depth + sat_depth_influence * sat

    # Ensure depth is in [0, 1] range after mediation
    return np.clip(s_depth, 0.0, 1.0)


# --- Main Generation and Visualization Functions ---

def generate_s_depth_map(image_path, depth_method='nuit_distance',
                         s_bounds=(-2.0, 2.0), smooth_sigma=1.0,
                         intensity_weight=0.3,
                         nuit_radius_method='coverage_75', fixed_nuit_radius=None,
                         nuit_align_s_point=None,
                         depth_falloff_factor=5.0,
                         stereographic_mode='closest_center',
                         use_color_mediation=False,
                         lum_map_weight=0.0, sat_map_weight=0.0,
                         lum_depth_influence=0.0, sat_depth_influence=0.0,
                         blend_method_weights=None, # Added for blending
                         num_steps=10): # Added for stepped_xy_s_coords

    # Load image
    try:
        original_image_pil = Image.open(image_path).convert("RGB")
        original_image = np.array(original_image_pil)
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return None, None, None, None
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None, None, None

    height, width, _ = original_image.shape
    depth_map = np.zeros((height, width), dtype=np.float32)
    s_coordinate_map = np.zeros((height * width, 2), dtype=np.float32)

    # Convert to grayscale for intensity blending
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    # Get HSV components for color mediation if enabled
    lum_map = np.zeros((height, width), dtype=np.float32)
    sat_map = np.zeros((height, width), dtype=np.float32)
    if use_color_mediation:
        hsv_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
        lum_map = hsv_image[:, :, 2] # V channel for luminosity
        sat_map = hsv_image[:, :, 1] # S channel for saturation

    # Calculate adaptive Nuit radius if needed (done once before pixel loop)
    # First, generate a preliminary s_coord_map for calculation if using adaptive Nuit
    if 'nuit_distance' in (blend_method_weights.keys() if blend_method_weights else [depth_method]):
        temp_s_coords = np.zeros((height * width, 2), dtype=np.float32)
        for py in range(height):
            for px in range(width):
                idx = py * width + px
                current_lum = lum_map[py, px] if use_color_mediation else 0.5
                current_sat = sat_map[py, px] if use_color_mediation else 0.5
                temp_s_coords[idx] = pixel_to_s_coordinate_mediated(
                    px, py, width, height, s_bounds,
                    current_lum, current_sat, lum_map_weight, sat_map_weight
                )
        nuit_radius_used = calculate_adaptive_nuit_radius(temp_s_coords, nuit_radius_method, fixed_nuit_radius)
        del temp_s_coords # Free memory
    else:
        nuit_radius_used = fixed_nuit_radius if fixed_nuit_radius is not None else 1.0 # Default if not used

    print(f"Nuit radius used: {nuit_radius_used:.4f}")

    # Main pixel loop to generate depth map
    for py in range(height):
        for px in range(width):
            idx = py * width + px

            current_lum = lum_map[py, px] if use_color_mediation else 0.5
            current_sat = sat_map[py, px] if use_color_mediation else 0.5
            
            # Map pixel to S-coordinate (potentially mediated by color)
            S_2d = pixel_to_s_coordinate_mediated(
                px, py, width, height, s_bounds,
                current_lum, current_sat, lum_map_weight, sat_map_weight
            )
            s_coordinate_map[idx] = S_2d

            # Calculate depth from S-coordinate, with optional mediation and blending
            s_depth = calculate_s_depth_mediated(
                S_2d, depth_method, nuit_radius_used, nuit_align_s_point,
                s_bounds, # s_bounds passed here
                current_lum, current_sat,
                lum_depth_influence, sat_depth_influence,
                depth_falloff_factor,
                stereographic_mode,
                blend_method_weights, # Pass blend weights
                num_steps # Pass num_steps
            )

            # Blend calculated S-depth with original image intensity
            final_depth = (1.0 - intensity_weight) * s_depth + intensity_weight * (1.0 - gray_image[py, px])
            
            depth_map[py, px] = np.clip(final_depth, 0.0, 1.0) # Ensure 0-1 range

    # Apply Gaussian smoothing if requested
    if smooth_sigma > 0:
        depth_map = gaussian_filter(depth_map, sigma=smooth_sigma)
        depth_map = np.clip(depth_map, 0.0, 1.0) # Re-clip after smoothing

    return depth_map, s_coordinate_map, original_image, nuit_radius_used


def visualize_depth_map(depth_map, s_coordinate_map, original_image, nuit_radius_used, depth_method, s_bounds, output_path=None, nuit_align_s_point=None):
    """
    Visualizes the generated depth map and related components.
    """
    if depth_map is None:
        print("No depth map to visualize.")
        return

    height, width = depth_map.shape

    fig = plt.figure(figsize=(18, 12)) # Adjusted for more plots
    
    # Plot 1: Original Image
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(original_image)
    ax1.set_title("Original Image")
    ax1.axis('off')

    # Plot 2: Generated Depth Map (Grayscale Colormap)
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(depth_map, cmap='gray') # Grayscale is typical for depth maps
    ax2.set_title("Generated Depth Map")
    ax2.axis('off')

    # Plot 3: 3D Surface Plot of Depth Map
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    X = np.arange(0, width, 1)
    Y = np.arange(0, height, 1)
    X, Y = np.meshgrid(X, Y)
    ax3.plot_surface(X, Y, depth_map, cmap=cm.viridis, rstride=1, cstride=1, antialiased=True)
    ax3.set_title("3D Depth Surface")
    ax3.set_zlabel("Depth")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.view_init(elev=50, azim=-120) # Adjust view for better perspective

    # Plot 4: S-coordinate Map (X component as image)
    ax4 = fig.add_subplot(2, 3, 4)
    S_x_map = s_coordinate_map[:, 0].reshape(height, width)
    ax4.imshow(S_x_map, cmap='RdBu', origin='lower') # RdBu for divergent, origin='lower' for correct S-coord display
    ax4.set_title("S_x Coordinate Map")
    ax4.axis('off')
    # Add colorbar for S_x
    cbar4 = fig.colorbar(ax4.imshow(S_x_map, cmap='RdBu', origin='lower'), ax=ax4, shrink=0.7)
    cbar4.set_label('S_x Value')


    # Plot 5: S-coordinate Map (Y component as image)
    ax5 = fig.add_subplot(2, 3, 5)
    S_y_map = s_coordinate_map[:, 1].reshape(height, width)
    ax5.imshow(S_y_map, cmap='RdBu', origin='lower') # RdBu for divergent
    ax5.set_title("S_y Coordinate Map")
    ax5.axis('off')
    # Add colorbar for S_y
    cbar5 = fig.colorbar(ax5.imshow(S_y_map, cmap='RdBu', origin='lower'), ax=ax5, shrink=0.7)
    cbar5.set_label('S_y Value')


    # Plot 6: S-coordinate Scatter Plot with Nuit Circle/Align Point
    ax6 = fig.add_subplot(2, 3, 6)
    # Color points by their depth
    scatter_colors = depth_map.flatten()
    scatter = ax6.scatter(s_coordinate_map[:, 0], s_coordinate_map[:, 1], c=scatter_colors, cmap='gray', s=1, alpha=0.5)
    ax6.set_title("S-coordinate Scatter Plot (Colored by Depth)")
    ax6.set_xlabel("S_x")
    ax6.set_ylabel("S_y")
    ax6.axvline(0, color='grey', linestyle='--', linewidth=0.5)
    ax6.axhline(0, color='grey', linestyle='--', linewidth=0.5)
    
    # Set limits based on S_bounds to show the full conceptual space
    if isinstance(s_bounds, tuple):
        ax6.set_xlim(s_bounds[0], s_bounds[1])
        ax6.set_ylim(s_bounds[0], s_bounds[1])
    else:
        ax6.set_xlim(-s_bounds, s_bounds)
        ax6.set_ylim(-s_bounds, s_bounds)

    # Add Nuit circle or alignment point if relevant to the method
    if 'nuit_distance' in depth_method or ('nuit_distance' in (args.blend_methods if args.blend_methods else '')):
        circle = plt.Circle((0, 0), nuit_radius_used, color='red', fill=False, linestyle='--', linewidth=2, label=f'Nuit Circle (R={nuit_radius_used:.2f})')
        ax6.add_artist(circle)
        ax6.legend(loc='upper right')
    elif 'nuit_aligned_point' in depth_method or ('nuit_aligned_point' in (args.blend_methods if args.blend_methods else '')):
        if nuit_align_s_point is not None:
            ax6.plot(nuit_align_s_point[0], nuit_align_s_point[1], 'ro', markersize=10, label=f'Align Point ({nuit_align_s_point[0]:.2f},{nuit_align_s_point[1]:.2f})')
            ax6.legend(loc='upper right')

    fig.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"Visualization saved to: {output_path}")
    # Show plot (if not saving)
    plt.show()

def save_depth_map(depth_map, output_path):
    """
    Saves the depth map as a 16-bit grayscale PNG.
    """
    if depth_map is None:
        print("No depth map to save.")
        return

    # Normalize to 0-65535 for 16-bit PNG
    depth_map_16bit = (depth_map * 65535).astype(np.uint16)
    
    # Using Pillow to save 16-bit grayscale
    img_pil = Image.fromarray(depth_map_16bit, mode='I;16') # I;16 is 16-bit grayscale
    img_pil.save(output_path)
    print(f"Depth map saved as 16-bit PNG: {output_path}")

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Generate a depth map from an image using S-coordinate analysis.")
    parser.add_argument('image_path', type=str, help='Path to the input image.')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save results.')
    
    # Core depth method parameters
    parser.add_argument('--depth_method', type=str, default='intensity_weighted',
                        choices=['nuit_distance', 'nuit_aligned_point', 'stereographic_z',
                                 'radial_distance', 'golden_spiral', 'intensity_weighted',
                                 's_x_as_depth', 's_y_as_depth', 'sum_xy_s_coords', 'stepped_xy_s_coords'], # <--- ADDED stepped_xy_s_coords
                        help='Primary method for calculating depth from S-coordinates (used if no blend_weights).')
    parser.add_argument('--s_bounds', type=float, default=2.0,
                        help='Symmetric bounds for S-coordinate space mapping (e.g., 2.0 means -2.0 to 2.0 for both X and Y).')
    parser.add_argument('--smooth_sigma', type=float, default=1.0,
                        help='Standard deviation for Gaussian smoothing of the depth map (0 for no smoothing).')
    parser.add_argument('--intensity_weight', type=float, default=0.3,
                        help='Weight to blend original image intensity into the depth map (0 for pure S-depth, 1 for pure intensity).')

    # Nuit-specific parameters
    parser.add_argument('--nuit_radius_method', type=str, default='coverage_75',
                        choices=['coverage_75', 'coverage_90', 'max_extent', 'aspect_aware', 'mean_distance'],
                        help='Method to calculate Nuit radius for "nuit_distance" depth. (e.g., coverage_75, max_extent).')
    parser.add_argument('--fixed_nuit_radius', type=float, default=None,
                        help='Use a fixed Nuit radius instead of calculating adaptively. Overrides --nuit_radius_method.')
    parser.add_argument('--nuit_align_s_x', type=float, default=0.0,
                        help='S_x coordinate for "nuit_aligned_point" method.')
    parser.add_argument('--nuit_align_s_y', type=float, default=0.0,
                        help='S_y coordinate for "nuit_aligned_point" method.')

    # Depth Falloff and Stereographic Mode
    parser.add_argument('--depth_falloff_factor', type=float, default=5.0,
                        help='Controls the steepness of depth falloff for methods like nuit_distance, radial_distance, intensity_weighted.')
    parser.add_argument('--stereographic_mode', type=str, default='closest_center',
                        choices=['closest_center', 'furthest_center', 'neutral_center_recede'],
                        help='Interpretation of Z component for "stereographic_z" method.')

    # Color Mediation Parameters
    parser.add_argument('--use_color_mediation', action='store_true',
                        help='Enable color (luminosity/saturation) to influence S-coordinate mapping and/or depth calculation.')
    parser.add_argument('--lum_map_weight', type=float, default=0.0,
                        help='Influence of luminosity on S-coordinate mapping (0-1).')
    parser.add_argument('--sat_map_weight', type=float, default=0.0,
                        help='Influence of saturation on S-coordinate mapping (0-1).')
    parser.add_argument('--lum_depth_influence', type=float, default=0.0,
                        help='Influence of luminosity on final depth value (0-1).')
    parser.add_argument('--sat_depth_influence', type=float, default=0.0,
                        help='Influence of saturation on final depth value (0-1).')

    # Blending Parameters
    parser.add_argument('--blend_methods', type=str, default='',
                        help='Comma-separated list of method:weight pairs for blending (e.g., "nuit_distance:0.7,sum_xy_s_coords:0.3"). Overrides --depth_method if provided.')

    # Stepped method parameters
    parser.add_argument('--num_steps', type=int, default=10, # <--- ADDED num_steps ARG
                        help='Number of discrete depth steps for "stepped_xy_s_coords" method. A smaller number means fewer, larger steps.')

    # Output and Display
    parser.add_argument('--no_display', action='store_true',
                        help='Do not display matplotlib plots.')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare nuit_align_s_point if needed
    nuit_align_s_point = np.array([args.nuit_align_s_x, args.nuit_align_s_y])

    # Prepare blend_method_weights dictionary
    blend_method_weights = None
    if args.blend_methods:
        blend_method_weights = {}
        try:
            for item in args.blend_methods.split(','):
                method, weight_str = item.split(':')
                method = method.strip()
                if method not in parser.choices['depth_method']:
                    print(f"Warning: Unknown method '{method}' in --blend_methods. Skipping.")
                    continue
                blend_method_weights[method] = float(weight_str.strip())
            print(f"Blending methods: {blend_method_weights}")
            if not blend_method_weights:
                print("No valid blend methods specified. Falling back to primary depth_method.")
                blend_method_weights = None
        except ValueError:
            print("Error parsing --blend_methods. Ensure format is 'method:weight,method2:weight2'. Using primary depth_method instead.")
            blend_method_weights = None # Fallback to single method

    # Generate depth map
    depth_map, s_coordinate_map, original_image, nuit_radius_used = generate_s_depth_map(
        args.image_path,
        depth_method=args.depth_method,
        s_bounds=(-args.s_bounds, args.s_bounds), # Pass s_bounds as a tuple
        smooth_sigma=args.smooth_sigma,
        intensity_weight=args.intensity_weight,
        nuit_radius_method=args.nuit_radius_method,
        fixed_nuit_radius=args.fixed_nuit_radius,
        nuit_align_s_point=nuit_align_s_point,
        depth_falloff_factor=args.depth_falloff_factor,
        stereographic_mode=args.stereographic_mode,
        use_color_mediation=args.use_color_mediation,
        lum_map_weight=args.lum_map_weight,
        sat_map_weight=args.sat_map_weight,
        lum_depth_influence=args.lum_depth_influence,
        sat_depth_influence=args.sat_depth_influence,
        blend_method_weights=blend_method_weights, # Pass blend weights
        num_steps=args.num_steps # <--- PASS num_steps
    )

    if depth_map is None:
        return # Exit if image loading failed

    # Save results
    base_name = os.path.splitext(os.path.basename(args.image_path))[0]
    depth_map_output_path = os.path.join(args.output_dir, f"{base_name}_depth.png")
    visualization_output_path = os.path.join(args.output_dir, f"{base_name}_viz.png")

    save_depth_map(depth_map, depth_map_output_path)

    if not args.no_display:
        print("\nDisplaying results...")
        visualize_depth_map(
            depth_map, s_coordinate_map, original_image, nuit_radius_used, 
            args.depth_method, # Pass depth_method for visualization logic
            args.s_bounds, # Pass s_bounds for scatter plot limits
            visualization_output_path,
            nuit_align_s_point=nuit_align_s_point
        )
    else:
        print(f"Visualization not displayed. Saved to: {visualization_output_path}")

    # Additional stats
    print("\n--- Depth Map Statistics ---")
    print(f"Min depth: {depth_map.min():.4f}")
    print(f"Max depth: {depth_map.max():.4f}")
    print(f"Mean depth: {depth_map.mean():.4f}")

    # S-coordinate distribution check (optional)
    if s_coordinate_map.size > 0:
        distances = np.linalg.norm(s_coordinate_map, axis=1)
        # Check coverage for 75th percentile (a common measure)
        nuit_radius_check = np.percentile(distances, 75)
        within_circle = np.sum(distances <= nuit_radius_check)
        print(f"S-coords within 75th percentile radius ({nuit_radius_check:.2f}): {within_circle/len(s_coordinate_map)*100:.2f}%")
        print(f"S-coords min X: {s_coordinate_map[:,0].min():.2f}, max X: {s_coordinate_map[:,0].max():.2f}")
        print(f"S-coords min Y: {s_coordinate_map[:,1].min():.2f}, max Y: {s_coordinate_map[:,1].max():.2f}")


    print(f"\nResults saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
