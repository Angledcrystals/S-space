import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from scipy.ndimage import gaussian_filter
import argparse
import os

def spherical_to_cartesian(theta, phi, r=1):
    """Convert spherical coordinates to 3D cartesian."""
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.array([x, y, z])

def householder_reflection_3d(G, v):
    """Calculate the Householder reflection of 3D vector G across the hyperplane with normal v."""
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-10:
        return G
    v_unit = v / v_norm
    v_dot_G = np.dot(v_unit, G)
    return G - 2 * v_dot_G * v_unit

def project_3d_to_2d(G_3d, projection_type='stereographic'):
    """Project 3D vector to 2D using different projection methods."""
    if projection_type == 'stereographic':
        if G_3d[2] >= 1.0 - 1e-10:  # Near north pole
            return np.array([0, 0])
        denom = 1 - G_3d[2]
        return np.array([G_3d[0]/denom, G_3d[1]/denom])
    elif projection_type == 'orthographic':
        return np.array([G_3d[0], G_3d[1]])
    elif projection_type == 'cylindrical':
        theta = np.arctan2(G_3d[1], G_3d[0])
        z = G_3d[2]
        return np.array([theta, z])

def stereographic_inverse(S_2d):
    """Recover 3D point from 2D stereographic projection."""
    if np.linalg.norm(S_2d) < 1e-10:
        return np.array([0, 0, 1])
    
    norm_sq = np.dot(S_2d, S_2d)
    denom = 1 + norm_sq
    
    x = 2 * S_2d[0] / denom
    y = 2 * S_2d[1] / denom
    z = (norm_sq - 1) / denom
    
    return np.array([x, y, z])

def calculate_s_depth_mediated(S_2d, depth_method, nuit_radius, lum=0.5, sat=0.5, lum_depth_influence=0.0, sat_depth_influence=0.0):
    """
    Calculate depth value from S-coordinates using various methods, with optional mediation by luminosity and saturation.
    
    Args:
        S_2d: 2D S-coordinate array [S_x, S_y]
        depth_method: Method for calculating depth
        nuit_radius: Radius of the Nuit boundary circle
        lum: Pixel luminosity (0-1)
        sat: Pixel saturation (0-1)
        lum_depth_influence: Weight for luminosity's influence on depth
        sat_depth_influence: Weight for saturation's influence on depth
    
    Returns:
        depth: Depth value (0-1 range)
    """
    S_x, S_y = S_2d
    
    s_depth = 0.0 # Initialize
    
    if depth_method == 'nuit_distance':
        # Depth based on distance from Nuit boundary
        distance_from_origin = np.linalg.norm(S_2d)
        distance_from_nuit = abs(distance_from_origin - nuit_radius)
        # Closer to boundary = higher depth
        s_depth = 1.0 / (1.0 + distance_from_nuit * 5.0)
        
    elif depth_method == 'stereographic_z':
        # Use Z-coordinate from stereographic inverse projection
        S_3d = stereographic_inverse(S_2d)
        # Map Z from [-1, 1] to [0, 1]
        s_depth = (S_3d[2] + 1.0) / 2.0
        
    elif depth_method == 'radial_distance':
        # Simple radial distance from origin
        distance = np.linalg.norm(S_2d)
        s_depth = 1.0 / (1.0 + distance)
        
    elif depth_method == 'golden_spiral':
        # Depth based on golden spiral pattern
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        angle = np.arctan2(S_y, S_x)
        radius = np.linalg.norm(S_2d)
        
        # Calculate spiral depth
        spiral_factor = np.sin(angle * phi + radius * phi)
        s_depth = 0.5 + 0.5 * spiral_factor
        
    elif depth_method == 'intensity_weighted':
        # Depth based on coordinate magnitude with intensity weighting
        coord_magnitude = np.sqrt(S_x**2 + S_y**2)
        s_depth = np.tanh(coord_magnitude)
        
    else:
        # Default: simple coordinate-based depth
        s_depth = 0.5 + 0.25 * (S_x + S_y)
    
    # --- Mediation based on Luminosity and Saturation ---
    if lum_depth_influence > 0 or sat_depth_influence > 0:
        # Example mediation: Brighter/more saturated pixels get slightly higher depth
        # This pushes them 'forward'
        mediation_factor = 1.0 + (lum * lum_depth_influence + sat * sat_depth_influence)
        s_depth *= mediation_factor
        
    # Ensure depth is in [0, 1] range
    return np.clip(s_depth, 0.0, 1.0)

def pixel_to_s_coordinate_mediated(px, py, width, height, s_bounds, lum=0.5, sat=0.5, lum_map_weight=0.0, sat_map_weight=0.0):
    """
    Convert pixel coordinates to S-coordinate space, mediated by luminosity and saturation.
    
    Args:
        px, py: Pixel coordinates
        width, height: Image dimensions
        s_bounds: Bounds of S-coordinate space (tuple or single float)
        lum: Pixel luminosity (0-1)
        sat: Pixel saturation (0-1)
        lum_map_weight: Weight for luminosity's influence on S-coordinate mapping
        sat_map_weight: Weight for saturation's influence on S-coordinate mapping
        
    Returns:
        S_2d: 2D S-coordinate array [S_x, S_y]
    """
    if isinstance(s_bounds, tuple):
        s_min, s_max = s_bounds
    else:
        s_min, s_max = -s_bounds, s_bounds
    
    # Normalize pixel coordinates to [0, 1]
    norm_x = px / (width - 1)
    norm_y = py / (height - 1)
    
    # --- Mediation of S-coordinate mapping ---
    effective_s_min = s_min
    effective_s_max = s_max
    
    if lum_map_weight > 0 or sat_map_weight > 0:
        # Example Mediation: Higher lum/sat can expand the effective S-range for that pixel
        # This makes the S-coordinates for those pixels change more rapidly, potentially
        # leading to sharper gradients in depth for bright/saturated areas.
        mediation_factor = 1.0 + (lum * lum_map_weight + sat * sat_map_weight)
        
        # Ensure factor is reasonable
        mediation_factor = np.clip(mediation_factor, 0.8, 1.5) 
        
        center = (s_min + s_max) / 2.0
        current_range = (s_max - s_min)
        
        new_range = current_range * mediation_factor
        effective_s_min = center - new_range / 2.0
        effective_s_max = center + new_range / 2.0

    # Convert to S-coordinate space
    S_x = norm_x * (effective_s_max - effective_s_min) + effective_s_min
    S_y = (1 - norm_y) * (effective_s_max - effective_s_min) + effective_s_min  # Flip Y axis
    
    return np.array([S_x, S_y])

def calculate_adaptive_nuit_radius(s_coordinate_map, method='coverage_75'):
    """
    Calculate adaptive Nuit radius based on S-coordinate distribution.
    
    Args:
        s_coordinate_map: Array of S-coordinates [height, width, 2]
        method: Method for calculating radius
                - 'coverage_XX': Include XX% of points within circle
                - 'max_extent': Use maximum extent of coordinates
                - 'aspect_aware': Consider image aspect ratio
    
    Returns:
        nuit_radius: Calculated radius for Nuit boundary
    """
    s_coords_flat = s_coordinate_map.reshape(-1, 2)
    distances = np.linalg.norm(s_coords_flat, axis=1)
    
    if method.startswith('coverage_'):
        # Extract percentage from method name
        coverage_percent = int(method.split('_')[1])
        nuit_radius = np.percentile(distances, coverage_percent)
        
    elif method == 'max_extent':
        # Use 95% of maximum extent to avoid outliers
        nuit_radius = np.percentile(distances, 95)
        
    elif method == 'aspect_aware':
        # Consider the aspect ratio of the coordinate distribution
        s_x_range = np.ptp(s_coords_flat[:, 0])  # Peak-to-peak (max - min)
        s_y_range = np.ptp(s_coords_flat[:, 1])
        # Use the smaller range to ensure circle fits within bounds
        nuit_radius = min(s_x_range, s_y_range) / 2
        
    elif method == 'mean_distance':
        # Use mean distance plus one standard deviation
        nuit_radius = np.mean(distances) + np.std(distances)
        
    else:  # default
        nuit_radius = np.percentile(distances, 75)
    
    # Ensure minimum radius
    nuit_radius = max(nuit_radius, 0.1)
    
    return nuit_radius

def get_hsv_components(image_array):
    """
    Converts an RGB image array to HSV and returns normalized luminosity (Value) and saturation components.
    """
    hsv_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    
    # Normalize components to [0, 1]
    # Hue [0, 179] -> [0, 1] (not used for mediation here, but useful to return)
    # Saturation [0, 255] -> [0, 1]
    # Value (Luminosity) [0, 255] -> [0, 1]
    
    luminosity = hsv_image[:, :, 2].astype(np.float32) / 255.0 # Value channel from HSV
    saturation = hsv_image[:, :, 1].astype(np.float32) / 255.0 # Saturation channel from HSV
    hue = hsv_image[:, :, 0].astype(np.float32) / 179.0
    
    return luminosity, saturation, hue

def generate_s_depth_map(image_path, depth_method='nuit_distance', 
                        s_bounds=(-2.0, 2.0), smooth_sigma=1.0,
                        intensity_weight=0.3, # This is the global intensity blend weight
                        nuit_radius_method='coverage_75', fixed_nuit_radius=None,
                        use_color_mediation=False, # New flag
                        lum_map_weight=0.0, sat_map_weight=0.0,
                        lum_depth_influence=0.0, sat_depth_influence=0.0):
    """
    Generate a depth map from an image using S-coordinate analysis.
    
    Args:
        image_path: Path to input image
        depth_method: Method for calculating depth from S-coordinates
        s_bounds: Bounds of S-coordinate space
        smooth_sigma: Gaussian smoothing parameter
        intensity_weight: Weight for blending with original image intensity
        nuit_radius_method: Method for calculating adaptive Nuit radius
        fixed_nuit_radius: Fixed radius (overrides adaptive method if provided)
        use_color_mediation: Whether to mediate S-values by luminosity/saturation
        lum_map_weight: Weight for luminosity's influence on S-coordinate mapping
        sat_map_weight: Weight for saturation's influence on S-coordinate mapping
        lum_depth_influence: Weight for luminosity's influence on final depth value
        sat_depth_influence: Weight for saturation's influence on final depth value
    
    Returns:
        depth_map: Depth map as numpy array
        s_coordinate_map: 2D array of S-coordinates
        original_image: Original image array
        nuit_radius: Calculated or provided Nuit radius
    """
    # Load and process image
    try:
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_array = np.array(image)
        
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None, None, None

    height, width = image_array.shape[:2]
    
    # Get grayscale intensity for final blending
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    normalized_intensity = gray_image.astype(np.float32) / 255.0
    
    # Get luminosity and saturation maps for mediation
    luminosity_map = np.zeros((height, width), dtype=np.float32)
    saturation_map = np.zeros((height, width), dtype=np.float32)
    if use_color_mediation:
        luminosity_map, saturation_map, _ = get_hsv_components(image_array)
    
    # Initialize arrays
    depth_map = np.zeros((height, width), dtype=np.float32)
    s_coordinate_map = np.zeros((height, width, 2), dtype=np.float32)
    
    print(f"Processing image: {width}x{height} pixels")
    print(f"Depth method: {depth_method}")
    if use_color_mediation:
        print(f"Color Mediation Enabled: Map Weights (Lum:{lum_map_weight:.2f}, Sat:{sat_map_weight:.2f}), Depth Influence (Lum:{lum_depth_influence:.2f}, Sat:{sat_depth_influence:.2f})")
    print("Calculating S-coordinates...")
    
    # First pass: Calculate all S-coordinates (potentially mediated)
    for py in range(height):
        for px in range(width):
            current_lum = luminosity_map[py, px] if use_color_mediation else 0.5 # Default if not used
            current_sat = saturation_map[py, px] if use_color_mediation else 0.5 # Default if not used

            S_2d = pixel_to_s_coordinate_mediated(
                px, py, width, height, s_bounds, 
                current_lum, current_sat, 
                lum_map_weight, sat_map_weight
            )
            s_coordinate_map[py, px] = S_2d
    
    # Calculate adaptive Nuit radius (based on potentially mediated S-coords)
    if fixed_nuit_radius is not None:
        nuit_radius = fixed_nuit_radius
        print(f"Using fixed Nuit radius: {nuit_radius:.3f}")
    else:
        nuit_radius = calculate_adaptive_nuit_radius(s_coordinate_map, nuit_radius_method)
        print(f"Calculated adaptive Nuit radius ({nuit_radius_method}): {nuit_radius:.3f}")
    
    print("Calculating depth values...")
    
    # Second pass: Calculate depth values using the determined radius and mediation
    for py in range(height):
        if py % (height // 10) == 0:  # Progress indicator
            print(f"Progress: {py/height*100:.1f}%")
            
        for px in range(width):
            S_2d = s_coordinate_map[py, px]
            current_lum = luminosity_map[py, px] if use_color_mediation else 0.5
            current_sat = saturation_map[py, px] if use_color_mediation else 0.5
            
            # Calculate depth from S-coordinate, with optional mediation
            s_depth = calculate_s_depth_mediated(
                S_2d, depth_method, nuit_radius, 
                current_lum, current_sat, 
                lum_depth_influence, sat_depth_influence
            )
            
            # Blend with original image intensity (global weight)
            intensity = normalized_intensity[py, px]
            final_depth = (1 - intensity_weight) * s_depth + intensity_weight * intensity
            
            depth_map[py, px] = final_depth
    
    # Apply Gaussian smoothing
    if smooth_sigma > 0:
        depth_map = gaussian_filter(depth_map, sigma=smooth_sigma)
    
    print("Depth map generation complete!")
    return depth_map, s_coordinate_map, image_array, nuit_radius

def visualize_depth_map(depth_map, s_coordinate_map, original_image, nuit_radius,
                       save_path=None, show_plots=True):
    """Visualize the generated depth map and S-coordinate analysis."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original image
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Depth map
    depth_plot = axes[0, 1].imshow(depth_map, cmap='plasma', vmin=0, vmax=1)
    axes[0, 1].set_title('S-Coordinate Depth Map')
    axes[0, 1].axis('off')
    plt.colorbar(depth_plot, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # 3D depth visualization
    ax_3d = fig.add_subplot(2, 3, 3, projection='3d')
    h, w = depth_map.shape
    # Downsample for visualization performance
    step = max(1, min(h, w) // 50) 
    x_3d, y_3d = np.meshgrid(np.arange(0, w, step), np.arange(0, h, step))
    surface = ax_3d.plot_surface(x_3d, y_3d, 
                                depth_map[::step, ::step], cmap='plasma', alpha=0.8)
    ax_3d.set_title('3D Depth Surface')
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Depth')
    
    # S-coordinate X component
    s_x_plot = axes[1, 0].imshow(s_coordinate_map[:, :, 0], cmap='RdBu')
    axes[1, 0].set_title('S-Coordinate X Component')
    axes[1, 0].axis('off')
    plt.colorbar(s_x_plot, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # S-coordinate Y component
    s_y_plot = axes[1, 1].imshow(s_coordinate_map[:, :, 1], cmap='RdBu')
    axes[1, 1].set_title('S-Coordinate Y Component')
    axes[1, 1].axis('off')
    plt.colorbar(s_y_plot, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # Nuit boundary analysis with adaptive radius
    axes[1, 2].set_aspect('equal')
    
    # Calculate bounds for visualization
    s_coords_flat = s_coordinate_map.reshape(-1, 2)
    coord_range = np.max(np.abs(s_coords_flat))
    plot_range = max(coord_range * 1.1, nuit_radius * 1.2)
    
    axes[1, 2].set_xlim(-plot_range, plot_range)
    axes[1, 2].set_ylim(-plot_range, plot_range)
    
    # Sample S-coordinates for visualization
    sample_step = max(1, min(depth_map.shape) // 50)
    sample_s_coords = s_coordinate_map[::sample_step, ::sample_step].reshape(-1, 2)
    
    # Color points by their depth values
    sample_depths = depth_map[::sample_step, ::sample_step].flatten()
    scatter = axes[1, 2].scatter(sample_s_coords[:, 0], sample_s_coords[:, 1], 
                                c=sample_depths, cmap='plasma', s=1, alpha=0.6)
    
    # Add adaptive Nuit boundary circle
    circle = plt.Circle((0, 0), nuit_radius, fill=False, color='white', linewidth=2)
    axes[1, 2].add_patch(circle)
    axes[1, 2].set_title(f'S-Coordinates with Nuit Boundary (r={nuit_radius:.2f})')
    axes[1, 2].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1, 2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    if show_plots:
        plt.show()
    
    return fig

def save_depth_map(depth_map, output_path):
    """Save depth map as 16-bit grayscale image."""
    # Convert to 16-bit integer
    depth_16bit = (depth_map * 65535).astype(np.uint16)
    
    # Save as PNG
    cv2.imwrite(output_path, depth_16bit)
    print(f"Depth map saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate depth map using S-coordinates with adaptive Nuit radius and color mediation')
    parser.add_argument('image_path', help='Path to input image')
    parser.add_argument('--output_dir', default='./s_depth_output', 
                       help='Output directory for results')
    parser.add_argument('--depth_method', default='nuit_distance',
                       choices=['nuit_distance', 'stereographic_z', 'radial_distance', 
                               'golden_spiral', 'intensity_weighted'],
                       help='Method for calculating depth from S-coordinates')
    parser.add_argument('--s_bounds', type=float, default=2.0,
                       help='S-coordinate space bounds (symmetric)')
    parser.add_argument('--smooth_sigma', type=float, default=1.0,
                       help='Gaussian smoothing sigma (0 for no smoothing)')
    parser.add_argument('--intensity_weight', type=float, default=0.3,
                       help='Weight for blending with original image intensity')
    parser.add_argument('--nuit_radius_method', default='coverage_75',
                       choices=['coverage_50', 'coverage_60', 'coverage_70', 'coverage_75', 
                               'coverage_80', 'coverage_90', 'max_extent', 'aspect_aware', 'mean_distance'],
                       help='Method for calculating adaptive Nuit radius')
    parser.add_argument('--fixed_nuit_radius', type=float, default=None,
                       help='Fixed Nuit radius (overrides adaptive calculation)')
    parser.add_argument('--no_display', action='store_true',
                       help='Skip displaying plots')
    
    # New arguments for color mediation
    parser.add_argument('--use_color_mediation', action='store_true',
                       help='Enable mediation of S-values based on luminosity and saturation')
    parser.add_argument('--lum_map_weight', type=float, default=0.0,
                       help='Weight for luminosity influence on S-coordinate mapping (0-1, 0 for no influence)')
    parser.add_argument('--sat_map_weight', type=float, default=0.0,
                       help='Weight for saturation influence on S-coordinate mapping (0-1, 0 for no influence)')
    parser.add_argument('--lum_depth_influence', type=float, default=0.0,
                       help='Weight for luminosity influence on final depth value (0-1, 0 for no influence)')
    parser.add_argument('--sat_depth_influence', type=float, default=0.0,
                       help='Weight for saturation influence on final depth value (0-1, 0 for no influence)')

    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate depth map
    result = generate_s_depth_map(
        args.image_path,
        depth_method=args.depth_method,
        s_bounds=(-args.s_bounds, args.s_bounds),
        smooth_sigma=args.smooth_sigma,
        intensity_weight=args.intensity_weight,
        nuit_radius_method=args.nuit_radius_method,
        fixed_nuit_radius=args.fixed_nuit_radius,
        use_color_mediation=args.use_color_mediation,
        lum_map_weight=args.lum_map_weight,
        sat_map_weight=args.sat_map_weight,
        lum_depth_influence=args.lum_depth_influence,
        sat_depth_influence=args.sat_depth_influence
    )
    
    if result[0] is None:
        print("Failed to generate depth map")
        return
    
    depth_map, s_coordinate_map, original_image, nuit_radius = result
    
    # Generate output filenames
    base_name = os.path.splitext(os.path.basename(args.image_path))[0]
    depth_map_suffix = f"depth_{args.depth_method}"
    if args.fixed_nuit_radius is not None:
        depth_map_suffix += f"_fixedR{args.fixed_nuit_radius:.2f}"
    else:
        depth_map_suffix += f"_{args.nuit_radius_method}"
    
    if args.use_color_mediation:
        depth_map_suffix += f"_medL{args.lum_depth_influence:.2f}S{args.sat_depth_influence:.2f}"
        if args.lum_map_weight > 0 or args.sat_map_weight > 0:
            depth_map_suffix += f"mapL{args.lum_map_weight:.2f}S{args.sat_map_weight:.2f}"


    depth_map_path = os.path.join(args.output_dir, f"{base_name}_{depth_map_suffix}.png")
    visualization_path = os.path.join(args.output_dir, f"{base_name}_analysis_{depth_map_suffix}.png")
    
    # Save depth map
    save_depth_map(depth_map, depth_map_path)
    
    # Create and save visualization
    fig = visualize_depth_map(depth_map, s_coordinate_map, original_image, nuit_radius,
                             save_path=visualization_path, 
                             show_plots=not args.no_display)
    
    # Print statistics
    print(f"\nDepth Map Statistics:")
    print(f"Min depth: {np.min(depth_map):.4f}")
    print(f"Max depth: {np.max(depth_map):.4f}")
    print(f"Mean depth: {np.mean(depth_map):.4f}")
    print(f"Std depth: {np.std(depth_map):.4f}")
    
    # Analyze Nuit boundary alignment with adaptive radius
    s_coords_flat = s_coordinate_map.reshape(-1, 2)
    distances_from_nuit = np.abs(np.linalg.norm(s_coords_flat, axis=1) - nuit_radius)
    boundary_tolerance = nuit_radius * 0.1  # 10% of radius
    on_boundary_count = np.sum(distances_from_nuit < boundary_tolerance)
    
    print(f"\nNuit Boundary Analysis:")
    print(f"Adaptive Nuit radius: {nuit_radius:.3f}")
    print(f"Points near Nuit boundary (within {boundary_tolerance:.3f}): {on_boundary_count}")
    print(f"Percentage on boundary: {on_boundary_count/len(s_coords_flat)*100:.2f}%")
    
    # Additional coverage statistics
    distances = np.linalg.norm(s_coords_flat, axis=1)
    within_circle = np.sum(distances <= nuit_radius)
    print(f"Points within Nuit circle: {within_circle}")
    print(f"Coverage percentage: {within_circle/len(s_coords_flat)*100:.2f}%")
    
    print(f"\nResults saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
