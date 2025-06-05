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

def compute_s_coordinate_proper(px, py, width, height, image_intensity=None,
                              reflection_normal=None, projection_type='stereographic'):
    """
    Proper S-coordinate computation using the mathematical pipeline.
    
    This implements the actual S-coordinate transformation:
    1. Map pixel to unit sphere via spherical coordinates
    2. Apply optional Householder reflection 
    3. Project to 2D S-coordinates via stereographic/other projection
    
    Args:
        px, py: Pixel coordinates
        width, height: Image dimensions  
        image_intensity: Optional image intensity for coordinate modulation
        reflection_normal: Normal vector for Householder reflection (optional)
        projection_type: Type of 3D to 2D projection
    
    Returns:
        S_2d: Final S-coordinate
        intermediate_results: Dict with intermediate computation results
    """
    
    # Step 1: Map pixel to spherical coordinates on unit sphere
    # Use image intensity and position to determine spherical mapping
    norm_x = px / (width - 1)
    norm_y = py / (height - 1)
    
    # Create spherical coordinates with proper mapping
    # Use intensity to modulate the mapping if available
    if image_intensity is not None:
        # Intensity affects the spherical mapping
        theta = 2 * np.pi * norm_x * (0.5 + 0.5 * image_intensity)
        phi = np.pi * norm_y * (0.5 + 0.5 * image_intensity)
    else:
        theta = 2 * np.pi * norm_x
        phi = np.pi * norm_y
    
    # Step 2: Convert to 3D point on unit sphere
    G_3d = spherical_to_cartesian(theta, phi)
    
    # Step 3: Apply Householder reflection if specified
    if reflection_normal is not None:
        G_3d_reflected = householder_reflection_3d(G_3d, reflection_normal)
    else:
        G_3d_reflected = G_3d
    
    # Step 4: Project 3D sphere point to 2D S-coordinates
    S_2d = project_3d_to_2d(G_3d_reflected, projection_type)
    
    # Store intermediate computation results
    intermediate_results = {
        'theta': theta,
        'phi': phi, 
        'G_3d_original': G_3d,
        'G_3d_reflected': G_3d_reflected,
        'reflection_applied': reflection_normal is not None,
        'intensity_used': image_intensity
    }
    
    return S_2d, intermediate_results

def calculate_s_depth(S_2d, depth_method='nuit_distance', intermediate_results=None):
    """
    Calculate depth value from S-coordinates using various methods.
    
    Args:
        S_2d: 2D S-coordinate array [S_x, S_y]
        depth_method: Method for calculating depth
        intermediate_results: Optional intermediate computation results
    
    Returns:
        depth: Depth value (0-1 range)
    """
    S_x, S_y = S_2d
    
    if depth_method == 'nuit_distance':
        # Depth based on distance from Nuit boundary (radius = 1)
        distance_from_origin = np.linalg.norm(S_2d)
        distance_from_nuit = abs(distance_from_origin - 1.0)
        # Closer to boundary = higher depth
        depth = 1.0 / (1.0 + distance_from_nuit * 5.0)
        
    elif depth_method == 'stereographic_z':
        # Use Z-coordinate from stereographic inverse projection
        S_3d = stereographic_inverse(S_2d)
        # Map Z from [-1, 1] to [0, 1]
        depth = (S_3d[2] + 1.0) / 2.0
        
    elif depth_method == 'radial_distance':
        # Simple radial distance from origin
        distance = np.linalg.norm(S_2d)
        depth = 1.0 / (1.0 + distance)
        
    elif depth_method == 'golden_spiral':
        # Depth based on golden spiral pattern
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        angle = np.arctan2(S_y, S_x)
        radius = np.linalg.norm(S_2d)
        
        # Calculate spiral depth
        spiral_factor = np.sin(angle * phi + radius * phi)
        depth = 0.5 + 0.5 * spiral_factor
        
    elif depth_method == 'intensity_weighted':
        # Depth based on coordinate magnitude with intensity weighting
        coord_magnitude = np.sqrt(S_x**2 + S_y**2)
        depth = np.tanh(coord_magnitude)
        
    elif depth_method == 'spherical_phi':
        # Use original spherical phi coordinate if available
        if intermediate_results and 'phi' in intermediate_results:
            phi = intermediate_results['phi']
            depth = phi / np.pi  # Normalize to [0, 1]
        else:
            depth = 0.5 + 0.25 * (S_x + S_y)
            
    elif depth_method == 'reflection_magnitude':
        # Depth based on magnitude of reflection transformation
        if intermediate_results and intermediate_results.get('reflection_applied', False):
            original = intermediate_results['G_3d_original']
            reflected = intermediate_results['G_3d_reflected']
            reflection_magnitude = np.linalg.norm(reflected - original)
            depth = np.tanh(reflection_magnitude)
        else:
            depth = calculate_s_depth(S_2d, 'nuit_distance')
            
    else:
        # Default: simple coordinate-based depth
        depth = 0.5 + 0.25 * (S_x + S_y)
    
    # Ensure depth is in [0, 1] range
    return np.clip(depth, 0.0, 1.0)

def pixel_to_s_coordinate(px, py, width, height, s_bounds=(-2.0, 2.0)):
    """Convert pixel coordinates to S-coordinate space (legacy method for compatibility)."""
    if isinstance(s_bounds, tuple):
        s_min, s_max = s_bounds
    else:
        s_min, s_max = -s_bounds, s_bounds
    
    # Normalize pixel coordinates to [0, 1]
    norm_x = px / (width - 1)
    norm_y = py / (height - 1)
    
    # Convert to S-coordinate space
    S_x = norm_x * (s_max - s_min) + s_min
    S_y = (1 - norm_y) * (s_max - s_min) + s_min  # Flip Y axis
    
    return np.array([S_x, S_y])

def generate_reflection_normal(method='random', image_intensity=None, px=None, py=None):
    """
    Generate reflection normal vector for Householder reflection.
    
    Args:
        method: Method for generating normal ('random', 'image_based', 'fixed')
        image_intensity: Normalized image intensity (for image_based method)
        px, py: Pixel coordinates (for image_based method)
    
    Returns:
        normal: 3D normal vector
    """
    if method == 'random':
        # Random unit vector
        normal = np.random.randn(3)
        normal = normal / np.linalg.norm(normal)
        
    elif method == 'image_based' and image_intensity is not None:
        # Normal based on image intensity and position
        normal = np.array([
            image_intensity * np.cos(px * 0.01),
            image_intensity * np.sin(py * 0.01),
            1.0 - image_intensity
        ])
        normal = normal / np.linalg.norm(normal)
        
    elif method == 'fixed':
        # Fixed normal vector
        normal = np.array([0.5, 0.5, 0.707])  # Diagonal direction
        normal = normal / np.linalg.norm(normal)
        
    else:
        # Default to no reflection
        return None
    
    return normal

def generate_s_depth_map(image_path, depth_method='nuit_distance', 
                        s_bounds=(-2.0, 2.0), smooth_sigma=1.0,
                        use_image_intensity=True, intensity_weight=0.3,
                        use_full_pipeline=True, projection_type='stereographic',
                        reflection_method='none'):
    """
    Generate a depth map from an image using S-coordinate analysis.
    
    Args:
        image_path: Path to input image
        depth_method: Method for calculating depth from S-coordinates
        s_bounds: Bounds of S-coordinate space (for legacy method)
        smooth_sigma: Gaussian smoothing parameter
        use_image_intensity: Whether to blend with original image intensity
        intensity_weight: Weight for image intensity blending
        use_full_pipeline: Whether to use the complete S-coordinate pipeline
        projection_type: Type of 3D to 2D projection
        reflection_method: Method for generating reflection normal
    
    Returns:
        depth_map: Depth map as numpy array
        s_coordinate_map: 2D array of S-coordinates
        original_image: Original image array
        analysis_data: Additional analysis data
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
    
    # Convert to grayscale for intensity information
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    normalized_intensity = gray_image.astype(np.float32) / 255.0
    
    # Initialize arrays
    depth_map = np.zeros((height, width), dtype=np.float32)
    s_coordinate_map = np.zeros((height, width, 2), dtype=np.float32)
    
    # Analysis data storage
    analysis_data = {
        'spherical_coords': np.zeros((height, width, 2), dtype=np.float32),
        'reflection_applied': np.zeros((height, width), dtype=bool),
        'projection_type': projection_type,
        'reflection_method': reflection_method
    }
    
    print(f"Processing image: {width}x{height} pixels")
    print(f"Depth method: {depth_method}")
    print(f"Using full pipeline: {use_full_pipeline}")
    print(f"Projection type: {projection_type}")
    print(f"Reflection method: {reflection_method}")
    print("Calculating S-coordinates and depth values...")
    
    # Process each pixel
    for py in range(height):
        if py % (height // 10) == 0:  # Progress indicator
            print(f"Progress: {py/height*100:.1f}%")
            
        for px in range(width):
            if use_full_pipeline:
                # Generate reflection normal if needed
                reflection_normal = None
                if reflection_method != 'none':
                    reflection_normal = generate_reflection_normal(
                        reflection_method, 
                        normalized_intensity[py, px], 
                        px, py
                    )
                
                # Use proper S-coordinate calculation
                S_2d, intermediate_results = compute_s_coordinate_proper(
                    px, py, width, height, 
                    normalized_intensity[py, px],
                    reflection_normal, 
                    projection_type
                )
                
                # Store analysis data
                analysis_data['spherical_coords'][py, px] = [
                    intermediate_results['theta'], 
                    intermediate_results['phi']
                ]
                analysis_data['reflection_applied'][py, px] = intermediate_results['reflection_applied']
                
            else:
                # Use legacy method
                S_2d = pixel_to_s_coordinate(px, py, width, height, s_bounds)
                intermediate_results = None
            
            s_coordinate_map[py, px] = S_2d
            
            # Calculate depth from S-coordinate
            s_depth = calculate_s_depth(S_2d, depth_method, intermediate_results)
            
            # Optionally blend with image intensity
            if use_image_intensity:
                intensity = normalized_intensity[py, px]
                # Combine S-coordinate depth with image intensity
                final_depth = (1 - intensity_weight) * s_depth + intensity_weight * intensity
            else:
                final_depth = s_depth
            
            depth_map[py, px] = final_depth
    
    # Apply Gaussian smoothing
    if smooth_sigma > 0:
        depth_map = gaussian_filter(depth_map, sigma=smooth_sigma)
    
    print("Depth map generation complete!")
    return depth_map, s_coordinate_map, image_array, analysis_data

def visualize_depth_map(depth_map, s_coordinate_map, original_image, 
                       analysis_data=None, save_path=None, show_plots=True):
    """Visualize the generated depth map and S-coordinate analysis."""
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    
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
    ax_3d = fig.add_subplot(3, 3, 3, projection='3d')
    h, w = depth_map.shape
    x_3d, y_3d = np.meshgrid(np.arange(w), np.arange(h))
    surface = ax_3d.plot_surface(x_3d[::10, ::10], y_3d[::10, ::10], 
                                depth_map[::10, ::10], cmap='plasma', alpha=0.8)
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
    
    # Nuit boundary analysis
    axes[1, 2].set_aspect('equal')
    axes[1, 2].set_xlim(-2.5, 2.5)
    axes[1, 2].set_ylim(-2.5, 2.5)
    
    # Sample S-coordinates for visualization
    sample_step = max(1, min(depth_map.shape) // 50)
    sample_s_coords = s_coordinate_map[::sample_step, ::sample_step].reshape(-1, 2)
    
    # Color points by their depth values
    sample_depths = depth_map[::sample_step, ::sample_step].flatten()
    scatter = axes[1, 2].scatter(sample_s_coords[:, 0], sample_s_coords[:, 1], 
                                c=sample_depths, cmap='plasma', s=1, alpha=0.6)
    
    # Add Nuit boundary circle
    circle = plt.Circle((0, 0), 1.0, fill=False, color='white', linewidth=2)
    axes[1, 2].add_patch(circle)
    axes[1, 2].set_title('S-Coordinates with Nuit Boundary')
    axes[1, 2].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1, 2], fraction=0.046, pad=0.04)
    
    # Additional analysis plots if analysis_data is available
    if analysis_data is not None:
        # Spherical coordinates theta
        theta_plot = axes[2, 0].imshow(analysis_data['spherical_coords'][:, :, 0], 
                                      cmap='hsv')
        axes[2, 0].set_title('Spherical Theta (Azimuthal)')
        axes[2, 0].axis('off')
        plt.colorbar(theta_plot, ax=axes[2, 0], fraction=0.046, pad=0.04)
        
        # Spherical coordinates phi
        phi_plot = axes[2, 1].imshow(analysis_data['spherical_coords'][:, :, 1], 
                                    cmap='viridis')
        axes[2, 1].set_title('Spherical Phi (Polar)')
        axes[2, 1].axis('off')
        plt.colorbar(phi_plot, ax=axes[2, 1], fraction=0.046, pad=0.04)
        
        # Reflection applied map
        reflection_plot = axes[2, 2].imshow(analysis_data['reflection_applied'], 
                                           cmap='binary')
        axes[2, 2].set_title('Householder Reflection Applied')
        axes[2, 2].axis('off')
    else:
        # Hide unused subplots
        axes[2, 0].axis('off')
        axes[2, 1].axis('off')
        axes[2, 2].axis('off')
    
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
    parser = argparse.ArgumentParser(description='Generate depth map using S-coordinates')
    parser.add_argument('image_path', help='Path to input image')
    parser.add_argument('--output_dir', default='./s_depth_output', 
                       help='Output directory for results')
    parser.add_argument('--depth_method', default='nuit_distance',
                       choices=['nuit_distance', 'stereographic_z', 'radial_distance', 
                               'golden_spiral', 'intensity_weighted', 'spherical_phi',
                               'reflection_magnitude'],
                       help='Method for calculating depth from S-coordinates')
    parser.add_argument('--s_bounds', type=float, default=2.0,
                       help='S-coordinate space bounds (symmetric, for legacy method)')
    parser.add_argument('--smooth_sigma', type=float, default=1.0,
                       help='Gaussian smoothing sigma (0 for no smoothing)')
    parser.add_argument('--intensity_weight', type=float, default=0.3,
                       help='Weight for blending with image intensity')
    parser.add_argument('--use_legacy', action='store_true',
                       help='Use legacy pixel-to-S-coordinate method instead of full pipeline')
    parser.add_argument('--projection_type', default='stereographic',
                       choices=['stereographic', 'orthographic', 'cylindrical'],
                       help='3D to 2D projection method')
    parser.add_argument('--reflection_method', default='none',
                       choices=['none', 'random', 'image_based', 'fixed'],
                       help='Method for Householder reflection')
    parser.add_argument('--no_display', action='store_true',
                       help='Skip displaying plots')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set pipeline mode (default to full pipeline unless legacy is requested)
    use_full_pipeline = not args.use_legacy
    
    # Generate depth map
    depth_map, s_coordinate_map, original_image, analysis_data = generate_s_depth_map(
        args.image_path,
        depth_method=args.depth_method,
        s_bounds=(-args.s_bounds, args.s_bounds),
        smooth_sigma=args.smooth_sigma,
        intensity_weight=args.intensity_weight,
        use_full_pipeline=use_full_pipeline,
        projection_type=args.projection_type,
        reflection_method=args.reflection_method
    )
    
    if depth_map is None:
        print("Failed to generate depth map")
        return
    
    # Generate output filenames
    base_name = os.path.splitext(os.path.basename(args.image_path))[0]
    pipeline_suffix = "full" if use_full_pipeline else "legacy"
    depth_map_path = os.path.join(args.output_dir, 
                                 f"{base_name}_depth_{args.depth_method}_{pipeline_suffix}.png")
    visualization_path = os.path.join(args.output_dir, 
                                    f"{base_name}_analysis_{args.depth_method}_{pipeline_suffix}.png")
    
    # Save depth map
    save_depth_map(depth_map, depth_map_path)
    
    # Create and save visualization
    fig = visualize_depth_map(depth_map, s_coordinate_map, original_image, 
                             analysis_data,
                             save_path=visualization_path, 
                             show_plots=not args.no_display)
    
    # Print statistics
    print(f"\nDepth Map Statistics:")
    print(f"Min depth: {np.min(depth_map):.4f}")
    print(f"Max depth: {np.max(depth_map):.4f}")
    print(f"Mean depth: {np.mean(depth_map):.4f}")
    print(f"Std depth: {np.std(depth_map):.4f}")
    
    # Analyze Nuit boundary alignment
    s_coords_flat = s_coordinate_map.reshape(-1, 2)
    distances_from_nuit = np.abs(np.linalg.norm(s_coords_flat, axis=1) - 1.0)
    on_boundary_count = np.sum(distances_from_nuit < 0.1)  # Within 0.1 units
    
    print(f"\nNuit Boundary Analysis:")
    print(f"Points near Nuit boundary (within 0.1): {on_boundary_count}")
    print(f"Percentage on boundary: {on_boundary_count/len(s_coords_flat)*100:.2f}%")
    
    # Additional analysis for full pipeline
    if analysis_data is not None:
        reflection_count = np.sum(analysis_data['reflection_applied'])
        print(f"\nPipeline Analysis:")
        print(f"Pixels with Householder reflection applied: {reflection_count}")
        print(f"Reflection percentage: {reflection_count/(analysis_data['reflection_applied'].size)*100:.2f}%")
        print(f"Projection type used: {analysis_data['projection_type']}")
        print(f"Reflection method: {analysis_data['reflection_method']}")
    
    print(f"\nResults saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
