#!/usr/bin/env python3
"""
Create Study Area Figure for Ganges Delta Manuscript
Generates fig1_study_area.png with:
- (a) Geomorphological zones with district outlines (fixed for Mature Delta)
- (b) Sampling locations
- (c) Elevation (DEM)
- (d) SOC stock by zone
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import geopandas as gpd
import pandas as pd
import rasterio
from pathlib import Path
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10

def load_data():
    """Load all required data"""
    base_path = Path(__file__).resolve().parent.parent / 'data' / 'gis'

    # Load the three intact zone shapefiles (upazila / ADM3 polygons)
    tidal = gpd.read_file(base_path / 'TidalActiveD1.shp')
    active = gpd.read_file(base_path / 'ActiveDeltaD4.shp')
    moribund = gpd.read_file(base_path / 'MoribundDeltaD3.shp')

    # Mature Delta: MatureDeltaD2.shp is broken (only 1 upazila). Reconstruct it from
    # the master delta upazila layer (GangesDeltaD.shp, 159 ADM3 polygons) as the
    # upazilas NOT assigned to the other three zones (34 polygons; verified bounds
    # [88.7, 22.52, 89.87, 23.57] match the original MatureDeltaD2 extent).
    master = gpd.read_file(base_path / 'GangesDeltaD.shp')
    assigned = set(tidal['ADM3_PCODE']) | set(active['ADM3_PCODE']) | set(moribund['ADM3_PCODE'])
    mature = master[~master['ADM3_PCODE'].isin(assigned)].copy()

    zones = {
        'Tidal Active': tidal,
        'Active Delta': active,
        'Mature Delta': mature,
        'Moribund Delta': moribund
    }

    # Load sampling locations (use main dataset with coordinates)
    locations_file = base_path.parent / 'GangesSOC.csv'
    locations = pd.read_csv(locations_file)

    # Load DEM
    dem_file = base_path / 'DEM.tif'
    with rasterio.open(dem_file) as src:
        dem_data = src.read(1)
        dem_bounds = src.bounds
        dem_transform = src.transform

    # Load SOC data (LocationMerged has Delta column for zones)
    soc_file = base_path.parent / 'LocationMerged.csv'
    soc_data = pd.read_csv(soc_file)
    # Rename columns to match expected names
    soc_data.rename(columns={'Delta': 'Zone', 'SOC Stock ': 'SOC_Stock'}, inplace=True)
    # Also try without trailing space
    if 'SOC_Stock' not in soc_data.columns:
        for col in soc_data.columns:
            if 'SOC' in col and 'Stock' in col:
                soc_data.rename(columns={col: 'SOC_Stock'}, inplace=True)
                break

    # Load Bangladesh national boundary for the inset (geoBoundaries gbOpen ADM0,
    # CC BY 4.0; full-country extent ~88-92.7E, 20.6-26.6N). GangesDelta.shp covers
    # only the southern delta districts, so it cannot serve as the national locator.
    boundary_file = base_path / 'bgd_adm0.geojson'
    bangladesh = gpd.read_file(boundary_file)

    return zones, locations, dem_data, dem_bounds, soc_data, bangladesh

def plot_panel_a(ax, zones, bangladesh):
    """Panel (a): Geomorphological zones with district outlines"""

    # Define colors for zones
    colors = {
        'Tidal Active': '#2ecc71',      # Green
        'Active Delta': '#3498db',      # Blue
        'Mature Delta': '#e74c3c',      # Red
        'Moribund Delta': '#f39c12'     # Orange
    }

    # Plot each zone with district boundaries
    for zone_name, zone_gdf in zones.items():
        zone_gdf.plot(ax=ax, color=colors[zone_name], alpha=0.7, edgecolor='black', linewidth=0.5)

    # Set extent
    ax.set_xlim(88, 91.5)
    ax.set_ylim(21.5, 24.5)
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    ax.set_title('(a) Geomorphological zones', fontsize=11, fontweight='bold', loc='left')

    # Create legend with proper placement (BOTTOM LEFT)
    legend_elements = [
        mpatches.Patch(facecolor='#2ecc71', edgecolor='black', label='Tidal Active'),
        mpatches.Patch(facecolor='#3498db', edgecolor='black', label='Active Delta'),
        mpatches.Patch(facecolor='#e74c3c', edgecolor='black', label='Mature Delta'),
        mpatches.Patch(facecolor='#f39c12', edgecolor='black', label='Moribund Delta')
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=9, framealpha=0.95)

    # Add Bangladesh inset map (TOP RIGHT CORNER)
    inset_ax = ax.inset_axes([0.65, 0.65, 0.33, 0.33])
    bangladesh.plot(ax=inset_ax, color='lightgray', edgecolor='black', linewidth=1)
    inset_box = Rectangle((88.0, 21.5), 2.0, 3.0, fill=False, edgecolor='red', linewidth=1.5)
    inset_ax.add_patch(inset_box)
    inset_ax.set_xlim(87, 93.2)
    inset_ax.set_ylim(20, 27)
    inset_ax.set_title('Bangladesh', fontsize=8)
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])

    ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='0.35', alpha=0.65, zorder=5)
    ax.set_axisbelow(False)

def plot_panel_b(ax, locations, zones):
    """Panel (b): Sampling locations"""

    # Plot zones as background
    colors = {
        'Tidal Active': '#2ecc71',
        'Active Delta': '#3498db',
        'Mature Delta': '#e74c3c',
        'Moribund Delta': '#f39c12'
    }

    for zone_name, zone_gdf in zones.items():
        zone_gdf.plot(ax=ax, color=colors[zone_name], alpha=0.3, edgecolor='none')

    # Define markers and colors for locations based on zone
    zone_colors = {
        'Tidal Active': '#2ecc71',
        'Active Delta': '#3498db',
        'Mature Delta': '#e74c3c',
        'Moribund Delta': '#f39c12'
    }

    zone_markers = {
        'Tidal Active': 'o',
        'Active Delta': '^',
        'Mature Delta': 's',
        'Moribund Delta': 'D'
    }

    # Get unique locations with coordinates
    loc_df = locations.drop_duplicates(subset='Location')[['Location', 'Latitude', 'Longitude']].copy()

    # Assign each location to its zone from the authoritative per-zone sample files
    # (5 locations per zone), not by a latitude heuristic.
    base_path = Path(__file__).resolve().parent.parent / 'data' / 'gis'
    zone_files = {'Tidal Active': 'Tidal.csv', 'Active Delta': 'Active.csv',
                  'Mature Delta': 'Mature.csv', 'Moribund Delta': 'Moribund.csv'}
    loc_to_zone = {}
    for zname, zf in zone_files.items():
        zc = pd.read_csv(base_path.parent / zf)
        zc.columns = [c.strip() for c in zc.columns]
        for L in zc['Location'].unique():
            loc_to_zone[L] = zname

    loc_df['Zone'] = loc_df['Location'].map(loc_to_zone)

    # Plot sampling locations
    for zone in zone_colors:
        zone_locs = loc_df[loc_df['Zone'] == zone]
        if len(zone_locs) > 0:
            ax.scatter(zone_locs['Longitude'], zone_locs['Latitude'],
                      s=80, color=zone_colors.get(zone, 'gray'),
                      marker=zone_markers.get(zone, 'o'),
                      edgecolor='black', linewidth=1, label=zone, zorder=5)

    ax.set_xlim(88, 91.5)
    ax.set_ylim(21.5, 24.5)
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    ax.set_title('(b) Sampling locations', fontsize=11, fontweight='bold', loc='left')
    ax.legend(loc='lower left', fontsize=8, framealpha=0.95)
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='0.35', alpha=0.65, zorder=5)
    ax.set_axisbelow(False)

def plot_panel_c(ax, dem_data, dem_bounds):
    """Panel (c): Elevation (DEM)"""

    # Mask invalid values
    dem_masked = np.ma.masked_where(dem_data <= 0, dem_data)

    # Plot DEM
    im = ax.imshow(dem_masked, cmap='terrain', extent=[dem_bounds.left, dem_bounds.right,
                                                         dem_bounds.bottom, dem_bounds.top],
                   origin='upper', aspect='auto', vmin=0, vmax=50)

    ax.set_xlim(88, 91.5)
    ax.set_ylim(21.5, 24.5)
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    ax.set_title('(c) Elevation (m)', fontsize=11, fontweight='bold', loc='left')

    cbar = plt.colorbar(im, ax=ax, label='Elevation (m)')
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='0.35', alpha=0.65, zorder=5)
    ax.set_axisbelow(False)

def plot_panel_d(ax, soc_data):
    """Panel (d): SOC stock by zone"""

    # Prepare data for boxplot
    if 'Zone' in soc_data.columns and 'SOC_Stock' in soc_data.columns:
        available_zones = soc_data['Zone'].unique()
        zone_order = ['Active', 'Mature', 'Moribund', 'Tidal']

        # Filter zones that exist in data
        zones_to_plot = [z for z in zone_order if z in available_zones]

        label_map = {
            'Active': 'Active\nDelta',
            'Mature': 'Mature\nDelta',
            'Moribund': 'Moribund\nDelta',
            'Tidal': 'Tidal\nActive'
        }

        # Create boxplot
        bp = ax.boxplot([soc_data[soc_data['Zone'] == z]['SOC_Stock'].values for z in zones_to_plot],
                        labels=[label_map.get(z, z) for z in zones_to_plot],
                        patch_artist=True)

        # Color boxes
        colors_list = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']
        for patch, color in zip(bp['boxes'], colors_list[:len(zones_to_plot)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Add sample size annotations
        for i, (patch, zone) in enumerate(zip(bp['boxes'], zones_to_plot)):
            n_samples = len(soc_data[soc_data['Zone'] == zone])
            ax.text(i+1, ax.get_ylim()[1]*0.95, f'n={n_samples}',
                   ha='center', va='top', fontsize=8)

    ax.set_ylabel('SOC stock (t C ha⁻¹)')
    ax.set_title('(d) SOC stock by zone', fontsize=11, fontweight='bold', loc='left')
    ax.grid(True, alpha=0.3, axis='y')

def create_figure():
    """Create the complete study area figure"""

    print("Loading data...")
    zones, locations, dem_data, dem_bounds, soc_data, bangladesh = load_data()

    print("Creating figure...")
    fig = plt.figure(figsize=(14, 12))

    # Create grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Panel A
    ax_a = fig.add_subplot(gs[0, 0])
    plot_panel_a(ax_a, zones, bangladesh)

    # Panel B
    ax_b = fig.add_subplot(gs[0, 1])
    plot_panel_b(ax_b, locations, zones)

    # Panel C
    ax_c = fig.add_subplot(gs[1, 0])
    plot_panel_c(ax_c, dem_data, dem_bounds)

    # Panel D
    ax_d = fig.add_subplot(gs[1, 1])
    plot_panel_d(ax_d, soc_data)

    # Save figure
    output_path = str(Path(__file__).resolve().parent / 'fig1_study_area.png')
    print(f"Saving figure to: {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Figure saved successfully!")

    plt.close()

if __name__ == '__main__':
    create_figure()
