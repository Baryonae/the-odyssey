from datetime import timedelta

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from opendrift.readers import reader_netCDF_CF_generic
from opendrift.models.openoil import OpenOil


# ============================================================
# 1. CREATE OPENOIL SIMULATION
# ============================================================

o = OpenOil(loglevel=20)


# ============================================================
# 2. LOAD ENVIRONMENTAL DATA
# ============================================================

reader_norkyst = reader_netCDF_CF_generic.Reader(
    "data/norkyst800_subset_16Nov2015.nc"
)

reader_arome = reader_netCDF_CF_generic.Reader(
    "data/arome_subset_16Nov2015.nc"
)

o.add_reader([reader_norkyst, reader_arome])


# ============================================================
# 3. RELEASE OIL
# ============================================================

o.seed_elements(
    lon=5.05,
    lat=59.95,
    radius=1000,
    number=1000,
    time=reader_norkyst.start_time,
)


# ============================================================
# 4. RUN SIMULATION
# ============================================================

o.run(
    duration=timedelta(hours=8),
    time_step=900,
    time_step_output=3600,
)


# ============================================================
# 5. GET FINAL PARTICLE POSITIONS
# ============================================================

lon = np.asarray(o.elements.lon)
lat = np.asarray(o.elements.lat)

print(f"Total particles: {len(lon)}")


# ============================================================
# 6. REMOVE INVALID POSITIONS
# ============================================================

valid = np.isfinite(lon) & np.isfinite(lat)

lon = lon[valid]
lat = lat[valid]

print(f"Valid particles: {len(lon)}")


# ============================================================
# 7. CREATE GEOGRAPHIC GRID
# ============================================================

# Grid resolution in degrees
resolution = 0.001

# Extra space around the particle cloud
padding = 0.01

min_lon = lon.min() - padding
max_lon = lon.max() + padding

min_lat = lat.min() - padding
max_lat = lat.max() + padding


lon_edges = np.arange(
    min_lon,
    max_lon + resolution,
    resolution
)

lat_edges = np.arange(
    min_lat,
    max_lat + resolution,
    resolution
)


# ============================================================
# 8. PUT PARTICLES INTO GRID
# ============================================================

particle_count, _, _ = np.histogram2d(
    lat,
    lon,
    bins=[lat_edges, lon_edges]
)


# ============================================================
# 9. SMOOTH PARTICLE DISTRIBUTION
# ============================================================

smoothed = gaussian_filter(
    particle_count,
    sigma=2
)


# ============================================================
# 10. CONVERT TO BINARY SPILL MASK
# ============================================================

threshold = 0.20 * smoothed.max()

predicted_mask = smoothed >= threshold

print(f"Predicted spill pixels: {predicted_mask.sum()}")


# ============================================================
# 11. DISPLAY PREDICTED MASK
# ============================================================

plt.figure(figsize=(10, 7))

plt.imshow(
    predicted_mask,
    origin="lower",
    extent=[
        lon_edges[0],
        lon_edges[-1],
        lat_edges[0],
        lat_edges[-1],
    ],
    aspect="auto",
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("OpenOil Predicted Spill Mask")

plt.show()


# ============================================================
# 12. CREATE TEMPORARY "ACTUAL" MASK
# ============================================================
#
# THIS IS ONLY FOR TESTING.
#
# We shift the predicted mask slightly so that the two masks
# partially overlap.
#
# Later, this will be replaced by the real satellite mask.
# ============================================================

actual_mask = np.roll(
    predicted_mask,
    shift=5,
    axis=1,
)


# ============================================================
# 13. CALCULATE INTERSECTION
# ============================================================

intersection = np.logical_and(
    predicted_mask,
    actual_mask,
)


# ============================================================
# 14. CALCULATE UNION
# ============================================================

union = np.logical_or(
    predicted_mask,
    actual_mask,
)


# ============================================================
# 15. CALCULATE IoU
# ============================================================

if union.sum() == 0:
    iou = 0.0
else:
    iou = intersection.sum() / union.sum()


print()
print("==============================")
print(f"Intersection pixels: {intersection.sum()}")
print(f"Union pixels:        {union.sum()}")
print(f"IoU:                 {iou:.4f}")
print("==============================")


# ============================================================
# 16. DISPLAY BOTH MASKS
# ============================================================

plt.figure(figsize=(10, 7))

plt.imshow(
    predicted_mask,
    origin="lower",
    extent=[
        lon_edges[0],
        lon_edges[-1],
        lat_edges[0],
        lat_edges[-1],
    ],
    aspect="auto",
    alpha=0.5,
)

plt.imshow(
    actual_mask,
    origin="lower",
    extent=[
        lon_edges[0],
        lon_edges[-1],
        lat_edges[0],
        lat_edges[-1],
    ],
    aspect="auto",
    alpha=0.5,
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(f"Predicted vs Actual Spill — IoU = {iou:.4f}")

plt.show()
