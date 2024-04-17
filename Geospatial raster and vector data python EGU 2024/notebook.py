# %% URL https://esciencecenter-digital-skills.github.io/2024-04-15-geospatial-python-EGU/#/

# Access OpenStreetMap data
import osmnx as ox
# Access satellite data
import pystac_client
import odc.stac

# Vector data handling
import geopandas as gpd

# Raster data handling
import rioxarray

# Raster analyses
import xrspatial 

# Visualization
import matplotlib.pyplot as plt
# %%
# Get polygon of Rhodes
rhodes = ox.geocode_to_gdf("Rhodes")
rhodes.plot()

# %%
# Setup search client for STAC API
api_url = 'https://earth-search.aws.element84.com/v1'
client = pystac_client.Client.open(api_url)
# See collections with: list(client.get_collections())
collection = 'sentinel-2-c1-l2a'  # Sentinel-2, Level 2A
# Get the search geometry from the GeoDataFrame
polygon = rhodes.loc[0, 'geometry']
# %%
# Setup the search
search = client.search(
    collections=[collection],
    intersects=polygon,
    datetime='2023-07-01/2023-08-31', # date range 
    query=['eo:cloud_cover<10'] # cloud cover less than 10%
)
search.matched()
items = search.item_collection()
# %% SAVe objects

items.save_object('rhodes_sentinel-2.json')
# %% Open satellite images
# Load the search results as a xarray Dataset
ds = odc.stac.load(
    items,
    groupby="solar_day", # group the images within the same day
    bands=["red", "green", "blue", "nir", "scl"],
    resolution=40, # loading resolution
    chunks={'x': 2048, 'y':  2048}, # lazy loading with Dask
    bbox=polygon.bounds,
    dtype="uint16"
)

#  %%
ds_before = ds.sel(time="2023-07-13", method="nearest")
ds_after = ds.sel(time="2023-08-27", method="nearest")
#  %%

# %%
def rgb_img(ds):
    """ 
    Generate RGB raster.
    
    Sentiel-2 L2A images are provided in Digital Numbers (DNs).
    Convert to reflectance by dividing by 10,000. Set reflectance
    equal to one for values >= 10,000.
    """
    ds_rgb = ds[["red", "green", "blue"]].to_array()
    ds_rgb = ds_rgb.clip(max=10000)
    return ds_rgb / 10_000

rgb_before = rgb_img(ds_before)

rgb_after = rgb_img(ds_after)
#  Figura
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Use "robust=True" to improve contrast
rgb_before.plot.imshow(ax=axs[0], robust=True) 
rgb_after.plot.imshow(ax=axs[1], robust=True)

#  raster    calculations
#  %%
def mask_water_and_clouds(ds):
    """
    Mask water and clouds using the Sentinel-2
    scene classification map.
    """
    mask = ds["scl"].isin([3, 6, 8, 9, 10])
    return ds.where(~mask)
#  %%
ds_masked = mask_water_and_clouds(ds)

# %%
ndvi = (
    (ds_masked["nir"] - ds_masked["red"]) / 
    (ds_masked["nir"] + ds_masked["red"])
)
ndvi_before = ndvi.sel(time="2023-07-13", method="nearest")
ndvi_after = ndvi.sel(time="2023-08-27", method="nearest")
ndvi_diff = ndvi_after - ndvi_before
ndvi_diff.plot.imshow(robust=True)
# %%
#  arbitrary threshold
burned_mask = ndvi_diff < -0.4
# set 1. in red channel.
#  0 corresponde al rojo en el array. ~mask invierte la mascara. A los otros valores les asigna el 1.
rgb_after[0, :, :] = rgb_after[0, :, :].where(~burned_mask, other=1) 

# set 0. in green and blue channels
# A los otros calnales les pone un 0 
rgb_after[1:3, :, :] = rgb_after[1:3, :, :].where(~burned_mask, other=0)
# %%
rgb_after.plot.imshow(robust=True)  
# %%
# SPATIAL ANALYSIS How far are roads from burned area
# Download road network from OSM
highways = ox.features_from_place(
    "Rhodes",
    tags={
        "highway":[
            "primary", 
            "secondary", 
            "tertiary",
        ]
    }
)
highways.head()
# %%
# compute (horizontal) distance from burned area 
distance = xrspatial.proximity(burned_mask) # libreria par analisis especial de datos de rasted que esta muy eficientemente implementada, calcula desde los puntos ceros, la distancia mas corta a un pixel 1
distance.plot.imshow()

#  buffer. convierte las carreteeras a las mismas sitema de referencia que el buffer data
highways_buffer = highways \
    .to_crs(burned_mask.rio.crs) \
    .buffer(50)  # 50m buffer around the roads
highways_buffer.head()
distance_clip = distance.rio.clip(highways_buffer) #quita los pixeles que no entran en nuestras geometries.
#  %%
# Visualize roads, focusing on distances <500m from the fires
distance_clip.plot.imshow(vmax=500)  