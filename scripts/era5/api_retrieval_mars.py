# %%
import cdsapi
import sys

# %%
c = cdsapi.Client()
path = "/work/bm1183/m301049/icon_hcap_data/publication/vertical_vel/era5_vel_2.grb"
request = {
    "class": "ea",
    "date": "2024-09-01",
    "expver": "1",
    "levelist": "95",
    "levtype": "ml",
    "param": "135",
    "stream": "oper",
    "time": "00:00:00",
    "type": "an",
    "grid": "0.25/0.25",
}

c.retrieve("reanalysis-era5-complete", request, path)



# %%
