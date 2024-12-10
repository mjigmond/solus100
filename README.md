### Soil Estimator POC

The purpose of the code is to show how to build a simple soil estimator using soil data properties in raster format. There are two versions of this POC:
 - `main.py`: will estimate percent total sand at 60cm depth using other soil types at the same depth.
   - files used for this POC: `[claytotal_60_cm_p.tif, sandtotal_60_cm_p.tif, gypsum_60_cm_p.tif, silttotal_60_cm_p.tif]`
 - `main2.py`: will estimate percent total sand at any depth up to 200cm using percent total sand at specific depth intervals.
   - files used for this POC: `[sandtotal_0_cm_p.tif, sandtotal_5_cm_p.tif, sandtotal_15_cm_p.tif, sandtotal_30_cm_p.tif, sandtotal_60_cm_p.tif, sandtotal_100_cm_p.tif, sandtotal_150_cm_p.tif]`
