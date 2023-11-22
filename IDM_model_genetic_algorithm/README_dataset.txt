Napoli car-following database.

It includes:

- 5 txt files, whose names include an ID and the date of the car-following experiment. Each file contains one experiment (some minutes of consecutives car-following data for a platoon of 4 vehicles).
- the file “Experimental campaign data.xls” which contains information about the vehicles.
- two papers in which the car-following experiments (Punzo and Simonelli, 2005) and the processing of data (Punzo et al. 2005) are described.

*******************************************************

Information about the databases in the .txt files.

Measurement units 
speeds = metres/second
distances = metres

Measurements are taken at 10 Hz. (i.e. records are every 0.1 second)

Each database consists of 7 columns (occasionally 8, where the first one is the time).
-	The first 4 columns are the speeds [m/s] of the vehicles in the platoon.
-	The last 3 columns are the distances [m] between the GPS antennas of each couple of vehicles, respectively, d(1-2), d(2-3) and d(3-4). The distance of each antenna from the front bumper, together with the main vehicle characteristics are reported in the file “experimental campaign data”. From such file and the last 3 columns it is possible to derive the distance gaps or the space-headways.

