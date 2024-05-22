Autonomous Robotics
Ex0 - GNSS Raw Mesurments

This assignment focuses on the basic principles of GNSS, in particular, you are asked to implement a naive positioning algorithm based on RMS (Root Mean Square) of selected (i.e., weighted) pseudo-ranges.

ToDo:
Get to know the basic concepts of GNSS in particular, the notion of pseudo-ranges, see Presentation, GNSS Raw Measurements, and android GnssLogger App.
Download the following dataset, go over the data, and design a parsing tool that converts the log file to a csv (txt) file with the following format:
GPS time, SatPRN (ID), Sat.X, Sat.Y, Sat.Z, Pseudo-Range, CN0, Doppler (nice to have), See this link for a complete explanation + python code for solving most of this assignment. Note: the sat position should be in ECEF coordinates.
Given a csv file (as in the above format) Implement a Positioning Algorithm which for a Given GPS time, computes the appropriate positioning (in X,Y,Z coordinates). Your are expected to implement an iterative numerical minimal RMS algorithm on a weighted set of SatPRNs.
Implement a converting method from X,Y,Z to Lat, Lon, Alt (see wiki page)
Integrate the above tasks into a complete solution that receives a Raw Measurements GNSS (log) file and computes two output files: (i) KML file (see time and animation) with the computed path. (ii) CSV file as in (2) with the following additional columns: Pos.X, Pos.Y, Pos,Z, Lat, Lon, Alt
Perform a testing on the given dataset and add your own data files. Conclude your work as a GitHub repo - including a detailed readme with “how to run”

How to Run:
Download files, this can be ran via the main function.

Resources:
https://www.johnsonmitchelld.com/2021/03/14/least-squares-gps.html
