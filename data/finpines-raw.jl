# You have to make sure RCall is correctly installed to run this script.
# This script only extracts raw data from R without any pre-processing.
# This is aimed to make the rest of data preparation R indepedent.

using DrWatson
@quickactivate

using RCall

# Load data from R's spatstat package
R"""
library(spatstat)
data(finpines)
x <- finpines$x
y <- finpines$y
"""

# Save as BSON
@rget x y
wsave(projectdir("finpines-raw.bson"), @dict(x, y))
