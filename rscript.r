# Input data
# Specify either l,b or rlen. Set other to NA. rlen takes precedence.
w    <- 0.4679639686895466 # parallax in mas (corrected for any zeropoint offset; +0.029mas in the catalogue)
wsd  <- 0.014949423260986805 # parallax uncertainty in mas
glon <- 71.33496545364491 # Galactic longitude in degrees (0 to 360)
glat <-  3.0668344818073843 # Galactic latitude (-90 to +90)
rlen <-  NA # length scale in pc
# Plotting parameters in pc
# rlo,rhi are range for computing normalization of posterior
# rplotlo, rplothi are plotting range (computed automatically if set to NA)
rlo <- 0
rhi <- 1e5
rplotlo <- NA
rplothi <- NA

source("distest_single.R")
