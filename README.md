# Wormhole
## Exoplanet detection with flaxoil

![TESS First Light image](assets/TESS_firstlight1280.jpg)

This repository contains in-progress work  on building a Flaxoil model which can classify exoplanets in the TESS Lightcurve dataset.

## What is TESS?

The [Transiting Exoplanet Survey Satellite](https://en.wikipedia.org/wiki/Transiting_Exoplanet_Survey_Satellite) was launched on 18 April 2018 on an all-sky survey mission to detect transiting exoplanets.

Data from the mission is hosted at [MAST](https://archive.stsci.edu/missions-and-data/tess).

Photometric data products include time-series of full-frame CCD sensor images, time-series of selected pixels around target stars taken at faster cadences, and flux time-series generated from aperture photometry on these target pixels. MAST also hosts Planet Search data products such as statistics on detected threshold crossing events.

This project uses data products from the Primary Mission (Year 1 and 2), specifically: Light Curve (LC) data consisting of the flux time-series, and Threshold Crossing Event (TCE) data to generate classification labels for training.

# What is Flaxoil?

[Flaxoil](https://github.com/MNahad/flaxoil) is a port of the [ncps](https://github.com/mlech26l/keras-ncp) Python package, which itself is an implementation of Liquid Neural Networks [1] [2].

A Liquid Neural Network is a novel ML algorithm that is bio-inspired by the brain of the _C. Elegans_ nematode. The network contains sparsely-connected RNN-based ODE solver cells, mimicing the roundworm's neural synapses.

Flaxoil ports the original ncps package to the Google Flax ML framework.

# Acknowledgements

- Funding for the TESS mission is provided by NASA's Science Mission directorate. This research includes data collected by the TESS mission, which are publicly available from the Mikulski Archive for Space Telescopes (MAST).
- Images courtesy of NASA/MIT/TESS.

# References
1. M. Lechner, R. Hasani, A. Amini, T. A. Henzinger, D. Rus, and R. Grosu, "Neural circuit policies enabling auditable autonomy," Nature Machine Intelligence, vol. 2, no. 10, pp. 642-652, Oct 2020.
1. R. Hasani, M. Lechner, A. Amini, D. Rus, and R. Grosu, "Liquid Time-constant Networks", AAAI, vol. 35, no. 9, pp. 7657-7666, May 2021.