# Scripts used for the data analysis presented in the paper:   Enhancing Vital Sign Monitoring with Laterally Placed Radar Using Persistence Diagram-Based Multiple Range Bin Selection
readme authored by Benedek Szmola

## Usage
* Execute "main.py" to do the data analysis that was done for this article
* With "computeStats.py" the results presented can be computed
* "createFigs.py" will reproduce the figures in the article (some of the figures were done entirely in PowerPoint, or they were assembled from subplots in PowerPoint)

## Files
* main.py - the main function to do the analysis
* computeStats.py - computes the results presented in the article using the save files created by main.py
* createFigs.py - produces the figures of the article
* readRecordingData.py - this function is called to extract raw data from the saved recordings
* helperFunctions.py - contains various utility functions called by other functions
* breathing_classification_rateCompute.py - has the functions, algorithms for breathing range bin selection and rate computation
* heartbeat_classification_rateCompute.py - has the functions, algorithms for heartbeat range bin selection and rate computation
* persHomAlgs.py - stores the persistence homology algorithms
* radarSettings.csv - stores radar configuration parameters required for the analysis
* figures/ - createFigs.py will store the figures in this subdirectory
* processedData/ - main.py will save the analysis results in this subdirectory

## Note on data source
For the publication, the same raw data was used as in the following study: 
Hornig, L.; Szmola, B.; Pätzold, W.; Vox, J.P.; Wolf, K.I. Evaluation of Lateral Radar Positioning for Vital Sign Monitoring: An Empirical Study. Sensors 2024, 24, 3548. https://doi.org/10.3390/s24113548

For data protection reasons, the raw data cannot be published.