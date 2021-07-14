0. Install Anaconda with Python 3.6

1. Create a new Python 3.6 Conda environment (I use Command Line):
conda create -n <envname> python=3.6

2. Activate environment
source activate <envname>

3. Install the packages listed in the "Notes.docx" document. It is best to install tensorflow first because it provides base packages:
conda install tensorflow-gpu=2.1

4. Open Jupyter Notebook to DEC/. Test-run 'reef_DEC_simulations.ipynb'. The FFT parameters and padding can be updated in 'set_params_sim.py'. 
NOTE: if you change FFT size, it affects the size of inputs and you will need a new CNN model with correct dimensionality. Some options are found in the folder. Edit model loading in Cell 6 just above '# PRETRAIN CAE'.

  # 5. will not run without external data access. Feel free to use your own data here.
5. To run model on real data, fix data path in 'set_params.py' to direct to DASAR X recording from Feb. 25 (or preferred day). Run 'make_data.ipynb'. Run 'reef_DEC_data.ipynb'. The same note applies about FFT size.
NOTE: You can run this for different detector runs from the 'output' Matlab structure. An example file is used here but can be replaced. The Notebook will regenerate the spectrogram inputs directly from the raw data based on the timings given in Output.

6. Explore parameters. Update the environment to run on GPU. Process more data. 
