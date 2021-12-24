# OSR -- Geothermal Open-Source Reservoir

Repo with the tools developed for predicting energy produced at OSR. 
It includes both simulation data for OSR, as well as Jupyter notebooks for training and evaluating prediction models.

---------------

**Disclaimer**: OSR was constructed based on the data from Brady Hot Springs reservoir (Nevada, USA) but has a number of sufficiently modified characteristics and does not disclose any sensitive data.

---------------

# Requirements:

`conda` is not a requirement strictly speaking, but it makes it very easy to get Python dependenies for the code included in this repo. If you don't currently have conda on your machine, it is recommended to install it by following instructions from: [https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) 

conda environment with the dependencies required to run code from this repo can be built using the `environment.yml` file and the following terminal command:

`conda env create -f environment.yml`

Once this step successfully completes, run:

`conda activate geothermal_osr`

The last command should be run every time you close the terminal window with the activated environment and want to go back to running the code using a new terminal. When the environment is activated, the terminal prompt should start with: 

`(geothermal_osr)`

Alternatively, if you choose to avoid using conda, install manually all Python packages listed in `environment.yml` and that should be sufficient for running the notebooks. 

# Running:

From the terminal, launch JupyterLab by running: 

`jupyter lab --no-browser --ip=0.0.0.0`

This works well for running JupyterLab on a remote machine and then using ssh to set up port forwarding (i.e., `ssh -L <local port>:localhost:<remote port> <hostname>`). If you run the notebooks on a local machine, it is enough to run: `jupyter lab`.

-------
Written by: Dmitry Duplyakin (dmitry.duplyakin@nrel.gov) in collaboration with the National Renewable Energy Laboratories.

Full team: Dmitry Duplyakin, Koenraad F. Beckers, Drew L. Siler, Michael J. Martin, Henry E. Johnston

