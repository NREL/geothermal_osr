# OSR -- Geothermal Open-Source Reservoir

Repo with the tools developed for predicting energy produced at OSR. 
It includes both simulation data for OSR, as well as Jupyter notebooks for training and evaluating prediction models.

---------------

**Disclaimer**: OSR was constructed based on the data from Brady Hot Springs reservoir (Nevada, USA) but has a number of sufficiently modified characteristics and does not disclose any sensitive data.

---------------

# Requirements

`conda` is not a requirement strictly speaking, but it makes it very easy to get Python dependenies for the code included in this repo. If you don't currently have conda on your machine, it is recommended to install it by following instructions from: [https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) 

conda environment with the dependencies required to run code from this repo can be built using the `environment.yml` file and the following terminal command:

`conda env create -f environment.yml`

Note that `enironment.yml` is an environment file was tested and confirmed to be working on Eagle, NREL's supercomputer (more about machine specs: [https://www.nrel.gov/hpc/eagle-system.html](https://www.nrel.gov/hpc/eagle-system.html)). No adjustment should be required for using it on similar Linux systems, whereas Windows and OSX machines might require some tuning in this file  (e.g., changing versions of the packages). 

Once the last command successfully completes, run:

`conda activate geothermal_osr`

The last command should be run every time you close the terminal window with the activated environment and want to go back to running the code using a new terminal. When the environment is activated, the terminal prompt should start with: 

`(geothermal_osr)`

Alternatively, if you choose to avoid using conda, install manually all Python packages listed in `environment.yml` and that should be sufficient for running the notebooks. 

# Running

From the terminal, launch JupyterLab by running: 

`jupyter lab --no-browser --ip=0.0.0.0`

This works well for running JupyterLab on a remote machine and then using `ssh` to set up port forwarding (i.e., `ssh -L <local_port>:localhost:<remote_port> <hostname>`). If you run the notebooks on a local machine, it is enough to run: `jupyter lab`.

After this step, open your browser and navigate to URL that looks like this:

`http://127.0.0.1:<port>/?token=XXXYYY...ZZZ` -- this URL can be found near the end of the output of the jupyter command in the terminal. 

This should take you to the running instance of JupyterLab and you should see the directory listing on the left-hand side of the screen.

Open the `notebooks` directory where you can find the `osr-[0-3]-*.ipynb` notebooks. It is recommended that you run them in the logical order, according to the numbers in the names.

Summary of the notebooks:
* `notebooks/osr-0-minimal.ipynb` -- minimal notebook that trains a single model on the Train subset and evalutes it on the Test subset.
* `notebooks/osr-1-train_with_cross_validation.ipynb` -- notebook with cross-validation for model training. Takes a long time to run for a fairly large set of hyperparameter combinations being evaluated. Saves results of cross-validation into a file, which can be read by the following notebook.
* `notebooks/osr-2-analyze_cross_validation.ipynb` -- loads results of cross-validation from the specified file, displays summaries, and produces plots.
* `notebooks/osr-3-train_production.ipynb` -- trains production models according to the configuration selected based on cross-validation; the notebook also includes final model evaluation and sensitivity analysis.

# Paper

We will add a link to our paper here once it is published.

# Credit

Written by: Dmitry Duplyakin (dmitry.duplyakin@nrel.gov) in collaboration with the National Renewable Energy Laboratories.

Full team: Dmitry Duplyakin, Koenraad F. Beckers, Drew L. Siler, Michael J. Martin, Henry E. Johnston

