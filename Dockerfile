# This is an Ubuntu image with system Python 3.10.
FROM python:3.10
RUN pip install --no-cache-dir smart_open[s3] numpy scipy scikit-learn cython hmmlearn
RUN pip install --no-cache-dir git+https://github.com/lindermanlab/ssm
RUN pip install --no-cache-dir dynamax

# Install braingeneerspy, always from the latest commit.
ADD "https://api.github.com/repos/braingeneers/braingeneerspy/commits?per_page=1" /tmp/latest_braingeneers_commit
RUN pip install "git+https://github.com/braingeneers/braingeneerspy#egg=braingeneerspy[analysis]"

# Install juliacall and force it to install all the relevant Julia
# libraries. Copy over the Manifest and Project files FIRST so this won't
# need to be re-run every time the source changes.
RUN pip install --no-cache-dir juliacall
WORKDIR /root
COPY NeuroHMM/Manifest.toml NeuroHMM/Manifest.toml
COPY NeuroHMM/Project.toml NeuroHMM/Project.toml
RUN python -c "from juliacall import Main as jl; jl.seval('using Pkg'); jl.Pkg.activate('NeuroHMM'); jl.Pkg.instantiate()"

# Now copy over the actual source files.
COPY NeuroHMM/src NeuroHMM/src
COPY *.py .

# Define the variables used by the scripts.
ENV AWS_ENDPOINT https://s3-west.nrp-nautilus.io
