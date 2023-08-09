# This is an Ubuntu image with system Python 3.10.
FROM python:3.10
ENV PIP_NO_CACHE_DIR=1
RUN pip install numpy scipy scikit-learn cython juliacall # hmmlearn dynamax
RUN pip install git+https://github.com/lindermanlab/ssm

# Install braingeneerspy, always from the latest commit.
ADD "https://api.github.com/repos/braingeneers/braingeneerspy/commits?per_page=1" /tmp/latest_braingeneers_commit
RUN pip install "git+https://github.com/braingeneers/braingeneerspy#egg=braingeneerspy[analysis,iot]"
RUN pip install "git+https://github.com/atspaeth/mat7.3"

# Install juliacall and force it to install all the relevant Julia
# libraries. Copy over the Manifest and Project files FIRST so this won't
# need to be re-run every time the source changes.
WORKDIR /root
# COPY NeuroHMM/Manifest.toml NeuroHMM/Manifest.toml
# COPY NeuroHMM/Project.toml NeuroHMM/Project.toml
# RUN python -c "from juliacall import Main as jl; jl.seval('using Pkg'); jl.Pkg.activate('NeuroHMM'); jl.Pkg.instantiate()"

# Now copy over the actual source files.
# COPY NeuroHMM/src NeuroHMM/src
COPY *.py .
