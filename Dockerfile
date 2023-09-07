# This is an Ubuntu image with system Python 3.10.
FROM python:3.10
ENV PIP_NO_CACHE_DIR=1
RUN pip install numpy scipy scikit-learn cython
RUN pip install git+https://github.com/lindermanlab/ssm

# Install braingeneerspy, always from the latest commit.
ADD "https://api.github.com/repos/braingeneers/braingeneerspy/commits?per_page=1" /tmp/latest_braingeneers_commit
RUN pip install "git+https://github.com/braingeneers/braingeneerspy#egg=braingeneerspy[analysis,iot]"

# Also mat73. The feature I added hasn't made it to PyPI yet.
RUN pip install "git+https://github.com/atspaeth/mat7.3"

# Copy over the source files.
WORKDIR /root
COPY *.py .
