# This is an Ubuntu image with system Python 3.10.
FROM python:3.10
ENV PIP_NO_CACHE_DIR=1
RUN pip install numpy scipy scikit-learn cython mat73
RUN pip install git+https://github.com/lindermanlab/ssm
RUN pip install braingeneers

# Copy over the source files.
WORKDIR /root
COPY *.py .
