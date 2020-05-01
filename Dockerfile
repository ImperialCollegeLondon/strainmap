FROM python:3.8-slim AS strainmap
RUN apt-get update \
    && apt-get -y install --no-install-recommends gfortran python3-tk \
    && rm -fr /var/lib/apt/lists/*
WORKDIR /usr/src/app
COPY setup.py .
COPY strainmap ./strainmap
RUN python -mpip install -e .

FROM strainmap
RUN apt-get update \
    && apt-get -y install --no-install-recommends xvfb xauth \
    && rm -fr /var/lib/apt/lists/*
COPY . .
ENTRYPOINT xvfb-run -a python setup.py -q test
