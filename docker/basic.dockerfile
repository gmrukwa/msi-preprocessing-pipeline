FROM spectreteam/python_msi:v5.0.0

COPY requirements/basic-strict.txt /requirements.txt

RUN pip install --no-cache-dir -r /requirements.txt &&\
    rm /requirements.txt

RUN mkdir /app

WORKDIR /app

COPY bin bin

COPY components components

COPY test test

RUN python -m unittest discover

RUN rm -rf test
