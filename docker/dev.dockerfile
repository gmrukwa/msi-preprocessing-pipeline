FROM spectreteam/python_msi:v5.0.0

COPY requirements/basic-strict.txt /requirements.txt

RUN pip install --no-cache-dir -r /requirements.txt &&\
    rm /requirements.txt

RUN mkdir /app

WORKDIR /app
