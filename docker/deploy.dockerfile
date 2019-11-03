FROM spectreteam/python_msi:v5.0.0

RUN apt-get update &&\
    apt-get install -y procps &&\
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /requirements.txt

RUN pip install --no-cache-dir -r /requirements.txt &&\
    rm /requirements.txt

RUN mkdir /app

WORKDIR /app

COPY components components

COPY pipeline pipeline

COPY test test

RUN python -m unittest discover

RUN rm -rf test

# Luigi scheduler port
EXPOSE 8082

# Data mount point
VOLUME /data

RUN mkdir /luigi

COPY entrypoint.sh entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
