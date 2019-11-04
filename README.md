# MSI Preprocessing Pipeline

Default preprocessing pipeline for MSI data in raw ASCII format,
as used by Data Mining Group in Silesian University of Technology.

## Process

The packaged pipeline consists of the following steps:

1) Find common m/z range
2) Find resampled m/z axis that will be common for all datasets
3) Resample all datasets to common m/z axis
4) Remove baseline using adaptive window method ([as proposed by Katarzyna Frątczak](http://scholar.google.com/scholar_lookup?title=Adaptive%20baseline%20correction%20algorithm%20for%20MALDI%20spectra.&author=K.%20Bednarczyk&author=M.%20Gawin&author=M.%20Pietrowska&author=P.%20Wid%C5%82ak&author=J.%20Polanska&publication_year=2017)).
5) Detect outliers in the data with respect to TIC value
6) Align spectra to average spectrum with PAFFT method
7) Normalize spectra to common TIC
8) Build Gaussian Mixture Model of the average spectrum
9) Remove outlier components of the GMM model
10) Compute convolutions of the spectra and the GMM model
11) Merge multiple GMM components resembling single peak

## Installation

The preferred installation is via
[Docker](https://www.docker.com/products/docker-desktop).

Having Docker installed, you can just pull the image:

```bash
docker pull gmrukwa/msi-preprocessing
```

## Running

### Data Format

You need to prepare your data for processing:

1) Create an empty directory `/mydata`
2) Create a directory `/mydata/raw` - this is where pipeline expects your
original data
3) Each dataset should be contained in subdirectory:

```plaintext
/mydata
    |- raw
        |- my-dataset1
        |- my-dataset2
        |- ...
```

4) Each subdirectory should contain ASCII files in the organization as
provided by Bruker, e.g.:

```plaintext
/mydata
    |- raw
        |- my-dataset1
            |- my-dataset1_0_R00X309Y111_1.txt
            |- my-dataset1_0_R00X309Y112_1.txt
            |- my-dataset1_0_R00X309Y113_1.txt
            |- my-dataset1_0_R00X309Y114_1.txt
            |- my-dataset1_0_R00X309Y115_1.txt
            |- ...
```

**Note:** File names are important, since `R`, `X` and `Y` is parsed as
metadata! If you put there broken values, spatial dependencies between
spectra will be lost.

5) Each ASCII file should be in the format as provided by Bruker, e.g.:

```plaintext
700,043096125457 2
700,051503297599 2
700,059910520559 1
700,068317794335 0
...
<another-mz-value> <another-ions-count>
...
3496,66186447226 1
3496,68071341296 3
3496,69956240485 2
```

Both `.` and `,` are supported as decimal separator.

Example of the expected structure can be found in
[`sample-data`](./sample-data).

### Launch

You can launch preprocessing via:

```bash
docker run -v /mydata:/data -p 8082:8082 gmrukwa/msi-preprocessing '["my-dataset1","my-dataset2"]'
```

Results will appear in the `/mydata` directory as soon as they are
available. You can track the progress live at
[`localhost:8082`](https://localhost:8082).

If you need output data also in the format of `.csv` files (not a binary
numpy-related `.npy`), you can simply add a switch `--export-csv`:

```bash
docker run -v /mydata:/data -p 8082:8082 gmrukwa/msi-preprocessing '["my-dataset1","my-dataset2"]' --export-csv
```

**Note:** There is no space between dataset names.

**Note:** The `--export-csv` switch must appear right after the datasets
(due to the way Docker handles arguments).

If you want to review time needed for each task to process, you can prevent
scheduler from being stopped with `--keep-alive` switch:

```bash
docker run -v /mydata:/data -p 8082:8082 gmrukwa/msi-preprocessing '["my-dataset1","my-dataset2"]' --keep-alive
```

**Note:** `--keep-alive` switch must always come last.

#### Launch Sample

1) Download `sample-data` directory
2) Run `docker run -v sample-data:/data -p 8082:8082 gmrukwa/msi-preprocessing '["my-dataset1","my-dataset2"]'`
3) Track progress at [`localhost:8082`](https://localhost:8082)

Building GMM model takes longer time (at least 1 hour), so be patient.

## Advanced

### E-Mail Notifications

You can simply add e-mail notifications to your configuration.
They will provide you with failure messages and notification, when
the pipeline completes successfully. Two methods are supported: via
SendGrid and via SMTP server.

#### SendGrid

1) Create API key on SendGrid account (try [here](https://app.sendgrid.com/settings/api_keys))
2) Download [`template.env`](./template.env) as `.env`
3) In the `.env` file, set following values (rest of the content preserve
intact):

```env
LUIGI_EMAIL_METHOD=sendgrid
LUIGI_EMAIL_RECIPIENT=<your-email-here>
LUIGI_SENDGRID_APIKEY=<your-api-key-here>
```

4) Launching processing with Docker, use additional switch
`--env-file .env`:

```bash
docker run -v /mydata:/data --env-file .env gmrukwa/msi-preprocessing '["my-dataset1","my-dataset2"]'
```

#### SMTP Server

1) For your e-mail provider, get configuration of mail program, like
[here](https://support.google.com/a/answer/176600?hl=en)
2) Download [`template.env](./template.env) as `.env`
3) In the `.env` file, set following values (rest of the content preserve
intact):

```env
LUIGI_EMAIL_METHOD=smtp
LUIGI_EMAIL_RECIPIENT=<your-email-here>
LUIGI_EMAIL_SENDER=<your-email-here>

LUIGI_SMTP_HOST=<smtp-host-of-your-provider>
LUIGI_SMTP_PORT=<smtp-port-of-your-provider>
LUIGI_SMTP_NO_TLS=<False-if-your-provider-uses-TLS-True-otherwise>
LUIGI_SMTP_SSL=<False-if-your-provider-uses-TLS-True-otherwise>
LUIGI_SMTP_PASSWORD=<password-to-your-email-account>
LUIGI_SMTP_USERNAME=<login-to-your-email-account>
```

4) Launching processing with Docker, use additional switch
`--env-file .env`:

```bash
docker run -v /mydata:/data --env-file .env gmrukwa/msi-preprocessing '["my-dataset1","my-dataset2"]'
```

### History Persistence

Task history is collected to SQLite database. If you want to persist the
database, you need to mount the `/luigi` directory. This can be done via:

```bash
docker run -v tasks-history:/luigi -v /mydata:/data gmrukwa/msi-preprocessing
```
