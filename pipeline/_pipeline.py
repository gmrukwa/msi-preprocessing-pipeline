import luigi
import luigi.notifications

from pipeline.export import ExportCsv
from pipeline.gmm import MergeDataset
from pipeline.metadata import AssembleMetadata


class PreprocessingPipeline(luigi.Task):
    datasets = luigi.ListParameter(description="Names of the datasets to use")
    export_csv = luigi.BoolParameter(
        description="If specified, exports spectra as csv files",
        significant=False, visibility=luigi.parameter.ParameterVisibility.HIDDEN
    )

    def requires(self):
        for dataset in self.datasets:
            yield AssembleMetadata(dataset=dataset)
        for dataset in self.datasets:
            yield MergeDataset(dataset=dataset, datasets=self.datasets)
        if self.export_txt:
            for dataset in self.datasets:
                yield ExportCsv(dataset=dataset, datasets=self.datasets)


@PreprocessingPipeline.event_handler(luigi.Event.SUCCESS)
def send_notification(task):
    if luigi.notifications.email().receiver:
        luigi.notifications.send_email(
            subject='MSI Preprocessing Pipeline finished!',
            message='Your preprocessing pipeline has completed successfully.',
            sender=luigi.notifications.email().sender,
            recipients=[luigi.notifications.email().receiver]
        )
