import os

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
    pool_size = luigi.IntParameter(
        default=os.cpu_count() or 1,
        description='Size of parallel pool to use for computations. Choose carefully '
        'to not exceed the memory.',
        significant=False, visibility=luigi.parameter.ParameterVisibility.HIDDEN
    )

    def requires(self):
        for dataset in self.datasets:
            yield AssembleMetadata(dataset=dataset, pool_size=self.pool_size)
        for dataset in self.datasets:
            yield MergeDataset(dataset=dataset, datasets=self.datasets,
                               pool_size=self.pool_size)
        if self.export_csv:
            for dataset in self.datasets:
                yield ExportCsv(dataset=dataset, datasets=self.datasets,
                                pool_size=self.pool_size)


@PreprocessingPipeline.event_handler(luigi.Event.SUCCESS)
def send_notification(task):
    if luigi.notifications.email().receiver:
        luigi.notifications.send_email(
            subject='MSI Preprocessing Pipeline finished!',
            message='Your preprocessing pipeline has completed successfully.',
            sender=luigi.notifications.email().sender,
            recipients=[luigi.notifications.email().receiver]
        )
