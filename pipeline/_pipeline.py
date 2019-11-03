import luigi

from pipeline.export import ExportCsv
from pipeline.gmm import MergeDataset
from pipeline.metadata import AssembleMetadata


class PreprocessingPipeline(luigi.Task):
    datasets = luigi.ListParameter(description="Names of the datasets to use")
    export_txt = luigi.BoolParameter(
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
