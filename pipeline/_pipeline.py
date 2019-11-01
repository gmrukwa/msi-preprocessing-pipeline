import luigi

from pipeline.metadata import AssembleMetadata
from pipeline.resampling import ResampleDataset


class PreprocessingPipeline(luigi.Task):
    datasets = luigi.ListParameter(description="Names of the datasets to use")

    def requires(self):
        for dataset in self.datasets:
            yield AssembleMetadata(dataset=dataset)
        for dataset in self.datasets:
            yield ResampleDataset(dataset=dataset, datasets=self.datasets)
