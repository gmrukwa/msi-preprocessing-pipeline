import luigi

from pipeline.metadata import AssembleMetadata
from pipeline.baseline import RemoveBaseline


class PreprocessingPipeline(luigi.Task):
    datasets = luigi.ListParameter(description="Names of the datasets to use")

    def requires(self):
        for dataset in self.datasets:
            yield AssembleMetadata(dataset=dataset)
        for dataset in self.datasets:
            yield RemoveBaseline(dataset=dataset, datasets=self.datasets)
