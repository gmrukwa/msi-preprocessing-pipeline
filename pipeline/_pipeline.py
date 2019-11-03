import luigi

from pipeline.gmm import Convolve
from pipeline.metadata import AssembleMetadata


class PreprocessingPipeline(luigi.Task):
    datasets = luigi.ListParameter(description="Names of the datasets to use")

    def requires(self):
        for dataset in self.datasets:
            yield AssembleMetadata(dataset=dataset)
        for dataset in self.datasets:
            yield Convolve(dataset=dataset, datasets=self.datasets)
        