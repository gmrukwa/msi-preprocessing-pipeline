import luigi

from pipeline.metadata import AssembleMetadata
from pipeline.outlier import DetectOutliers


class PreprocessingPipeline(luigi.Task):
    datasets = luigi.ListParameter(description="Names of the datasets to use")

    def requires(self):
        for dataset in self.datasets:
            yield AssembleMetadata(dataset=dataset)
        yield DetectOutliers(datasets=self.datasets)
