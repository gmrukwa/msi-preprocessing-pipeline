import json
import logging

import docker
import luigi
import luigi.contrib.docker_runner


logger = logging.getLogger('luigi-interface')


class DockerfileTask(luigi.contrib.docker_runner.DockerTask):
    @property
    def dockerfile(self):
        return None

    @property
    def tag(self):
        return None

    @property
    def image(self):
        return self.tag

    @property
    def path(self):
        return '.'

    def run(self):
        # This addresses the bug in the docker_runner
        self._image = self.image
        
        docker_host = self.docker_url or 'localhost'
        logger.info('Building image %s from %s on %s',
                    self._image, self.dockerfile, docker_host)
        log = self._client.build(
            path=self.path, dockerfile=self.dockerfile, tag=self.tag)
        for msg in log:
            for line in msg.decode().splitlines():
                logger.debug(json.loads(line).get('stream', ''))
        super().run()


class BasicDockerTask(DockerfileTask):
    data_dir = luigi.Parameter()
    intermediate_dir = luigi.Parameter()
    result_dir = luigi.Parameter()

    @property
    def dockerfile(self):
        return 'docker/basic.dockerfile'
    
    @property
    def tag(self):
        return 'msi-preprocessing:basic'
    
    @property
    def binds(self):
        return super().binds + [
            '%s:/data' % self.data_dir,
            '%s:/intermediate' % self.intermediate_dir,
            '%s:/result' % self.result_dir,
        ]
