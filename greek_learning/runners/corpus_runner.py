import os
from typing import Iterable, Generator
from greek_learning.runners.single_runner import SingleRunner


class CorpusRunner:
    def __init__(self, language: str = 'greek', batch_size: int = 100):
        self.language = language
        self.batch_size = batch_size

    def run_file(self, filepath: str) -> list:
        runner = SingleRunner(language=self.language)
        results = []
        with open(filepath, encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if line:
                    results.append(runner.run(line))
        return results

    def run_directory(self, dirpath: str, extension: str = '.txt') -> dict:
        output = {}
        for filename in os.listdir(dirpath):
            if filename.endswith(extension):
                filepath = os.path.join(dirpath, filename)
                output[filename] = self.run_file(filepath)
        return output

    def run_stream(self, lines: Iterable) -> Generator:
        runner = SingleRunner(language=self.language)
        for line in lines:
            line = line.strip() if isinstance(line, str) else line
            if line:
                yield runner.run(line)
