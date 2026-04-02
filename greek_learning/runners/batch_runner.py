from greek_learning.runners.single_runner import SingleRunner


class BatchRunner:
    def __init__(self, language: str = 'greek', max_items: int = None):
        self.language = language
        self.max_items = max_items

    def run(self, items: list) -> list:
        runner = SingleRunner(language=self.language)
        results = []
        for i, item in enumerate(items):
            if self.max_items is not None and i >= self.max_items:
                break
            results.append(runner.run(item))
        return results

    def run_controlled(self, items: list, filters: dict = None) -> list:
        filters = filters or {}
        language = filters.get('language', self.language)
        min_tokens = filters.get('min_tokens', None)
        max_tokens = filters.get('max_tokens', None)

        runner = SingleRunner(language=language)
        results = []
        count = 0
        for item in items:
            if self.max_items is not None and count >= self.max_items:
                break
            token_count = len(item.split())
            if min_tokens is not None and token_count < min_tokens:
                continue
            if max_tokens is not None and token_count > max_tokens:
                continue
            results.append(runner.run(item))
            count += 1
        return results
