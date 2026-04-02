from dataclasses import dataclass, field, asdict


@dataclass
class LatinMorphemeRecord:
    token: str
    segments: list
    language: str = 'latin'

    def to_dict(self):
        return asdict(self)


class LatinMorphemeSegmenter:
    LATIN_PREFIXES = [
        'a', 'ab', 'ad', 'ante', 'con', 'contra', 'de', 'dis', 'e', 'ex',
        'in', 'inter', 'ob', 'per', 'post', 'prae', 'pro', 're', 'sub',
        'super', 'trans', 'un'
    ]

    LATIN_VERB_ENDINGS = [
        'o', 's', 't', 'mus', 'tis', 'nt', 're', 'ri',
        'sse', 'isse', 'ari', 'iri',
    ]

    LATIN_NOUN_ENDINGS = [
        'a', 'ae', 'am', 'arum', 'as', 'e', 'em', 'es', 'ei', 'erum',
        'i', 'is', 'o', 'os', 'um', 'us', 'ui', 'u', 'uum', 'ibus', 'ium',
    ]

    def _sort_desc(self, lst):
        return sorted(lst, key=len, reverse=True)

    def segment(self, normalized_text: str) -> dict:
        token = normalized_text
        segments = []
        remaining = token

        prefixes = self._sort_desc(self.LATIN_PREFIXES)
        all_endings = self._sort_desc(list(set(self.LATIN_VERB_ENDINGS + self.LATIN_NOUN_ENDINGS)))

        # Try to strip a prefix
        prefix_found = None
        for prefix in prefixes:
            if remaining.startswith(prefix) and len(remaining) > len(prefix):
                prefix_found = prefix
                remaining = remaining[len(prefix):]
                break

        if prefix_found:
            segments.append({'form': prefix_found, 'role': 'prefix'})

        # Try to strip an ending
        ending_found = None
        for ending in all_endings:
            if remaining.endswith(ending) and len(remaining) > len(ending):
                ending_found = ending
                remaining = remaining[:-len(ending)]
                break

        if remaining:
            segments.append({'form': remaining, 'role': 'root'})

        if ending_found:
            segments.append({'form': ending_found, 'role': 'ending'})

        if not segments:
            segments.append({'form': token, 'role': 'root'})

        record = LatinMorphemeRecord(token=token, segments=segments, language='latin')
        return asdict(record)
