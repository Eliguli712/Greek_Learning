from dataclasses import dataclass, field, asdict


@dataclass
class MorphemeRecord:
    token: str
    segments: list
    language: str

    def to_dict(self):
        return asdict(self)


class MorphemeSegmenter:
    GREEK_PREFIXES = [
        'ἀ', 'ἀν', 'συν', 'κατα', 'ἐπί', 'ἀπό', 'διά', 'ἐκ', 'ἐν',
        'μετά', 'παρά', 'περί', 'πρό', 'πρός', 'ὑπό', 'ὑπέρ', 'ἀντί',
        'ἀμφί', 'εἰς', 'ἐξ', 'ἐφ'
    ]

    GREEK_VERB_ENDINGS = [
        'ω', 'εις', 'ει', 'ομεν', 'ετε', 'ουσι', 'ουσιν',
        'ον', 'ες', 'ε', 'ομεν', 'ετε', 'ον',
        'σω', 'σεις', 'σει', 'σομεν', 'σετε', 'σουσι',
        'α', 'ας', 'αμεν', 'ατε', 'αν',
        'ναι', 'ειν', 'σαι', 'ομαι', 'εσαι', 'εται',
    ]

    GREEK_NOUN_ENDINGS = [
        'ος', 'ου', 'ῳ', 'ον', 'οι', 'ων', 'οις', 'ους',
        'α', 'ης', 'ῃ', 'αν', 'αι', 'ῶν', 'αις', 'ας',
        'η', 'ης', 'ῃ', 'ην', 'αι', 'ῶν', 'αις', 'ας',
        'ις', 'εως', 'ει', 'ιν', 'εις', 'εων', 'εσι',
        'υς', 'υος', 'υι', 'υν', 'υες', 'υων', 'υσι',
    ]

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

    def _sort_by_length_desc(self, lst):
        return sorted(lst, key=len, reverse=True)

    def segment(self, normalized_text: str, language: str = 'greek') -> dict:
        token = normalized_text
        segments = []
        remaining = token

        if language == 'greek':
            prefixes = self._sort_by_length_desc(self.GREEK_PREFIXES)
            verb_endings = self._sort_by_length_desc(self.GREEK_VERB_ENDINGS)
            noun_endings = self._sort_by_length_desc(self.GREEK_NOUN_ENDINGS)
        else:
            prefixes = self._sort_by_length_desc(self.LATIN_PREFIXES)
            verb_endings = self._sort_by_length_desc(self.LATIN_VERB_ENDINGS)
            noun_endings = self._sort_by_length_desc(self.LATIN_NOUN_ENDINGS)

        # Try to strip a prefix
        prefix_found = None
        for prefix in prefixes:
            if remaining.startswith(prefix) and len(remaining) > len(prefix):
                prefix_found = prefix
                remaining = remaining[len(prefix):]
                break

        if prefix_found:
            segments.append({'form': prefix_found, 'role': 'prefix'})

        # Try to strip an ending (verb endings first, then noun endings)
        ending_found = None
        all_endings = [(e, 'ending') for e in verb_endings] + [(e, 'ending') for e in noun_endings]
        # deduplicate preserving order
        seen = set()
        unique_endings = []
        for e, role in all_endings:
            if e not in seen:
                seen.add(e)
                unique_endings.append((e, role))

        for ending, role in unique_endings:
            if remaining.endswith(ending) and len(remaining) > len(ending):
                ending_found = ending
                remaining = remaining[:-len(ending)]
                break

        # Whatever remains is the root
        if remaining:
            segments.append({'form': remaining, 'role': 'root'})

        if ending_found:
            segments.append({'form': ending_found, 'role': 'ending'})

        # If nothing was segmented, the whole token is root
        if not segments:
            segments.append({'form': token, 'role': 'root'})

        record = MorphemeRecord(token=token, segments=segments, language=language)
        return asdict(record)
