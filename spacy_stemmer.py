import langcodes
from nltk.stem.snowball import SnowballStemmer
from spacy.language import Doc, Language
from spacy.tokens import Span, Token


SPACY_LANG_TO_NLTK_STEMMER_LANG = dict()
for lang_name in SnowballStemmer.languages:
    try:
        lang_tag = langcodes.find(lang_name).to_tag()
        SPACY_LANG_TO_NLTK_STEMMER_LANG[lang_tag] = lang_name
    except LookupError:
        continue
SPACY_LANG_TO_NLTK_STEMMER_LANG["en"] = "porter"


class Stemmer:
    def __init__(
        self,
        *,
        nlp: Language,
        lowercase: bool,
        attr_name: str,
        attr_name_ws_suffix: str
    ) -> None:
        self.nlp = nlp
        self.lowercase = lowercase
        self.attr_name = attr_name
        self.attr_name_ws = attr_name + attr_name_ws_suffix

        # Set language
        if self.nlp.lang in SPACY_LANG_TO_NLTK_STEMMER_LANG:
            self.stemmer = SnowballStemmer(
                language=SPACY_LANG_TO_NLTK_STEMMER_LANG[self.nlp.lang]
            )
        else:
            self.stemmer = SnowballStemmer(language="porter")

        # Set extensions
        if not Token.has_extension(self.attr_name):
            Token.set_extension(self.attr_name, default="")
        if not Token.has_extension(self.attr_name_ws):
            Token.set_extension(
                self.attr_name_ws,
                getter=lambda token: token._.get(self.attr_name) + token.whitespace_,
            )

        if not Span.has_extension(self.attr_name):
            Span.set_extension(
                self.attr_name,
                getter=lambda span: " ".join(
                    token._.get(self.attr_name) for token in span if not token.is_space
                ),
            )
        if not Span.has_extension(self.attr_name_ws):
            Span.set_extension(
                self.attr_name_ws,
                getter=lambda span: "".join(
                    token._.get(self.attr_name_ws) for token in span
                ),
            )

        if not Doc.has_extension(self.attr_name):
            Doc.set_extension(
                self.attr_name,
                getter=lambda doc: doc[:]._.get(self.attr_name),
            )
        if not Doc.has_extension(self.attr_name_ws):
            Doc.set_extension(
                self.attr_name_ws,
                getter=lambda doc: doc[:]._.get(self.attr_name_ws),
            )

    def __call__(self, doc: Doc) -> Doc:
        for token in doc:
            token._.set(
                self.attr_name,
                self.stemmer.stem(token.lower_ if self.lowercase else token.text),
            )
        return doc


@Language.factory(
    "stemmer",
    default_config={
        "lowercase": True,
        "attr_name": "stem",
        "attr_name_ws_suffix": "_with_ws",
    },
)
def make_stemmer(
    nlp: Language,
    name: str,
    *,
    lowercase: bool,
    attr_name: str,
    attr_name_ws_suffix: str
) -> Stemmer:
    return Stemmer(
        nlp=nlp,
        lowercase=lowercase,
        attr_name=attr_name,
        attr_name_ws_suffix=attr_name_ws_suffix,
    )
