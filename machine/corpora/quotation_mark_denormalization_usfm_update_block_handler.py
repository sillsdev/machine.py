from .punctuation_analysis.quote_convention import QuoteConvention
from .quotation_mark_update_settings import QuotationMarkUpdateSettings
from .quote_convention_changing_usfm_update_block_handler import QuoteConventionChangingUsfmUpdateBlockHandler


# This is a convenience class so that users don't have to know to normalize the source quote convention
class QuotationMarkDenormalizationUsfmUpdateBlockHandler(QuoteConventionChangingUsfmUpdateBlockHandler):

    def __init__(
        self,
        source_quote_convention: QuoteConvention,
        target_quote_convention: QuoteConvention,
        settings: QuotationMarkUpdateSettings = QuotationMarkUpdateSettings(),
    ):
        super().__init__(source_quote_convention.normalize(), target_quote_convention, settings)
