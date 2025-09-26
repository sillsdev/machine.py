from .quotation_mark_update_settings import QuotationMarkUpdateSettings
from .quote_convention import QuoteConvention
from .quote_convention_changing_usfm_update_block_handler import QuoteConventionChangingUsfmUpdateBlockHandler


# This is a convenience class so that users don't have to know to pass in two quote conventions,
# with the first being the normalized version of the second.
class QuotationMarkDenormalizationUsfmUpdateBlockHandler(QuoteConventionChangingUsfmUpdateBlockHandler):

    def __init__(
        self,
        target_quote_convention: QuoteConvention,
        settings: QuotationMarkUpdateSettings = QuotationMarkUpdateSettings(),
    ):
        super().__init__(target_quote_convention.normalize(), target_quote_convention, settings)
