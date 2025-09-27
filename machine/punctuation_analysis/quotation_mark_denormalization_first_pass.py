from .quotation_mark_update_first_pass import QuotationMarkUpdateFirstPass
from .quote_convention import QuoteConvention


# This is a convenience class so that users don't have to know to pass in two quote conventions,
# with the first being the normalized version of the second.
class QuotationMarkDenormalizationFirstPass(QuotationMarkUpdateFirstPass):

    def __init__(self, target_quote_convention: QuoteConvention):
        super().__init__(target_quote_convention.normalize(), target_quote_convention)
