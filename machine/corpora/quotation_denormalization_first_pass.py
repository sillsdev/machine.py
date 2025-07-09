from .punctuation_analysis.quote_convention import QuoteConvention
from .quotation_mark_update_first_pass import QuotationMarkUpdateFirstPass


# This is a convenience class so that users don't have to know to normalize the source quote convention
class QuotationDenormalizationFirstPass(QuotationMarkUpdateFirstPass):

    def __init__(self, source_quote_convention: QuoteConvention, target_quote_convention: QuoteConvention):
        super().__init__(source_quote_convention.normalize(), target_quote_convention)
