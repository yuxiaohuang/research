# Please cite the following paper when using the code

class Names:
    """The Names class"""

    def __init__(self):

        # The header
        self.header = None

        # The delimiter
        self.delim_whitespace = False

        # The separator
        self.sep = ','

        # The place holder for missing values
        self.place_holder_for_missing_vals = '?'

        # The (name of the) columns
        self.columns = None

        # The (name of the) target
        self.target = None

        # The (name of the) classes that should be combined
        self.combine_classes = {}

        # The (name of the) features that should be excluded
        self.exclude_features = []

        # The (name of the) categorical features
        self.categorical_features = []

        # The (name of the) features
        # This is not a parameter in the names file,
        # since it can either be inferred based on self.columns and self.target
        self.features = None

        # The parameter names
        self.para_names = ['header',
                           'delim_whitespace',
                           'sep',
                           'place_holder_for_missing_vals',
                           'columns',
                           'target',
                           'combine_classes',
                           'exclude_features',
                           'categorical_features']