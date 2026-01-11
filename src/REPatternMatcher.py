import re

class REqual(str):
    """Override str.__eq__ to match a regex pattern.
    Source: https://discuss.python.org/t/structural-pattern-matching-should-permit-regex-string-matches/22700/9
    """
    def __eq__(self, pattern):
        return re.fullmatch(pattern, self)