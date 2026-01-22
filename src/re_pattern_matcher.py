""" Stellt die Klasse REqual zur Verfügung """

import re

class REqual(str):
    """
    Überschreibe str.__eq__ zum matchen eines regulären Ausdrucks.
    Inspiration: https://discuss.python.org/t/structural-pattern-matching-should-permit-regex-string-matches/22700/9
    """
    def __eq__(self, pattern):
        """
        Vergleicht den String mit einem Pattern.

        :param pattern: regulärer Ausdruck als String
        :return: True, falls das Pattern passt
        """
        return re.fullmatch(pattern, self)
