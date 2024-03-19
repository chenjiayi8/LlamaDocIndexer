""" Contains utility functions for working with patterns. """

import re


def wildcard_to_regex(wildcard):
    """Converts a wildcard pattern to a regex pattern."""
    # Escape regex special characters, except for '*'
    escaped = re.escape(wildcard).replace("\\*", ".*")
    # Add anchors to match the start and end of the string
    return r"^" + escaped + r"$"


def ignored_files_to_patterns(ignored_files):
    """Converts a list of wildcard patterns to regex patterns."""
    return [re.compile(wildcard_to_regex(p)) for p in ignored_files]
