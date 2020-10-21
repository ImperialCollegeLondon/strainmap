import re


def get_sa_location(dataset):
    pattern = r"[sS][aA][xX]?([0-9])"
    m = re.search(pattern, dataset)
    return int(m.group(1)) if hasattr(m, "group") else 99
