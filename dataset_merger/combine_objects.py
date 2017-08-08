"""
Helper functions to combine (merge) objects (artists or artworks)
"""
import numpy as np

import prepare_artists
from art_utils.pandas_tools import is_null_object


def take_first(objects_list, key):
    items = [obj[key] for obj in objects_list if not is_null_object(obj[key])]
    if items:
        return items[0]
    else:
        return np.nan


def merge_years_range(objects_list, key, fallback_to_group_works=True):
    assert len(objects_list)
    ranges = [obj[key] for obj in objects_list if
              not obj['is_group_work'] or is_null_object(obj['is_group_work'])]
    if not len(ranges) and fallback_to_group_works:
        ranges = [obj[key] for obj in objects_list]

    years = [x for rng in ranges if not is_null_object(rng) for x in rng]
    years_range = prepare_artists.create_years_range(years)
    return years_range


def take_union(objects_list, key, take_group_works=True):
    for obj in objects_list:
        if not is_null_object(obj[key]) and not isinstance(obj[key], list):
            obj[key] = [obj[key]]
        assert isinstance(obj[key], list) or is_null_object(obj[key]), obj[key]

    # if years => don't take group works
    if take_group_works:
        items = list(set(
            [x for obj in objects_list if not is_null_object(obj[key]) for x in obj[key]]))
    else:
        items = list(set(
            [x for obj in objects_list if not is_null_object(obj[key]) for x in obj[key]
             if not obj['is_group_work'] or is_null_object(obj['is_group_work'])]))
    items = [x for x in items if not is_null_object(x)]
    for x in items:
        assert not is_null_object(x), x
    if not items:
        items = np.nan
    return items
