# -*- coding: utf-8 -*-
"""Combine metadata with tumor types

Created on Sat Jun 16 13:39:48 2018

@author: Grzegorz Mrukwa
"""
import json
import os
from typing import List, NamedTuple

import numpy as np
import pandas as pd


class MatrixInformation:
    def __init__(self, path: str):
        with open(path) as infile:
            content = json.load(infile)
        assert 'cancerTypes' in content
        assert 'donorsId' in content
        assert 'regionNumbers' in content
        self.tumor_type = np.array(content['cancerTypes'])
        self.tumor_type[self.tumor_type==''] = '?'
        self.region_number = np.array(content['regionNumbers'])
        self.region_number = self.region_number.astype(int)
        self.donor = np.array(content['donorsId'])

    def _get_halves(self, second_half: bool=None):
        if second_half is not None and second_half:
            start = int(self.region_number.shape[1] / 2)
            return (
                self.region_number[:, start:],
                self.tumor_type[:, start:],
                self.donor[:, start:]
            )
        elif second_half is not None and not second_half:
            end = int(self.region_number.shape[1] / 2)
            return (
                self.region_number[:, :end],
                self.tumor_type[:, :end],
                self.donor[:, :end]
            )
        else:
            return self.region_number, self.tumor_type, self.donor
    
    def get_tumor(self, region_numbers: np.ndarray, second_half: bool=None) \
            -> np.ndarray:
        regions, types, _ = self._get_halves(second_half)
        matches = region_numbers[:, np.newaxis] == regions.ravel()[np.newaxis,:]
        locations = np.argmax(matches, axis=1)
        tumor_types = types.ravel()[(locations,)]
        unknown_regions = np.max(matches[:, locations], axis=1) == 0
        tumor_types[unknown_regions] = 'N'
        return tumor_types
    
    def get_donor(self, region_numbers: np.ndarray, second_half: bool=None) \
            -> np.ndarray:
        regions, _, donors = self._get_halves(second_half)
        matches = region_numbers[:, np.newaxis] == regions.ravel()[np.newaxis,:]
        locations = np.argmax(matches, axis=1)
        donor_ids = donors.ravel()[(locations,)]
        unknown_regions = np.max(matches[:, locations], axis=1) == 0
        donor_ids[unknown_regions] = '?'
        return donor_ids


MetadataMatch = NamedTuple('MetadataMatch', [
    ('source', str),
    ('explanation', str),
    ('second_half', bool)
])


def result_filename(path: str) -> str:
    return os.path.join(os.path.dirname(path), 'corrected_metadata.csv')


def read_matches(path: str) -> List[MetadataMatch]:
    with open(path) as explanation_matches_file:
        explanation_matches = json.load(explanation_matches_file)
    return [MetadataMatch(**match) for match in explanation_matches]
    

if __name__ == '__main__':
    matches_source = 'metadata_matches.json'
    matches = read_matches(matches_source)
    for match in matches:
        metadata = pd.read_csv(match.source, index_col=False)
        explanation = MatrixInformation(match.explanation)
        metadata['diagnosis'] = explanation.get_tumor(metadata.R + 1,
                                                      match.second_half)
        metadata['donor'] = explanation.get_donor(metadata.R + 1,
                                                  match.second_half)
        metadata.X -= metadata.X.min()
        metadata.Y -= metadata.Y.min()
        metadata.to_csv(result_filename(match.source), index=False)
