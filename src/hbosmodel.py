

from sklearn.cluster import KMeans
import numpy as np
from pandas import DataFrame
from math import pow
import math



from pandas import DataFrame
import datetime
from itertools import repeat

class HBOS:
        
    def __init__(self, log_scale=True, ranked=False, bin_info_array=[], mode_array=[], nominal_array=[]):
        self.log_scale = log_scale
        self.ranked = ranked
        self.bin_info_array = bin_info_array
        self.mode_array = mode_array
        self.nominal_array = nominal_array
        # self.histogram_list = []
        
    def fit(self, data):
        attr_size = len(data.columns)
        total_data_size = len(data)
        
        # init params if needed
        if len(self.bin_info_array) == 0:
            self.bin_info_array = list(repeat(-1, attr_size))
        
        if len(self.mode_array) == 0:
            self.mode_array = list(repeat('dynamic binwidth', attr_size))
            
        if len(self.nominal_array) == 0:
            self.nominal_array = list(repeat(False, attr_size))
        
        if self.ranked:
            self.log_scale = False
            
        normal = 1.0
        
        # calculate standard _bin size if needed
        for i in range(len(self.bin_info_array)):
            if self.bin_info_array[ i ] == -1:
                self.bin_info_array[ i ] = round(math.sqrt(len(data)))
                
        # initialize histogram
        self.histogram_list = []
        for i in range(attr_size):
            self.histogram_list.append([])
            
        # save maximum value for every attribute(needed to normalize _bin width)
        maximum_value_of_rows = data.apply(max).values
        
        # sort data
        sorted_data = data.apply(sorted)
        
        # create histograms
        for attrIndex in range(len(sorted_data.columns)):
            attr = sorted_data.columns[ attrIndex ]
            last = 0
            bin_start = sorted_data[ attr ][ 0 ]
            if self.mode_array[ attrIndex ] == 'dynamic binwidth':
                if self.nominal_array[ attrIndex ] == True:
                    while last < len(sorted_data) - 1:
                        last = self.create_dynamic_histogram(self.histogram_list, sorted_data, last, 1, attrIndex, True)
                else:
                    length = len(sorted_data)
                    binwidth = self.bin_info_array[ attrIndex ]
                    while last < len(sorted_data) - 1:
                        values_per_bin = math.floor(len(sorted_data) / self.bin_info_array[ attrIndex ])
                        last = self.create_dynamic_histogram(self.histogram_list, sorted_data, last, values_per_bin, attrIndex, False)
                        if binwidth > 1:
                            length = length - self.histogram_list[ attrIndex ][ -1 ].quantity
                            binwidth = binwidth - 1
            else:
                count_bins = 0
                binwidth = (sorted_data[ attr ][ len(sorted_data) - 1 ] - sorted_data[ attr ][ 0 ]) / self.bin_info_array[ attrIndex ]
                if (self.nominal_array[ attrIndex ] == True) | (binwidth == 0):
                    binwidth = 1
                while last < len(sorted_data):
                    is_last_bin = count_bins == self.bin_info_array[ attrIndex ] - 1
                    last = self.create_static_histogram(self.histogram_list, sorted_data, last, binwidth, attrIndex, bin_start, is_last_bin)
                    bin_start = bin_start + binwidth
                    count_bins = count_bins + 1
    
        # calculate score using normalized _bin width
        # _bin width is normalized to the number of datapoints
        # save maximum score for every attr(needed to normalize score)
        max_score = []
        
        # loop for all histograms
        for i in range(len(self.histogram_list)):
            max_score.append(0)
            histogram = self.histogram_list[ i ]
            
            # loop for all bins
            for k in range(len(histogram)):
                _bin = histogram[ k ]
                _bin.total_data_size = total_data_size
                _bin.calc_score(maximum_value_of_rows[ i ])
                if max_score[ i ] < _bin.score:
                    max_score[ i ] = _bin.score
                    
        for i in range(len(self.histogram_list)):
            histogram = self.histogram_list[ i ]
            for k in range(len(histogram)):
                _bin = histogram[ k ]
                _bin.normalize_score(normal, max_score[ i ], self.log_scale)
                
        # if ranked
        
    def predict(self, data):
        score_array = []
        for i in range(len(data)):
            each_data = data.values[ i ]
            value = 1
            if self.log_scale | self.ranked:
                value = 0
            for attr in range(len(data.columns)):
                score = self.get_score(self.histogram_list[ attr ], each_data[ attr ])
                if self.log_scale:
                    value = value + score
                elif self.ranked:
                    value = value + score
                else:
                    value = value * score
            score_array.append(value)
        return score_array
    
    def fit_predict(self, data):
        self.fit(data)
        return self.predict(data)
                
    def get_score(self, histogram, value):
        for i in range(len(histogram) - 1):
            _bin = histogram[ i ]
            if (_bin.range_from <= value) & (value < _bin.range_to):
                return _bin.score
            
        _bin = histogram[ -1 ]
        if (_bin.range_from <= value) & (value <= _bin.range_to):
            return _bin.score
        return 0
          
    @staticmethod  
    def check_amount(sortedData, first_occurrence, values_per_bin, attr):
        # check if there are more than values_per_bin values of a given value
        if first_occurrence + values_per_bin < len(sortedData):
            if sortedData[ attr ][ first_occurrence ] == sortedData[ attr ][ first_occurrence + values_per_bin ]:
                return True
            else:
                return False
        else:
            return False
            
    @staticmethod
    def create_dynamic_histogram(histogram_list, sortedData, first_index, values_per_bin, attrIndex, isNominal):
        last_index = 0
        attr = sortedData.columns[ attrIndex ]
        
        # create new _bin
        _bin = HistogramBin(sortedData[ attr ][ first_index ], 0, 0)
            
        # check if an end of the data is near
        if first_index + values_per_bin < len(sortedData):
            last_index = first_index + values_per_bin
        else:
            last_index = len(sortedData)
    
        # the first value always goes to the _bin
        _bin.add_quantitiy(1)
        
        # for every other value
        # check if it is the same as the last value
        # if so
        #   put it into the _bin
        # if not
        #   check if there are more than values_per_bin of that value
        #   if so
        #     open new _bin
        #   if not
        #     continue putting the value into the _bin
        
        cursor = first_index
        for i in range(first_index + 1, last_index):
            if sortedData[ attr ][ i ] == sortedData[ attr ][ cursor ]:
                _bin.add_quantitiy(1)
                cursor = cursor + 1
            else:
                if HBOS.check_amount(sortedData, i, values_per_bin, attr):
                    break
                else:
                    _bin.add_quantitiy(1)
                    cursor = cursor + 1
                    
        # continue to put values in the _bin until a new values arrive
        for i in range(cursor + 1, len(sortedData)):
            if sortedData[ attr ][ i ] == sortedData[ attr ][ cursor ]:
                _bin.quantity = _bin.quantity + 1
                cursor = cursor + 1
            else:
                break
                
        # adjust range of the bins
        if cursor + 1 < len(sortedData):
            _bin.range_to = sortedData[ attr ][ cursor + 1 ]
        else:  # last data
            if isNominal:
                _bin.range_to = sortedData[ attr ][ len(sortedData) - 1 ] + 1
            else:
                _bin.range_to = sortedData[ attr ][ len(sortedData) - 1 ]
                
        # save _bin
        if _bin.range_to - _bin.range_from > 0:
            histogram_list[ attrIndex ].append(_bin)
        elif len(histogram_list[ attrIndex ]) == 0:
            _bin.range_to = _bin.range_to + 1
            histogram_list[ attrIndex ].append(_bin)
        else:
            # if the _bin would have length of zero
            # we merge it with previous _bin
            # this can happen at the end of the histogram
            lastBin = histogram_list[ attrIndex ][ -1 ]
            lastBin.add_quantitiy(_bin.quantity)
            lastBin.range_to = _bin.range_to
        
        return cursor + 1

    @staticmethod
    def create_static_histogram(histogram_list, sorted_data, first_index, binwidth, attrIndex, bin_start, last_bin):
        attr = sorted_data.columns[ attrIndex ]
        _bin = HistogramBin(bin_start, bin_start + binwidth, 0)
        if last_bin == True:
            _bin = HistogramBin(bin_start, sorted_data[ attr ][ len(sorted_data) - 1 ], 0)
        
        last = first_index - 1
        cursor = first_index
        
        while True:
            if cursor >= len(sorted_data):
                break
            if sorted_data[ attr ][ cursor ] > _bin.range_to:
                break
            _bin.quantity = _bin.quantity + 1
            last = cursor
            cursor = cursor + 1
            
        histogram_list[ attrIndex ].append(_bin)
        return last + 1                 
       

class HistogramBin:

    def __init__(self, range_from, range_to, quantity):
        self.range_from = range_from
        self.range_to = range_to
        self.quantity = quantity
        self.score = 0
        self.total_data_size = 0
        
    def get_height(self):
        width = self.range_to - self.range_from
        height = self.quantity / width
        return height
    
    def add_quantitiy(self, anz):
        self.quantity = self.quantity + anz
        
    def calc_score(self, max_score):
        if max_score == 0:
            max_score = 1
        
        if self.quantity > 0:
            self.score = self.quantity / ((self.range_to - self.range_from) * self.total_data_size / abs(max_score))
        
    def normalize_score(self, normal, max_score, log_scale):
        self.score = self.score * normal / max_score
        if(self.score == 0):
            return
        self.score = 1 / self.score
        if log_scale:
            self.score = math.log10(self.score)