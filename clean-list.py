#!/usr/bin/env python

import sys
import os
from os import path
from timeit import default_timer as timer
import uproot
import csv

def clean_list(input_list, output_list='out.csv', fail_list='fail.csv') :

    if path.exists(output_list) :
        os.remove(output_list)
    if path.exists(fail_list) :
        os.remove(fail_list)

    with open(input_list) as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            file_name = row[0]
            exist = path.exists(file_name)

            good = True
            if exist :
                try:
                    uproot.open(file_name)
                except :
                    print('Failed to load ', file_name)
                    row.append('Fail_Load')
                    good = False
            else :
                good = False

            current_list = output_list
            if not good:
                current_list = fail_list

            with open(current_list, 'a+', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=' ')
                writer.writerow(row)

if __name__ == '__main__' :
    input_list=sys.argv[1]

    clean_list(input_list)
