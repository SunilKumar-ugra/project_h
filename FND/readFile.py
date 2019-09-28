# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 14:46:56 2019

@author: USER
"""



import os
import csv

		



def readcsv():
    file_path = input("Enter the file name:")
        
    if file_path.endswith('.csv'):
        print(file_path)
        with open(file_path, 'r', encoding="utf-8") as f1:
            csvlines = csv.reader(f1, delimiter=',')
            for lineNum, line in enumerate(csvlines):
                if lineNum == 0:
                    if len(line)==2:
                        if line[0] == 'Statement' and line[1] == 'Label':
                            print('pass')
                            return file_path
                        else:
                            print("Invalid headings")
                            os._exit(1)
                                  
                    else:
                        print("invalid count of columns")
                        os._exit(1)
    else:
        print("Invalid file")
        os._exit(1)


			
'''
def readcsv():
file_path = input("Enter the file name:")
    
if file_path.endswith('.csv'):
print(file_path)
with open(file_path, 'r', encoding="utf-8") as f1:
csvlines = csv.reader(f1, delimiter=',')
for lineNum, line in enumerate(csvlines):
if lineNum == 0:
if len(line)==2:
if line[0] == 'Statement' and line[1] == 'Label':
print('pass')
return file_path
else:
print('Invalid headings")
os._exit()
                      
else:
print("invalid count of columns")
os._exit()
else:
print("Invalid file")
os._exit()
'''   			
	
	
				
      