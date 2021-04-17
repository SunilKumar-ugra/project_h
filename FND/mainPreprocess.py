from FND import  main2
import seaborn as sns
from FND import readFile
import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import os
import csv

import matplotlib.pyplot as plt

def cleanText(text):
    text = re.sub(r'([^\s\w]|_)+', '',text)  
    text = re.sub(r"\s+"," ", text)
    text = text.lower()
    return text


def dataInitial(path):
    #Filename = input("Enter the file name:")
    Filename = readcsv(path)
    
    dataf = pd.read_csv(Filename)
    print(dataf.head(10))
    print('\n PRE-PROCESS \n')
    dataf = dataf[pd.notnull(dataf['Statement'])]
    dataf = dataf[pd.notnull(dataf['Label'])]
    #print(dataf.head(10))
    #df.shape
    dataf.index = range(dataf.shape[0])
    #print(dataf.head(10))
    			
    				
    dataf['Statement'] = dataf['Statement'].apply(cleanText)
    #print(dataf.head(10))
    			
    stop = stopwords.words('english')
    dataf['Statement'] = dataf['Statement'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    #print(dataf.head(10))
    cnt_pro = dataf['Label'].value_counts()
    plt.figure(figsize=(5,8))
    sns.barplot(cnt_pro.index, cnt_pro.values, alpha=0.8)
    plt.title('Data Distribution', fontsize=18)
    plt.ylabel('Number of Occurrences', fontsize=18)
    plt.xlabel('Label', fontsize=18)
    plt.yticks(fontsize=18)
    plt.xticks(rotation=0,fontsize=18)
    plt.savefig('dataDistribution.png')
    plt.show()
    return dataf

def initial(path):
    
    df = dataInitial(path)
    print(df.head(10))

    main2.first(df)


def getcsvPath(csv):
    Data_DIR = os.getcwd() + '\\FND\\Dataset\\' + csv
    print("test" + Data_DIR)
    return initial(Data_DIR)

def readcsv(file):
    file_path = file

    if file_path.endswith('.csv'):
        print(file_path)
        with open(file_path, 'r', encoding="utf-8") as f1:
            csvlines = csv.reader(f1, delimiter=',')
            for lineNum, line in enumerate(csvlines):
                if lineNum == 0:
                    if len(line) == 2:
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


'''if __name__ == '__main__':
	initial()'''
    
