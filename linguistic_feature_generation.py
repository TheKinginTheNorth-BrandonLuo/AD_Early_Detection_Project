import glob
import os
import spacy
from spacy.lang.en import English
import re
import pandas as pd
from collections import Counter
import re
import math
import seaborn as sns
import datetime
import matplotlib.dates as dates
from lexicalrichness import LexicalRichness
import numpy as np
from textacy import TextStats
nlp = spacy.load('en_core_web_sm')

def pre_process(text):
    # lowercase
    text=text.lower()
    #remove tags
    text=re.sub("<!--?.*?-->"," ",text)
    # remove special characters and digits
    text=re.sub("\n","",text)
    text = re.sub(" -- ","",text)
    text = re.sub("[\(\[].*?[\)\]]", "", text)
    return text

#Adverbs
def Adverbs(corpus):
    doc_adv = []
    for i in corpus:
        doc = nlp(i)
        s_n = list(doc.sents)
        T = 0
        for j in s_n:
            for token in j:
                if token.pos_ == 'ADV':
                    T += 1
        doc_adv.append(T)
    return doc_adv

def Nouns(corpus):
    doc_noun = []
    for i in corpus:
        doc = nlp(i)
        s_n = list(doc.sents)
        T = 0
        for j in s_n:
            for token in j:
                if token.pos_ == 'NOUN':
                    T += 1
        doc_noun.append(T)
    return doc_noun

def Verbs(corpus):
    doc_verb = []
    for i in corpus:
        doc = nlp(i)
        s_n = list(doc.sents)
        T = 0
        for j in s_n:
            for token in j:
                if token.pos_ == 'VERB':
                    T += 1
        doc_verb.append(T)
    return doc_verb

def Pron(corpus):
    doc_pron = []
    for i in corpus:
        doc = nlp(i)
        s_n = list(doc.sents)
        T = 0
        for j in s_n:
            for token in j:
                if token.pos_ == 'PRON':
                    T += 1
        doc_pron.append(T)
    return doc_pron

#rate of frequency(excluding stop words)
def word_freq_rate(corpus):
    word_freq_rate = []
    for i in corpus:
        doc = nlp(i)
        words = [token.text for token in doc if token.is_stop != True and token.is_punct != True]
        word_freq = Counter(words)
        freq_dict={}
        for name,i in word_freq.items():
            if i>10:
                freq_dict[name] = i
        rate = 0
        for key,value in freq_dict.items():
            rate += value/len(doc)
        rate_percent = rate*100
        word_freq_rate.append(rate_percent)
    return word_freq_rate

#rate of frequency in each document without excluding stop words
def word_freq_rate_Nsw(corpus):
    word_freq_rate = []
    for i in corpus:
        doc = nlp(i)
        words = [token.text for token in doc if token.is_punct != True]
        word_freq = Counter(words)
        freq_dict={}
        for name,i in word_freq.items():
            if i>10:
                freq_dict[name] = i
        rate = 0
        for key,value in freq_dict.items():
            rate += value/len(doc)
        rate_percent = rate*100
        word_freq_rate.append(rate_percent)
    return word_freq_rate

#Rate of verb Frequency
def verb_freq_rate(corpus):
    verb_freq_rate = []
    for i in corpus:
        doc = nlp(i)
        verbs = [token.text for token in doc if token.is_punct != True and token.pos_ == "VERB"]
        verb_freq = Counter(verbs)
        freq_dict={}
        for name,i in verb_freq.items():
            if i>10:
                freq_dict[name] = i
        rate = 0
        for key,value in freq_dict.items():
            rate += value/len(doc)
        rate_percent = rate*100
        verb_freq_rate.append(rate_percent)
    return verb_freq_rate

#Noun Frequency Rate
def noun_freq_rate(corpus):
    noun_freq_rate = []
    for i in corpus:
        doc = nlp(i)
        nouns = [token.text for token in doc if token.is_punct != True and token.pos_ == "NOUN"]

        noun_freq = Counter(nouns)
        freq_dict={}
        for name,i in noun_freq.items():
            if i>10:
                freq_dict[name] = i
        rate = 0
        for key,value in freq_dict.items():
            rate += value/len(doc)
        rate_percent = rate*100
        noun_freq_rate.append(rate_percent)
    return noun_freq_rate

#Pronoun Frequency Rate
def pron_freq_rate(corpus):
    PRON_freq_rate = []

    for i in corpus:
        doc = nlp(i)
        prons = [token.text for token in doc if token.is_punct != True and token.pos_ == "PRON"]


        pron_freq = Counter(prons)
        freq_dict={}
        for name,i in pron_freq.items():
            if i>10:
                freq_dict[name] = i
        rate = 0
        for key,value in freq_dict.items():
            rate += value/len(doc)
        rate_percent = rate*100
        PRON_freq_rate.append(rate_percent)
    return PRON_freq_rate

#Adverb Frequency rate
def adv_freq_rate(corpus):
    ADV_freq_rate = []

    for i in corpus:
        doc = nlp(i)
        advs = [token.text for token in doc if token.is_punct != True and token.pos_ == "ADV"]


        advs_freq = Counter(advs)
        freq_dict={}
        for name,i in advs_freq.items():
            if i>10:
                freq_dict[name] = i
        rate = 0
        for key,value in freq_dict.items():
            rate += value/len(doc)
        rate_percent = rate*100
    
        ADV_freq_rate.append(rate_percent)
    return ADV_freq_rate

#Honores Statistic
def honores(corpus):
    HS = []
    for i in range(0,len(corpus)):
        doc = nlp(corpus[i])
        words = [token.lemma_ for token in doc if token.is_stop != True and token.is_punct != True]

        word_freq = Counter(words)
        V = len(set(words)) #Unique Words
        V_1 = 0    
        N = len(doc)
        for name,i in word_freq.items():#words of frquency one
            if i==1:
                V_1 += 1
        var_1 = V_1/V
        var_2 = (1 - var_1)
        var_3 = math.log(N)
        H_S = var_3/var_2
        HS.append(H_S)
    return HS

#brunets measure
def brunet(corpus):
    BM = []
    for i in range(0,len(corpus)):
        doc = nlp(corpus[i])
        words = [token.lemma_ for token in doc if token.is_stop != True and token.is_punct != True]

        word_freq = Counter(words)
        V = len(set(words)) #Unique Words    
        N = len(doc)
        A = -0.165
        BM_1 = V**A
        B_M = N**BM_1
        BM.append(B_M)
    return BM

#type-token ratio
def ttr(corpus):
    TTR = []
    for i in range(0,len(corpus)):
        doc = nlp(corpus[i])
        words = [token.lemma_ for token in doc if token.is_stop != True and token.is_punct != True]

        word_freq = Counter(words)
        V = len(set(words)) #Unique Words    
        N = len(doc)
        TT_R = (V/N)*100
        TTR.append(TT_R)
    return TTR

#sichel measure
def sichel(corpus):
    SICH = []
    for i in range(0,len(corpus)):
        doc = nlp(corpus[i])
        words = [token.lemma_ for token in doc if token.is_stop != True and token.is_punct != True]

        word_freq = Counter(words)
        V = len(set(words)) #Unique Words
        V_1 = 0    
        N = len(doc)
        for name,i in word_freq.items():#words of frquency one
            if i==2:
                V_1 += 1
        var_1 = V_1/V
        SICH.append(var_1*100)
    return SICH

#Readability measure - Automated readability index
def ari(corpus):
    a_r_i = []
    for i in corpus:
        doc = nlp(i)
        #text = i.text()
        ts = TextStats(doc)
        a_r_i.append(round(ts.automated_readability_index))
    return a_r_i

#Flesch Kincaide Grade 
def F_K_R(corpus):
    fkr = []
    for i in corpus:
        doc = nlp(i)
        #text = i.text()
        ts = TextStats(doc)
        fkr.append(round(ts.flesch_kincaid_grade_level))
    return fkr

def avg_sent_length(corpus):
    lens = []   
    for i in range(0,len(corpus)):
        doc = nlp(corpus[i])
        words = [token.lemma_ for token in doc if token.is_punct != True]
        lens.append(round(len(words)/len(list(doc.sents))))
    print(len(list(doc.sents)))
    mean_len = np.mean(lens)
    return mean_len

def main():
    # Read the data
    regan_AD_path = glob.glob(os.path.join(os.getcwd(), "regan_AD", "*.txt")) #change path
    regan_CT_path = glob.glob(os.path.join(os.getcwd(), "regan_CT", "*.txt")) #change path

    AD_corpus = []
    CT_corpus = []
    un_AD_corpus = []
    un_CT_corpus = []

    for file in regan_AD_path:
        with open(file) as f_input:
            AD_corpus.append(f_input.read())

    for file in regan_CT_path:
        with open(file) as f_input:
            CT_corpus.append(f_input.read())

    un_AD_corpus = AD_corpus.copy() #copy of unprocessed corpus
    un_CT_corpus = CT_corpus.copy() #copy of unprocessed corpus

    for i in range(0,len(AD_corpus)):
        AD_corpus[i] = pre_process(AD_corpus[i])
    for i in range (0,len(CT_corpus)):
        CT_corpus[i] = pre_process(CT_corpus[i])

    AD_df = pd.DataFrame()
    CT_df = pd.DataFrame()  
    AD_df['ADV'] = Adverbs(AD_corpus)
    CT_df['ADV'] = Adverbs(CT_corpus)
    AD_df['NOUN'] = Nouns(AD_corpus)
    CT_df['NOUN'] = Nouns(CT_corpus)
    AD_df['VERB'] = Verbs(AD_corpus)
    CT_df['VERB'] = Verbs(CT_corpus)
    AD_df['PRON'] = Pron(AD_corpus)
    CT_df['PRON'] = Pron(CT_corpus)
    #length of each document
    CT_doc_len = []
    for i in CT_corpus:
        doc = nlp(i)
        CT_doc_len.append(len(doc))
        
    AD_doc_len = []
    for i in AD_corpus:
        doc = nlp(i)
        AD_doc_len.append(len(doc))

    # calculating pronoun ratio in each document
    AD_df['PRP_NOUN_Ratio'] = (AD_df['PRON']/AD_df['NOUN'])*100  
    CT_df['PRP-NOUN_Ratio'] = (CT_df['PRON']/CT_df['NOUN'])*100

    #Calculating noun ratio and pronoun ratio
    AD_df['NOUN_Ratio'] = (AD_df['NOUN']/AD_doc_len)*100
    AD_df['PRP_Ratio'] = (AD_df['PRON']/AD_doc_len)*100
    CT_df['NOUN_Ratio'] = (CT_df['NOUN']/CT_doc_len)*100
    CT_df['PRP_Ratio'] = (CT_df['PRON']/CT_doc_len)*100
    AD_df['word_freq_rate'] = word_freq_rate(AD_corpus)
    CT_df['word_freq_rate'] = word_freq_rate(CT_corpus)
    AD_df['word_freq_rate_Nsw'] = word_freq_rate_Nsw(AD_corpus)
    CT_df['word_freq_rate_Nsw'] = word_freq_rate_Nsw(CT_corpus)
    AD_df['verb_freq_rate'] = verb_freq_rate(AD_corpus)
    CT_df['verb_freq_rate'] = verb_freq_rate(CT_corpus)
    AD_df['Noun_rate'] = noun_freq_rate(AD_corpus)
    CT_df['Noun_rate'] = noun_freq_rate(CT_corpus)
    AD_df['PRON_rate'] = pron_freq_rate(AD_corpus)
    CT_df['PRON_rate'] = pron_freq_rate(CT_corpus)
    AD_df['ADV_rate'] = adv_freq_rate(AD_corpus)
    CT_df['ADV_rate'] = adv_freq_rate(CT_corpus)
    AD_df['H_S'] = honores(AD_corpus)
    CT_df['H_S'] = honores(CT_corpus)
    AD_df['BM'] = brunet(AD_corpus)
    CT_df['BM'] = brunet(CT_corpus)
    AD_df['TTR'] = ttr(AD_corpus)
    CT_df['TTR'] = ttr(CT_corpus)
    AD_df['SICH'] = sichel(AD_corpus)
    CT_df['SICH'] = sichel(CT_corpus)
    AD_df['A_R_I'] = ari(un_AD_corpus)
    CT_df['A_R_I'] = ari(un_CT_corpus)
    AD_df['F_K_R'] = F_K_R(AD_corpus)
    CT_df['F_K_R'] = F_K_R(CT_corpus)
    AD_df.to_csv('AD_features_2.csv',index=False)
    CT_df.to_csv('CT_features_2.csv', index=False)

    Dates_df = pd.read_csv('regan_data.csv')
    Dates_df=Dates_df.drop(Dates_df.index[[5,8,11,17]])

    AD_dates = []
    for i in Dates_df['Dates'][42:104]:
        AD_dates.append(i)

    CT_dates = []
    for i in Dates_df['Dates'][0:42]:
        CT_dates.append(i)

    AD_converted_dates = list(map(datetime.datetime.strptime, AD_dates, len(AD_dates)*['%m/%d/%Y']))
    #x_axis = converted_dates
    formatter = dates.DateFormatter('%m/%d/%Y')
    CT_converted_dates = list(map(datetime.datetime.strptime, CT_dates, len(CT_dates)*['%m/%d/%Y']))

    AD_df.insert(loc=0, column='Dates', value=AD_converted_dates)
    CT_df.insert(loc=0, column='Dates', value=CT_converted_dates)
    CT_df.insert(loc=1, column='Year', value= pd.DatetimeIndex(CT_df['Dates']).year)
    AD_df.insert(loc=1, column='Year', value= pd.DatetimeIndex(AD_df['Dates']).year)

if __name__ == '__main__':
    main()