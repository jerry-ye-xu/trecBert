#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 15:08:23 2019

@author: ryb003
"""

import re
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer


#_metamap_re = re.compile("^\s+\d+\s+(\w+|\s+|-)+\s+\((.+)\)\s+\[(.*)\]")
_clean_brackets_re = re.compile(r"\[\*\*(\w+|\s+|\-|\(|\))+\*\*\]")
_numbers_re = re.compile('[\d]+')


def replaceNumbers(str):
    if str is None:
        return ''
    return _numbers_re.sub("", str)





def getWords(str):

    if str is not None:
        return ' '.join(re.findall(r"[\w]+", str))
    else:
        return ''

##Change this? So we get terms. Or add a tokeniser on the entire string.
def remove_stopwords_punctuation(s):
    stop_words = set(stopwords.words('english'))
    stop_words.update(
        ['.', ',', '=', '-', '+', '~', '/', '\\', '<', '>', '`', '_', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']',
         '{', '}', '*', '%', '@', '^', '#'])

    wordpunct_tokenize = WordPunctTokenizer()
    #print s
    s = getWords(s)
    #print stop_words
    list_of_words = [i.lower() for i in wordpunct_tokenize.tokenize(s) if (i.lower() not in stop_words and len(i)>=2)]
    return ' '.join(list_of_words)#.encode('ascii', 'xmlcharrefreplace')


def replacement_fn(m):
    age = int(m)
    if age > 60:
        return 'adult elderly'
    elif age > 18:
        return 'adult'
    else:
        return 'child'  # not sure about this teminology

def replaceShorthandAndClean(string):
    if string==None:
        return ''
    shorthand = {
        "M": " male ",
        "F": " female ",
        "RR": " respiratory rate ",
        "US": " ultrasound ",
        "HR": " heart rate ",
        "hx": " medical history ",
        "hotn": "hotn hypotension ",
        "htn": " htn hypertension ",
        "pt": " patient ",
        "pmh": " past medical history ",
        "pmhx": " patient medical history ",
        "prn": " when necessary "
        # r"\~" : " approximately ",
        # "h\/o" : " history of ",
        # r"y\/o" : "",
        # r"\w/" : " with ",
        # r"\[\*\*(\w+|\s+|\-|\(|\))+\*\*\]": " ANONYMOUSTIGER ",
    }

    string = string.lower()

    pattern = re.compile(r'\b(' + '|'.join(shorthand.keys()) + r')\b')
    result = pattern.sub(lambda x: shorthand[x.group()], string)
    re.purge()

    # fix age related phrases
    agep = re.compile(
        r'\b([aA]|[aA]n)*\s*(\d+)\s*(-?|\s*)*([Mm]\b|[Ff]\b|[mM]ale\b|[fF]emale\b|y\/o\b|yo\b|y.o\b|y.o.\b|year-old\b|year old\b|years-old\b|years old\b)')  # match 67 y.o.or 76-year-old or A 40-year-old

    matched = agep.search(result)
    # print str
    #rint(agep.findall(str))
    if matched:
        age = int(matched.group(2))
        if matched.group(4) == 'F' or matched.group(4) == 'f':
            temp = agep.sub(" " + replacement_fn(age) + " Female ", result)
        elif matched.group(4) == 'M' or matched.group(4) == 'm':
            temp = agep.sub(" " + replacement_fn(age) + " Male ", result)
        else:
            temp = agep.sub(" " + replacement_fn(age), result)
        result = re.sub(agep, " ", temp)

    # remove annotations between [** and **]
    result = _clean_brackets_re.sub(" ", result)
    result = result.replace("w/", " with ").replace("~", " approximately ").replace("h/o", " history of ").replace(
        "y/o", " year old ").replace("  ", ", ").replace("B/P", " blood pressure ")
    result = replaceNumbers(result)
    result = remove_stopwords_punctuation(result)
    return result
