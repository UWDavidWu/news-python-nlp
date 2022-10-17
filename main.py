import json
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA


titleList = []
tokenizeTitleList = []
intensityList = []

def main():
    getTitleList()
    tokenizeTitle()
    print(tokenizeTitleList)
    getFrequency()
    getIntensity()
    print(intensityList)

    


def getTitleList():
    global titleList
    with open('response.json') as f:
        data = json.load(f)
    titleList = [x['title'].split("-")[0] for x in data['articles']]
    # print(titleList)

def tokenizeTitle():
    global tokenizeTitleList
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = stopwords.words('english')

    tokenizeTitleList = []
    for title in titleList:
        toks = tokenizer.tokenize(title)
        toks = [t.lower() for t in toks if t.lower() not in stop_words]
        tokenizeTitleList.extend(toks)

def getIntensity():
    global titleList 
    global intensityList

    sia = SIA()
    results = []

    for line in titleList:
        pol_score = sia.polarity_scores(line)
        pol_score['title'] = line
        if pol_score['compound'] > 0.2:
            pol_score['label'] = 1
        elif pol_score['compound'] < -0.2:
            pol_score['label'] = -1
        results.append(pol_score)
    print(results)

    intensityList = [x['compound'] for x in results]
    for index, x in enumerate(intensityList):
        if x > 0.2:
            intensityList[index]= 1
        elif x < -0.2:
            intensityList[index]= -1
        else:
            intensityList[index]= 0


def getFrequency():
    global tokenizeTitleList
    pos_freq = nltk.FreqDist(tokenizeTitleList)

    print(pos_freq.most_common(5)    )
    
if __name__ == '__main__':
    main()