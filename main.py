from dotenv import load_dotenv
import os
import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String,  create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from constants import countryList, countryExclusion, commonExclusion


Base = declarative_base()
DBSession = scoped_session(sessionmaker())


class News(Base):
    __tablename__ = 'news'
    id = Column(Integer, primary_key=True)
    country = Column(String)
    title = Column(String)
    intensity = Column(Integer)


class Topic(Base):
    __tablename__ = 'topic'
    id = Column(Integer, primary_key=True)
    country = Column(String)
    topic = Column(String)
    occurance = Column(Integer)


# connect to database
def init_SQLAlchemy():
    try:
        engine = create_engine(os.getenv("DATABASE_URL"))
        DBSession.configure(bind=engine)
        Base.metadata.create_all(engine)
        print("Connected to database")
    except Exception as e:
        print(f"Connection failed: {e}")


def main():
    init_SQLAlchemy()
    for country in countryList:
        generateTopic(country)
        generateIntensity(country)


def generateTopic(country):
    print(f'Generating topic for {country}')
    # step 1: get news by country
    countryNewsResult = getNewsByCountry(country)
    print(countryNewsResult)
    # step 2: tokenize title
    tokenizeTitleList = tokenizeTitle(country, countryNewsResult)
    print(tokenizeTitleList)
    # step 3: get frequency
    top10Topics = getFrequency(tokenizeTitleList)
    print(top10Topics)

    # insert into topic table
    for title, occurance in top10Topics:
        topic = Topic(country=country, topic=title, occurance=occurance)
        DBSession.add(topic)
    DBSession.commit()
    print(f"Topic generated for {country}")


def getNewsByCountry(country):
    return DBSession.query(News.id, News.title).filter(News.country == country).all()


def tokenizeTitle(country, countryNewsResult):
    tokenizeTitleList = []
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = stopwords.words('english')

    for title in [title for id, title in countryNewsResult]:
        toks = tokenizer.tokenize(title)
        toks = [t.lower() for t in toks if t.lower() not in stop_words]
        tokenizeTitleList.extend(toks)
    return [t for t in tokenizeTitleList if t not in countryExclusion[country] and t not in commonExclusion]


def getFrequency(tokenizeTitleList, n=10):
    pos_freq = nltk.FreqDist(tokenizeTitleList)
    return pos_freq.most_common(n)


def generateIntensity(country):

    sia = SIA()
    intensityResults = []

    for index, line in getNewsByCountry(country):
        pol_score = sia.polarity_scores(line)
        pol_score['title'] = line
        pol_score['id'] = index
        if pol_score['compound'] > 0.2:
            pol_score['intensity'] = 1
        elif pol_score['compound'] < -0.2:
            pol_score['intensity'] = -1
        else:
            pol_score['intensity'] = 0
        intensityResults.append(pol_score)
    # print(intensityResults)

    print(f"Updating intensity for {country}")
    for result in intensityResults:
        news = DBSession.query(News).filter(News.id == result['id']).first()
        news.intensity = result['intensity']
    DBSession.commit()
    print(f"Intensity generated for {country}")


if __name__ == '__main__':
    load_dotenv()
    main()
