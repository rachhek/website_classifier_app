from operator import itemgetter
import pickle
import re
from bs4 import BeautifulSoup
import bs4 as bs4
from urllib.parse import urlparse
import requests
from collections import Counter
import pandas as pd
import os
import sys
from clint.textui import puts, colored, indent
import spacy as sp
from collections import Counter
import numpy as np
sp.prefer_gpu()


class ScraperTool:
    def visit_url(self, website_url):
        '''
        Visit URL. Download the Content. Initialize the beautifulsoup object. Call parsing methods. Return Series object.
        '''
        if not self.validate_url(website_url):
            print(colored.red("Invalid Url: {}".format(colored.blue(website_url))))
            sys.exit(0)

        print(colored.yellow("Scraping URL: {}".format(website_url)))

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36'}
        content = requests.get(
            website_url, headers=headers, timeout=10).content

        # lxml is apparently faster than other settings.
        soup = BeautifulSoup(content, "lxml")
        result = {
            "website_url": website_url,
            "website_name": self.get_website_name(website_url),
            "title_tag_content": self.get_html_title_tag(soup),
            "meta_tag_content": self.get_html_meta_tags(soup),
            "headings_content": self.get_html_heading_tags(soup),
            "html_text_content": self.get_text_content(soup)
        }
        # get_tag_count returns a dictionary that has a key value pair of tag and its frequency count.
        # The tags are not always the same so that is why we update the dictionary with a separate update command.
        result.update(self.get_tag_count(soup))

        # Convert to pandas Series object and return
        return pd.Series(result)

    def validate_url(self, url):
        import re
        regex = re.compile(
            r'^(?:http|ftp)s?://'  # http:// or https://
            # domain...
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return re.match(regex, url)

    def get_website_name(self, website_url):
        '''
        Example: returns "google" from "www.google.com"
        '''
        return "".join(urlparse(website_url).netloc.split(".")[-2])

    def get_html_title_tag(self, soup):
        '''Return the text content of <title> tag from a webpage'''
        return '. '.join(soup.title.contents)

    def get_html_meta_tags(self, soup):
        '''Returns the text content of <meta> tags related to keywords and description from a webpage'''
        tags = soup.find_all(lambda tag: (tag.name == "meta") & (
            tag.has_attr('name') & (tag.has_attr('content'))))
        content = [str(tag["content"])
                   for tag in tags if tag["name"] in ['keywords', 'description']]
        return ' '.join(content)

    def get_html_heading_tags(self, soup):
        '''returns the text content of heading tags. The assumption is that headings might contain relatively important text.'''
        tags = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
        content = [" ".join(tag.stripped_strings) for tag in tags]
        return ' '.join(content)

    def get_text_content(self, soup):
        '''returns the text content of the whole page with some exception to tags. See tags_to_ignore.'''
        tags_to_ignore = ['style', 'script', 'head', 'title', 'meta',
                          '[document]', "h1", "h2", "h3", "h4", "h5", "h6", "noscript"]
        tags = soup.find_all(text=True)
        result = []
        for tag in tags:
            stripped_tag = tag.strip()
            if tag.parent.name not in tags_to_ignore\
                    and isinstance(tag, bs4.element.Comment) == False\
                    and not stripped_tag.isnumeric()\
                    and len(stripped_tag) > 0:
                result.append(stripped_tag)
        return ' '.join(result)

    def get_tag_count(self, soup):
        '''returns a dictionary with the frequency of tag for every unique tag found in the page.'''
        tags = soup.find_all()
        return dict(Counter(["tag_count_" + tag.name for tag in tags]))


class TextPreProcessingTool:
    def __init__(self):
        try:
            self.nlp = sp.load("en_core_web_sm")
        except Exception as e:
            print(colored.red(str(e)))

    def clean_text(self, doc):
        '''
        Clean the document. Remove pronouns, stopwords, lemmatize the words and lowercase them. Take only the first 500 words.
        '''
        print(colored.yellow("Cleaning the scraped text"))

        doc = self.nlp(doc)
        tokens = []
        exclusion_list = ["nan"]
        for token in doc:
            if token.is_stop or token.is_punct or token.text.isnumeric() or (token.text.isalnum() == False) or token.text in exclusion_list:
                continue
            token = str(token.lemma_.lower().strip())
            tokens.append(token)
        return " ".join(tokens[:500])


class WebsiteClassifierTool:
    def __init__(self):
        try:
            # Read the pre-trained tfidf vectorizer and logistric regression model
            tfidf_vectorizer_file = os.path.join(
                sys.path[0], "model\\tfidf_vectorizer.pk")
            log_reg_model_file = os.path.join(
                sys.path[0], "model\\logistic_regression_78_acc.sav")

            if os.path.exists(tfidf_vectorizer_file):
                self.tfidf_tokenizer = pd.read_pickle(tfidf_vectorizer_file)
                print(colored.blue(
                    "loaded pre-computed tfidf vectorizer object from {}".format(tfidf_vectorizer_file)))
            else:
                print(colored.red(
                    "Can't find file at {}".format(tfidf_vectorizer_file)))

            if os.path.exists(log_reg_model_file):
                self.log_reg_model = pd.read_pickle(log_reg_model_file)
                print(colored.blue(
                    "loaded pre-trained logistic regression model object from {}".format(log_reg_model_file)))
            else:
                print(colored.red(
                    "Can't find file at {}".format(log_reg_model_file)))

        except Exception as e:
            print(colored.red(str(e)))

    def predict(self, website_url):
        '''
        Step 1: Scrape the website
        Step 2: Clean and preprocess the scraped text content from the website
        Step 3: Load the tf-idf vectorizer and pre-trained logistic regression model
        Step 4: Classify the text.
        '''
        try:
             # Scrape the website.
            st = ScraperTool()
            res = st.visit_url(website_url)

            # Combine these columns from the result of the scraper tool to make a single text content.
            columns_to_combine = ["title_tag_content",
                                  "meta_tag_content", "headings_content", "html_text_content"]
            website_text_content = " ".join(
                res[columns_to_combine].astype(str))
            print("\n Sample scraped text: {}... \n".format(
                colored.yellow(website_text_content[:200])))

            # Clean the text content.
            tpt = TextPreProcessingTool()
            website_text_content = tpt.clean_text(website_text_content)

            print("\n Sample cleaned text: {}...\n".format(
                colored.yellow(website_text_content[:200])))

            print(colored.yellow("Converting the text to tfidf vector."))
            # convert the text into tf-idf vectors by using a tfidfvectorizer that was previously trained
            vectorized_text = self.tfidf_tokenizer.transform(
                [website_text_content])
            print("\n Words with highest TF-IDF values: {}\n".format(
                colored.yellow(
                    self.get_top_tf_idf_words(
                        vectorized_text)
                )))

            print(colored.yellow(
                "Classifying the text using logistic regression model. \n"))
            # use the vectorized text to predict using the trained logistic regression model
            result = self.log_reg_model.predict(vectorized_text)

            # getting the probability distribution for all 11 classes.
            class_prob = np.around(
                self.log_reg_model.predict_proba(vectorized_text)*100, 3)
            class_prob = {class_name: prob for class_name,
                          prob in zip(self.log_reg_model.classes_, class_prob[0])}
            if result:
                return {
                    "predicted_class": result[0],
                    "prob_distribution": sorted(class_prob.items(), key=itemgetter(1), reverse=True)
                }
        except Exception as e:
            print(colored.red(str(e)))

    def get_top_tf_idf_words(self, vectorized_text, top_n=10):
        '''
        Get the words with the highest tf-idf values.
        '''
        feature_names = np.array(self.tfidf_tokenizer.get_feature_names())
        sorted_vectors = np.argsort(vectorized_text.data)[:-(top_n):-1]
        return feature_names[vectorized_text.indices[sorted_vectors]]
