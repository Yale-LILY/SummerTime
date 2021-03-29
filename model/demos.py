# To be converted to documentation for first release

from lexrank_model import LexRank
from pegasus_model import pegasus
from path import Path 
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch


def pegasus_demo(): 
    src_text = [
            """ PG&E stated it scheduled the blackouts in response to forecasts for high winds amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow.""", 
            """ The Pats won the Super Bowl."""
    ]

    model = pegasus()

    summaries = model.summarize(src_text)

    print(summaries)


def lexrank_demo(): 
    corpus = []
    corpus_dir = Path('demo_data')


    for file_path in corpus_dir.files('*.txt'): 
        with file_path.open(mode='rt', encoding='utf-8') as fp: 
            corpus.append(fp.readlines())


    model = LexRank(corpus)

    documents = [[
    'One of David Cameron\'s closest friends and Conservative allies, '
    'George Osborne rose rapidly after becoming MP for Tatton in 2001.',

    'Michael Howard promoted him from shadow chief secretary to the '
    'Treasury to shadow chancellor in May 2005, at the age of 34.',

    'Mr Osborne took a key role in the election campaign and has been at '
    'the forefront of the debate on how to deal with the recession and '
    'the UK\'s spending deficit.',

    'Even before Mr Cameron became leader the two were being likened to '
    'Labour\'s Blair/Brown duo. The two have emulated them by becoming '
    'prime minister and chancellor, but will want to avoid the spats.',

    'Before entering Parliament, he was a special adviser in the '
    'agriculture department when the Tories were in government and later '
    'served as political secretary to William Hague.',

    'The BBC understands that as chancellor, Mr Osborne, along with the '
    'Treasury will retain responsibility for overseeing banks and '
    'financial regulation.',

    'Mr Osborne said the coalition government was planning to change the '
    'tax system \"to make it fairer for people on low and middle '
    'incomes\", and undertake \"long-term structural reform\" of the '
    'banking sector, education and the welfare state.',
    ]]

    model.show_capability()

    summaries = model.summarize(documents)

    print("Document: \n", documents[0])
    print("Summary: \n", summaries[0])

