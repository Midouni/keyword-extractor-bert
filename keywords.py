from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import pandas
from typing import Union,Tuple,List
from sklearn.feature_extraction.text import CountVectorizer



#model path
import os
PATH=os.path.dirname(__file__)+"/keyword_extractor_model"

#Initialization model
kw_model = KeyBERT(SentenceTransformer(PATH))


class Keywords:

    def __init__(self,
        candidates: List[str] = None,
        keyphrase_ngram_range: Tuple[int, int] = (1, 1),
        stop_words: Union[str, List[str]] = "english",
        top_n: int = 5,
        min_df: int = 1,
        use_maxsum: bool = False,
        use_mmr: bool = False,
        diversity: float = 0.5,
        nr_candidates: int = 20,
        vectorizer: CountVectorizer = None,
        highlight: bool = False,
        seed_keywords: List[str] = None,):

        self.candidates=candidates
        self.keyphrase_ngram_range=keyphrase_ngram_range
        self.stop_words=stop_words
        self.top_n=top_n
        self.min_df=min_df
        self.use_maxsum=use_maxsum
        self.use_mmr=use_mmr
        self.diversity=diversity
        self.nr_candidates=nr_candidates
        self.vectorizer=vectorizer
        self.highlight=highlight
        self.seed_keywords=seed_keywords


    def __to_dataframe(self,text:List,keyword_list:List,CSV_path):
        data={}
        data["text"]=[]
        for i in text:
            data["text"].append(i)

        for j in range(0,self.top_n):
            data["keyword"+str(j+1)]=[]
            data["similarity"+str(j+1)]=[]

        
        for i in range(len(keyword_list)):
            for j in range(0,self.top_n):
                key=keyword_list[i][j]
                data["keyword"+str(j+1)].append(key[0])
                data["similarity"+str(j+1)].append(key[1])

        df = pandas.DataFrame(data)
        df.to_csv(CSV_path)

    

    def extract_keywords(self,doc:List):
        """return list of tuple (keyword,similarity) """
        return kw_model.extract_keywords(
        docs=doc,
        candidates=self.candidates,
        keyphrase_ngram_range=self.keyphrase_ngram_range,
        stop_words=self.stop_words,
        top_n=self.top_n,
        min_df=self.min_df,
        use_maxsum=self.use_maxsum,
        use_mmr=self.use_mmr,
        diversity=self.diversity,
        nr_candidates=self.nr_candidates,
        vectorizer=self.vectorizer,
        highlight=self.highlight,
        seed_keywords=self.seed_keywords
        )

    def extract_keywords_list_to_csv(self,doc:list,csv_path):
        """
        text to csv
        """
        keys=self.extract_keywords(doc)
        self.__to_dataframe(doc,keys,csv_path)


    def extract_keywords_csv_to_csv(self,path_current_csv,path_new_csv,label_text="post"):
        """take csv path """
        df=pandas.read_csv(path_current_csv)
        data=list(df[label_text].values)
        self.extract_keywords_list_to_csv(data,path_new_csv)

    def extract_keywords_csv_to_list(self,path_current_csv,label_text="post"):
        df=pandas.read_csv(path_current_csv)
        data=list(df[label_text].values)
        return self.extract_keywords(data)



