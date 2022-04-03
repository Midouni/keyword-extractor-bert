from cProfile import label
import pandas
import os
from keywords import Keywords
data={"text":["Pizza is a dish of Italian origin consisting of a usually round, flat base of leavened wheat-based dough topped with tomatoes, cheese, and often various other ingredients, which is then baked at a high temperature, traditionally in a wood-fired oven","A hamburger is a food consisting of fillings —usually a patty of ground meat, typically beef—placed inside a sliced bun or bread roll."]}
df=pandas.DataFrame(data)
df.to_csv(os.path.dirname(__file__)+"/data.csv")
keys=Keywords()
#test1
keys.extract_keywords_csv_to_csv(os.path.dirname(__file__)+"/data.csv",os.path.dirname(__file__)+"/test1.csv",label_text="text")
#test2
l=["Pizza is a dish of Italian origin consisting of a usually round, flat base of leavened wheat-based dough topped with tomatoes, cheese, and often various other ingredients, which is then baked at a high temperature, traditionally in a wood-fired oven","A hamburger is a food consisting of fillings —usually a patty of ground meat, typically beef—placed inside a sliced bun or bread roll."]
keys.extract_keywords_list_to_csv(l,os.path.dirname(__file__)+"/test2.csv")
#test3
res1=keys.extract_keywords(l)
#test4
res2=keys.extract_keywords_csv_to_list(os.path.dirname(__file__)+"/data.csv",label_text="text")

print("res1")
print(res1)
print("res2")
print(res2)
