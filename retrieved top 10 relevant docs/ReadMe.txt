[Input Output]
1.  Input python main.py --query xxxx --part 1
    Output xxxx 的 TF weighting + Cosine similarity、TF weighting + Eucidean Distance、TF-IDF weighting + Cosine similarity、TF-IDF weighting + Eucidean Distance
2.  Input python main.py --query xxxx --part 2
    Output xxxx 的 Feedback Queries + TF-IDF weighting + Cosine similarity

[Added functions]
* search1(self,searchList) : calculate for the part 1 of hw
* search2(self,searchList) : calculate for the part 2 of hw
* getNewQuery(self,queryVector, doc): calculate a new query by feedback and original query
* doNLTK(self, fileName, originalQuery): open the retrieved document and build the vector for calculating the new query
* n_containing(word, docs_list): calculate this word in how many docs
* calDocSize(dictionary, list): calculate retrieved docs size
* printTemplate(list, label): print template
* check(num,tuple_list): this function can faciliate TA to check how many retrieved docs are correct with TA's top 10 retrieved docs