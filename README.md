# sentiments

## Mathod 
### TF-IDF and Categorical algorithm
1. 計算TFIDF
TF-IDF分為兩區塊，TF每個詞在文章出現的比例，IDF文章數總合/該自出現的文章篇數後取log
![alt text](https://miro.medium.com/max/1262/1*p33SpAy05KWRqxfp8JYx3w.png)
![alt text](https://miro.medium.com/max/1248/1*IetKUpVCuh2s3-FRfYPhlw.png)
2. 選擇分類方法
### Count and Categorical algorithm
1. 扣除停用字後計算詞頻
2. 選擇分類方法

###   BERT 
* 架構
    1. bert-base：12層，768維，12 self-attention，110M參數。
    2. bert-large：24層，1024維，16 self-attention，340M參數。

* 分析流程  
1. 原始數據： 資料預處理，以tsv檔儲存。
2. BERT相容格式
encoder
![alt text](https://leemeng.tw/images/bert/practical_bert_encoding_for_pytorch.jpg)
    * token embeddings：對應前面的wordpiece
    * segment embeddings：句子位置
    * positional embeddings：未置編碼
    * tokens_tensor：代表識別每個 token 的索引值，用 tokenizer 轉換即可
    * segments_tensor：用來識別句子界限。第一句為 0，第二句則為 1。另外注意句子間的 [SEP] 為 0
    * masks_tensor：用來界定自注意力機制範圍。1 讓 BERT 關注該位置，0 則代表是 padding 不需關注  
3. 加入下游任務
    * 基本款：
        - bertModel
        - bertTokenizer
     * 預訓練階段
          - bertForMaskedLM
          - bertForNextSentencePrediction
          - bertForPreTraining
      * Fine-tuning 階段
          - bertForSequenceClassification
          - bertForTokenClassification
          - bertForQuestionAnswering
          - bertForMultipleChoice
4. 訓練下游任務
5. 預測

### LSTM
1. 預處理，大小寫、詞性標註...
2. 進行tokenize，選擇最大字數、切割方式(空格)
3. 範例架構
    * 第一層：embedding
    * 第二層：Bidirectional LSTM
    * 第三層：Bidirectional LSTM
    * 第四層：Dense
## Python
* NLTK：NLP相關操作，停用字、字根還原、詞性標註  
* sentiment：文字轉為countvector、tfidfvector後，進行分類  
data：kaggle_movie_reviews  
code：https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data  
* sentiment_movie：利用bert進行分類  
data：kaggle_movie_reviews  
* bert_based: bert 基本概念 + 真假新聞辨識(base on pytorch)  
data：WSDM - Fake News Classification  
code：https://leemeng.tw/attack_on_bert_transfer_learning_in_nlp.html  
* bert_sentiment： 電影評論(base on tensorflow)  
data：aclImdb_v1  
code：https://towardsdatascience.com/sentiment-analysis-in-10-minutes-with-bert-and-hugging-face-294e8a04b671  
* LSTM_sentiment
data：aclImdb_v1  
code：https://www.kaggle.com/shyambhu/sentiment-classification-using-lstm  


### source 
TFIDF
https://medium.com/datamixcontent-lab/%E6%96%87%E6%9C%AC%E5%88%86%E6%9E%90%E5%85%A5%E9%96%80-%E6%A6%82%E5%BF%B5%E7%AF%87-%E7%B5%A6%E6%88%91%E4%B8%80%E6%AE%B5%E8%A9%B1-%E6%88%91%E5%91%8A%E8%A8%B4%E4%BD%A0%E9%87%8D%E9%BB%9E%E5%9C%A8%E5%93%AA-%E5%B0%8D%E6%96%87%E6%9C%AC%E9%87%8D%E9%BB%9E%E5%AD%97%E8%A9%9E%E5%8A%A0%E6%AC%8A%E7%9A%84tf-idf%E6%96%B9%E6%B3%95-f6a2790b4991
BERT
* 進擊的 BERT：NLP 界的巨人之力與遷移學
https://leemeng.tw/attack_on_bert_transfer_learning_in_nlp.html
* Sentiment Analysis in 10 Minutes with BERT and TensorFlow
https://towardsdatascience.com/sentiment-analysis-in-10-minutes-with-bert-and-hugging-face-294e8a04b671  
* 模型儲存
https://iter01.com/478666.html
* Dataset
https://iter01.com/524561.html
https://ithelp.ithome.com.tw/articles/10277163