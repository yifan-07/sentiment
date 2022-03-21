# sentiments

## Mathod 
### TF-IDF and Categorical algorithm

### Count and Categorical algorithm

###   BERT 
* 架構
1. bert-base：12層，768維，12 self-attention，110M參數。
2. bert-large：24層，1024維，16 self-attention，340M參數。

* 分析流程
1. 原始數據： 資料預處理，以tsv檔儲存。
2. BERT相容格式
encoder
![bert_encoder]('https://leemeng.tw/images/bert/practical_bert_encoding_for_pytorch.jpg)
* token embeddings：對應前面的wordpiece
* segment embeddings：句子位置
* positional embeddings：未置編碼
* tokens_tensor：代表識別每個 token 的索引值，用 tokenizer 轉換即可
* segments_tensor：用來識別句子界限。第一句為 0，第二句則為 1。另外注意句子間的 [SEP] 為 0
* masks_tensor：用來界定自注意力機制範圍。1 讓 BERT 關注該位置，0 則代表是 padding 不需關注
3.加入下游任務
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
4.訓練下游任務
5.預測

### LSTM

## Python
* bert_based: bert 基本概念 + 真假新聞辨識(base on pytorch)
data：WSDM - Fake News Classification
* bert_sentiment： 電影評論(base on tensorflow)
data：aclImdb_v1
* LSTM_sentiment


### source 
BERT
* 進擊的 BERT：NLP 界的巨人之力與遷移學
https://leemeng.tw/attack_on_bert_transfer_learning_in_nlp.html
* Sentiment Analysis in 10 Minutes with BERT and TensorFlow
https://towardsdatascience.com/sentiment-analysis-in-10-minutes-with-bert-and-hugging-face-294e8a04b671