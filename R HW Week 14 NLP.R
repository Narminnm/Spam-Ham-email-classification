library(data.table)
library(tidyverse)
library(text2vec)
library(caTools)
library(glmnet)
library(dplyr)

df <- read_csv("emails.csv")
df <- mutate(df, id=row_number())

df <- df %>% select(id, everything())


# Split data
set.seed(123)
split <- df$spam %>% sample.split(SplitRatio = 0.8)
train <- df %>% subset(split == T)
test <- df %>% subset(split == F)


stop_words <- c("i", "you", "he", "she", "it", "we", "they",
                "me", "him", "her", "them",
                "my", "your", "yours", "his", "our", "ours",
                "myself", "yourself", "himself", "herself", "ourselves",
                "the", "a", "an", "and", "or", "on", "by", "so",
                "from", "about", "to", "for", "of", 
                "that", "this", "is", "are")


it_train <- train$text %>% 
  itoken(preprocessor = tolower, 
         tokenizer = word_tokenizer,
         ids = train$id,
         progressbar = F) 


vocab <- it_train %>% create_vocabulary(stopwords = stop_words, ngram = c(1L, 2L))

pruned_vocab <- vocab %>% 
  prune_vocabulary(term_count_min = 10, 
                   doc_proportion_max = 0.5,
                   doc_proportion_min = 0.001)


vectorizer <- pruned_vocab %>% vocab_vectorizer()


dtm_train <- it_train %>% create_dtm(vectorizer)
dtm_train %>% dim()


glmnet_classifier <- dtm_train %>% 
  cv.glmnet(y = train[['spam']], 
            family = 'binomial',
            type.measure = "auc",
            nfolds = 4,
            thresh = 0.001,
            maxit = 1000)

glmnet_classifier$cvm %>% max() %>% round(3) %>% paste("-> Max AUC")


it_test <- test$text %>% tolower() %>% word_tokenizer() %>% itoken(ids = test$id,progressbar = F)
dtm_test <- it_test %>% create_dtm(vectorizer)
preds <- predict(glmnet_classifier, dtm_test, type = 'response')[,1]
glmnet:::auc(test$spam, preds) %>% round(3)







