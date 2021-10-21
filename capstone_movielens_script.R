##################################################
##################################################
#                                                #
#  HarvardX PH125.9x Movielens Capstone Project  #
#                                                #
#  Author: C. Heather                            #  
#                                                #
##################################################
##################################################


library(tidyverse)
library(caret)
library(lubridate)
library(data.table)
library(recosystem)


##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################


# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip



# Download, clean and combine data sets into movielens data frame


dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


#### Additional data wrangling


# Create new column with release year in edx

edx<- edx %>%
  mutate(release_year = str_extract(title, "(?<=\\()[:digit:]{4}(?=\\))"))

# create separate datafram with split genres into multiple rows, where necessary

edx_genres <- separate_rows(edx, genres, sep = "\\|") %>%
  select(rating, genres, userId, movieId)



##############################################
## Exploratory Data analysis                ##
##############################################

# Number of ratings
nrow(edx)


# Number of users
print("Number of users")
edx %>% group_by(userId) %>% summarise(n()) %>% nrow()

# Number of movies
print("Number of movies")
edx %>% group_by(movieId) %>% summarise(n()) %>% nrow()


# Number of ratings per user chart
edx %>% 
  group_by(userId) %>%
  summarise(n = n()) %>%
  ggplot(aes(n)) +
  geom_bar(colour = "black") +
  labs(x = "Number of ratings", y = "Count of users")


# mean rating
mu <- mean(edx$rating)

mu


# Average rating by user chart
edx %>% group_by(userId) %>%
  summarise(avg_rating = mean(rating)) %>%
  ggplot(aes(reorder(userId, avg_rating), avg_rating)) +
  geom_point(size = 0.01) +
  labs(
    title = "Average rating by userId",
    x = "userId",
    y = "Average Rating")+
  theme(
    axis.ticks.x = element_blank(),
    axis.text.x=element_blank())


# Correlation of number of rating per user and rating chart
edx %>% group_by(userId) %>%
  summarise(avg_rating = mean(rating), num_ratings = n()) %>%
  ggplot(aes(num_ratings, avg_rating)) +
  geom_point() +
  geom_smooth(method = "lm", color = "red")+
  labs(
    title = "Average rating by number of ratings for each user",
    x = "Ratings",
    y = "Average Rating")


# Number of ratings per movie chart
edx %>% group_by(movieId) %>%
  summarise(n = n()) %>%
  ggplot(aes(n)) +
  geom_bar(fill = "black") +
  scale_y_continuous(limits = c(0,40))+
  labs(x = "Number of ratings", y = "Count of movies")


# Average ratings by movie chart
edx %>% group_by(movieId) %>%
  summarise(avg_rating = mean(rating), num_ratings = n()) %>%
  ggplot(aes(reorder(movieId, avg_rating), avg_rating)) +
  geom_point(size = 0.01) +
  labs(
    title = "Average rating by movieId",
    x = "movieId",
    y = "Average Rating")+
  theme(
    axis.ticks.x = element_blank(),
    axis.text.x=element_blank()) 


# Correlation of number of ratings per movie and rating chart
edx %>% group_by(movieId) %>%
  summarise(avg_rating = mean(rating), num_ratings = n()) %>%
  ggplot(aes(num_ratings, avg_rating)) +
  geom_point() +
  geom_smooth(method = "lm", color = "red")+
  labs(
    title = "Average rating by number of ratings for each movie",
    x = "Ratings",
    y = "Average Rating")


# Number of reviews by release year chart
ggplot(edx, aes(as.numeric(release_year))) +
  geom_bar() +
  labs(x = "release year", y = "ratings")


# Correlation of rating with release year chart
edx %>% group_by(release_year) %>%
  summarise(avg_rating = mean(rating)) %>%
  ggplot(aes(as.numeric(release_year), avg_rating)) +
  geom_point() + 
  geom_smooth(method = "lm", color = "red")+
  labs(
    title = "Average rating by Release year",
    x = "Release Year",
    y = "Average Rating") +
  ylim(1,5)


# Genre rating mean and spread chart
edx_genres %>% group_by(movieId) %>%
  summarise(genres, avg_rating = mean(rating)) %>%
  ggplot(aes(genres, avg_rating)) +
  geom_boxplot() + 
  labs(
    title = "Average rating by genre",
    x = "Genre",
    y = "Average Rating") +
  theme(axis.text.x = element_text(angle = 45, hjust=1))

rm("edx_genres") # remove edx_genres from memory



############################################
##           Model development            ##
############################################


# RMSE function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


#####################################################
## User and Movie bias model from course materials ##
##          Included for reference                 ##
#####################################################

#Partition edx into test (20%) and train (80%) sets
set.seed(755)
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, 
                                  list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]

#ensure userId and movieId from train_set are in test_set
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")


# Select range for tuning of regularisation factor
lambdas <- seq(1, 5, 0.1)


# Create tuning function returning rmses
rmses <- sapply(lambdas, function(l){
  
  mu <- mean(train_set$rating)
  
  b_i <- train_set %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- train_set %>%
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_i)/(n()+l))
  
  predicted_ratings <-
    test_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i) %>%
    pull(pred)
  
  
  return(RMSE(predicted_ratings, test_set$rating))
})
qplot(lambdas, rmses) 

# select optimum regularisation parameter

l <- lambdas[which.min(rmses)]

# best RMSE with this method

rmse_1 <- min(rmses)

rmse_1

rm(test_index, test_set, train_set)


###################################################
## Matrix factorisation using recosystem package ##
###################################################

### Data preparation

# create data objects using recosystem::data_memory() function

train_data <- data_memory(user_index = edx$userId, 
                          item_index = edx$movieId, 
                          rating = edx$rating)

test_data <- data_memory(user_index = validation$userId, 
                         item_index = validation$movieId, 
                         rating = validation$rating)

# Memory cleanup
rm(edx) 



### Setting tuning parameters and tuning model

# Specify tuning parameters
parameters <- list(dim = c(10, 20, 30),  # number of latent factors
                   lrate = c(0.1, 0.2), # learning rate - the step size in gradient descent
                   costp_l1 = 0, # L1 regularisation cost for user factors. Set to 0. L2 regularisation used.
                   costq_l1 = 0, # L1 regularisation cost for movie factors. Set to 0. L2 regularisation used.
                   costp_l2 = c(0.01, 0.1), # L2 regularisation cost for user factors.
                   costq_l2 = c(0.01, 0.1), # L2 regularisation cost for movie factors.
                   niter = 10, # number of iterations
                   loss = "l2") # loss function. Specifying squared error ("l2")


# create empty recommender object
recommender <- Reco()

# Set training options using tuning parameters
opts = recommender$tune(train_data, opts = parameters)

# Optimal tuning parameters
opts$min



### Train model

# train the model using best options from tuning data "opts$min"
recommender$train(train_data, opts = c(opts$min, niter = 20))



### Determine RMSE using  predictions from recommender$predict method and validation$rating vector

# Create vector of predictions for each user
pred_rvec <- recommender$predict(test_data, out_memory())

# Run RMSE function
rmse_2 <- RMSE(validation$rating, pred_rvec)

rmse_2
