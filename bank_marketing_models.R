# Bank Marketing: Logistic Regression + Random Forest
# Author: <Marcel Obazuaye>
# Description: Data cleaning, missingness handling, MICE imputation, train/test split,
#              scaling, logistic regression, random forest, evaluation.

# ----------------------------
# 0) Packages
# ----------------------------
library(dplyr)
library(ggplot2)
library(naniar)
library(corrplot)
library(vcd)
library(mice)
library(caret)
library(party)

set.seed(42)

# ----------------------------
# 1) Load data
# ----------------------------
# Make sure bank.csv is in your project folder
bank <- read.csv("Bank.csv", stringsAsFactors = TRUE)

# ----------------------------
# 2) Quick exploration
# ----------------------------
dim(bank)
vis_miss(bank)

ggplot(bank, aes(x = y)) + geom_bar(fill = "black") + labs(title = "Distribution of Term Deposit Subscription")

# Numeric correlation
num_vars <- bank %>% select(where(is.numeric))
corrplot(cor(num_vars, use = "complete.obs"), col = colorRampPalette(c("blue", "white", "black"))(200))

# Categorical correlation (Cramer's V)
cat_vars <- bank %>% select(where(is.factor))
cramer_mat <- sapply(cat_vars, function(x) {sapply(cat_vars, function(y) {suppressWarnings(assocstats(table(x, y))$cramer)
})
})
corrplot(cramer_mat, col = colorRampPalette(c("blue", "white", "black"))(200))

# ----------------------------
# 3) Data cleaning / preprocessing
# ----------------------------
bank <- bank %>% distinct()

# Replace "unknown" with NA for selected columns
bank$job[bank$job == "unknown"] <- NA
bank$education[bank$education == "unknown"] <- NA
bank$contact[bank$contact == "unknown"] <- NA
bank$poutcome[bank$poutcome == "unknown"] <- NA

# Handle high-missingness categoricals
bank$contact[is.na(bank$contact)] <- "unknown"
bank$poutcome[is.na(bank$poutcome)] <- "unknown"

# Transform pdays
bank$pdays <- ifelse(bank$pdays == -1, "No_previous_contact", "contacted_before")
bank$pdays <- factor(bank$pdays)

# Ensure target is factor with levels no/yes (common in this dataset)
bank$y <- factor(bank$y, levels = c("no", "yes"))

# Ensure remaining categoricals are factors
factor_cols <- c("job","marital","education","default","housing","loan",
                 "contact","month","poutcome")
bank[factor_cols] <- lapply(bank[factor_cols], factor)

# ----------------------------
# 4) MICE Imputation (job, education)
# ----------------------------
method_vec <- make.method(bank)
method_vec[] <- ""                 # default: no imputation
method_vec["job"] <- "polyreg"
method_vec["education"] <- "polyreg"

pred_mat <- make.predictorMatrix(bank)
pred_mat[,] <- 0
pred_mat["job", ] <- 1
pred_mat["education", ] <- 1
pred_mat["job","job"] <- 0
pred_mat["education","education"] <- 0

imp <- mice(bank, m = 5, method = method_vec, predictorMatrix = pred_mat, seed = 123)
bank <- complete(imp, 1)

# ----------------------------
# 5) Train/test split
# ----------------------------
idx <- createDataPartition(bank$y, p = 0.8, list = FALSE)
train <- bank[idx, ]
test  <- bank[-idx, ]

# ----------------------------
# 6) Scale numeric features (fit on train, apply to test)
# ----------------------------
num_cols <- names(train)[sapply(train, is.numeric)]
pp <- preProcess(train[, num_cols], method = c("center", "scale"))

train[, num_cols] <- predict(pp, train[, num_cols])
test[, num_cols]  <- predict(pp, test[, num_cols])

# ----------------------------
# 7) Logistic Regression
# ----------------------------
log_model <- glm(y ~ ., data = train, family = binomial)

log_prob <- predict(log_model, newdata = test, type = "response")
log_pred <- ifelse(log_prob > 0.5, "yes", "no")
log_pred <- factor(log_pred, levels = c("no", "yes"))

confusionMatrix(log_pred, test$y, positive = "yes")

# ----------------------------
# 8) Random Forest (Conditional Inference Forest)
# ----------------------------
rf_model <- cforest(y ~ ., data = train,
                    control = cforest_unbiased(mtry = 5, ntree = 300))

rf_pred <- predict(rf_model, newdata = test, type = "response")
rf_pred <- factor(rf_pred, levels = c("no", "yes"))

confusionMatrix(rf_pred, test$y, positive = "yes")

# Feature importance
barplot(varimp(rf_model), main = "Random Forest Feature Importance", las = 2)