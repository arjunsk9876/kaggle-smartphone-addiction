import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train["age"] = train["age"].fillna(train["age"].median())
train["daily_screen_time_hours"] = train["daily_screen_time_hours"].fillna(train["daily_screen_time_hours"].median())
train["social_media_hours"] = train["social_media_hours"].fillna(train["social_media_hours"].median())
train["gaming_hours"] = train["gaming_hours"].fillna(train["gaming_hours"].median())
train["work_study_hours"] = train["work_study_hours"].fillna(train["work_study_hours"].median())
train["sleep_hours"] = train["sleep_hours"].fillna(train["sleep_hours"].median())
train["notifications_per_day"] = train["notifications_per_day"].fillna(train["notifications_per_day"].median())
train["app_opens_per_day"] = train["app_opens_per_day"].fillna(train["app_opens_per_day"].median())
train["weekend_screen_time"] = train["weekend_screen_time"].fillna(train["weekend_screen_time"].median())

train["gender"] = train["gender"].fillna("Unknown")
train["stress_level"] = train["stress_level"].fillna("Unknown")
train["academic_work_impact"] = train["academic_work_impact"].fillna("Unknown").map({"No": 0, "Yes": 1, "Unknown": 2})

train["recreation_hours"] = train["social_media_hours"] + train["gaming_hours"]
train["screen_sleep_ratio"] = train["daily_screen_time_hours"] / train["sleep_hours"]

y = train["addicted_label"]

features = [
    "age",
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "notifications_per_day",
    "app_opens_per_day",
    "weekend_screen_time",
    "gender",
    "stress_level",
    "academic_work_impact",
    "recreation_hours",
    "screen_sleep_ratio",
    "gender_Male", "gender_Female", "gender_Other", "gender_Unknown",
    "stress_level_Low", "stress_level_Medium", "stress_level_High", "stress_level_Unknown"
]

train = pd.get_dummies(train, columns=["gender", "stress_level"])
X = train.reindex(columns=features, fill_value=0)

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

scale_pos_weight = (y == 0).sum() / (y == 1).sum()
model = XGBClassifier(random_state=1, n_jobs=-1, n_estimators=500, max_depth=6, learning_rate=0.03, scale_pos_weight=scale_pos_weight)
model.fit(train_X, train_y)
val_predictions = model.predict(val_X)
val_proba = model.predict_proba(val_X)[:, 1]

val_accuracy = accuracy_score(val_y, val_predictions)
val_auc = roc_auc_score(val_y, val_proba)

print("Validation Accuracy:", val_accuracy)
print("Validation AUC:", val_auc)

model.fit(X, y)


test["age"] = test["age"].fillna(test["age"].median())
test["daily_screen_time_hours"] = test["daily_screen_time_hours"].fillna(test["daily_screen_time_hours"].median())
test["social_media_hours"] = test["social_media_hours"].fillna(test["social_media_hours"].median())
test["gaming_hours"] = test["gaming_hours"].fillna(test["gaming_hours"].median())
test["work_study_hours"] = test["work_study_hours"].fillna(test["work_study_hours"].median())
test["sleep_hours"] = test["sleep_hours"].fillna(test["sleep_hours"].median())
test["notifications_per_day"] = test["notifications_per_day"].fillna(test["notifications_per_day"].median())
test["app_opens_per_day"] = test["app_opens_per_day"].fillna(test["app_opens_per_day"].median())
test["weekend_screen_time"] = test["weekend_screen_time"].fillna(test["weekend_screen_time"].median())

test["gender"] = test["gender"].fillna("Unknown")
test["stress_level"] = test["stress_level"].fillna("Unknown")
test["academic_work_impact"] = test["academic_work_impact"].fillna("Unknown").map({"No": 0, "Yes": 1, "Unknown": 2})

test["recreation_hours"] = test["social_media_hours"] + test["gaming_hours"]
test["screen_sleep_ratio"] = test["daily_screen_time_hours"] / test["sleep_hours"]

test = pd.get_dummies(test, columns=["gender", "stress_level"])
test_X = test.reindex(columns=features, fill_value=0)
test_predictions = model.predict(test_X)

print(test_predictions[:10])

submission = pd.DataFrame({
    "id": test["id"],
    "addicted_label": test_predictions
})

submission.to_csv("submission.csv", index=False)