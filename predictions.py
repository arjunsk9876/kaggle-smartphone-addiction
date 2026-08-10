import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
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

train["gender"] = train["gender"].fillna("Unknown").map({"Male": 0, "Female": 1, "Other": 2, "Unknown": 3})
train["stress_level"] = train["stress_level"].fillna("Unknown").map({"Low": 0, "Medium": 1, "High": 2, "Unknown": 3})
train["academic_work_impact"] = train["academic_work_impact"].fillna("Unknown").map({"No": 0, "Yes": 1, "Unknown": 2})

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
    "stress_level"
]

X = train[features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

model = XGBClassifier(random_state=1, n_jobs=-1, n_estimators=300, max_depth=5, learning_rate=0.05)
model.fit(train_X, train_y)
val_predictions = model.predict(val_X)

val_accuracy = accuracy_score(val_y, val_predictions)

print("Validation Accuracy:", val_accuracy)

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

test["gender"] = test["gender"].fillna("Unknown").map({"Male": 0, "Female": 1, "Other": 2, "Unknown": 3})
test["stress_level"] = test["stress_level"].fillna("Unknown").map({"Low": 0, "Medium": 1, "High": 2, "Unknown": 3})
test["academic_work_impact"] = test["academic_work_impact"].fillna("Unknown").map({"No": 0, "Yes": 1, "Unknown": 2})

test_X = test[features]
test_predictions = model.predict(test_X)

print(test_predictions[:10])

submission = pd.DataFrame({
    "id": test["id"],
    "addicted_label": test_predictions
})

submission.to_csv("submission.csv", index=False)