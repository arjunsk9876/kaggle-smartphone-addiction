import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv("train.csv")

sns.scatterplot(x="Age", y="Fare", hue="Survived", data=train)
sns.lmplot(x="Age", y="Fare", hue="Survived", data=train)

plt.figure(figsize=(10,6))
plt.title("Average Survival Rate by Passenger Class for the Titanic")
sns.barplot(x="Pclass", y="Survived", data=train)
plt.xlabel("Passenger Class")
plt.ylabel("Average Survival Rate")


plt.show()
