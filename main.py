import pandas as pd 



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix, accuracy_score



df = pd.read_csv("titanic.csv")



df["Embarked"].fillna('S', inplace=True)

df[list(pd.get_dummies(df["Embarked"]).columns)] = pd.get_dummies(df["Embarked"])

df.drop(["PassengerId", "Name", "Ticket", "Cabin", "Embarked"], 1, inplace=True)



print(df.groupby("Pclass")["Age"].median())



first_class_age = df[df["Pclass"] == 1]["Age"].median()

second_class_age = df[df["Pclass"] == 2]["Age"].median()

third_class_age = df[df["Pclass"] == 3]["Age"].median()



def fill_age(row):

    if pd.isnull(row["Age"]):

        if row["Pclass"] == 1:

            return first_class_age

        elif row["Pclass"] == 2:

            return second_class_age

        return third_class_age

    return row["Age"]



def fill_sex(sex):

    if sex == "male":

        return 0

    return 1





df["Age"] = df.apply(fill_age, axis=1)

df["Sex"] = df["Sex"].apply(fill_sex)



x = df.drop("Survived", axis=1)

y = df["Survived"] 



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)



sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)



classifier = KNeighborsClassifier(n_neighbors = 5)

classifier.fit(x_train, y_train)



y_pred = classifier.predict(x_test)



percent = accuracy_score(y_test, y_pred) * 100



# print(df.info())

print(percent)

