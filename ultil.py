import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.over_sampling import SMOTEN
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
# Ultilities function for project


def plot_confusion(test, input_pred):
    df_cm = pd.DataFrame(confusion_matrix(test, input_pred), range(1,len(test)), range(1,len(test)))
    # plt.figure(figsize=(10,7))
    sns.set(font_scale=1.4)  # for label size
    sns.heatmap(df_cm, annot=True, cmap='PuRd')  # font size
    plt.title('Confusion Matrix')
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.show()
    return df_cm


def box_plot(param, df_input):
    sns.set(rc={'figure.figsize': (15, 15)})
    sns.boxplot(data=df_input[param], palette="Set2")
    plt.title("Categorical boxplot")
    plt.show()


def dis_plot(param, df_input):
    sns.set(rc={'figure.figsize': (15, 15)})
    df_input[param].apply(pd.Series.value_counts).plot.bar()
    plt.xlabel("Category", size=14)
    plt.ylabel("Count", size=14)
    plt.title("Categorical Distribution plot")
    plt.show()


def featureSelection(df, target):
    x_df = df.drop(target, axis=1)
    y_df = df[target]
    return x_df, y_df


def rebalancing(x_df, y_df):
    smoten = SMOTEN(random_state=42)
    X_res, y_res = smoten.fit_resample(x_df, y_df)
    # summarize distribution
    counter = Counter(y_res)
    for k, v in counter.items():
        per = v / len(y_res) * 100
        print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
    # plot the distribution
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.bar(counter.keys(), counter.values())
    plt.show()
    return X_res, y_res


def build_and_test_model(X_train,y_train,X_test,y_test, trial):
    param = trial.params.copy()
    scalers = param.get('scalers')
    if scalers == "minmax":
        scaler = MinMaxScaler()
    elif scalers == "standard":
        scaler = StandardScaler()
    else:
        scaler = RobustScaler()
    param.pop('scalers')
    param.pop('dim_red')
    clf= RandomForestClassifier(**param)
    pipeline = make_pipeline(scaler, clf)
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    print(classification_report(y_test, pred))
    plot_confusion(y_test, pred)
    print("The function will return the model")
    return pipeline