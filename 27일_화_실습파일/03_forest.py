# """실습 03 응용: 같은 조건의 단일 트리와 숲을 비교합니다."""
# from pathlib import Path
# import pandas as pd

# # 제공: 전날의 분할 소속뿐 아니라 position 순서까지 그대로 읽습니다.
# HERE = Path(__file__).resolve().parent
# raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
# manifest = pd.read_csv(HERE / "data/split_manifest.csv")
# train_ids = manifest.loc[manifest["split"] == "train"].sort_values("position")["request_id"]
# valid_ids = manifest.loc[manifest["split"] == "valid"].sort_values("position")["request_id"]
# train = raw.loc[train_ids]
# valid = raw.loc[valid_ids]
# FEATURES = ["input_chars", "allowed_tools", "task_type", "priority"]
# X_train, X_valid = train[FEATURES], valid[FEATURES]
# y_train, y_valid = train["failed"], valid["failed"]
# OUT = HERE / "outputs"
# OUT.mkdir(exist_ok=True)
# # test 20건의 입력·정답은 아래 모델에 전달하지 않습니다.
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.base import clone

# # 제공: 전날 배운 결측 대체와 문자 변환입니다. 통계는 fit할 때만 배웁니다.
# pre = ColumnTransformer([
#     ("num", SimpleImputer(strategy="median"), FEATURES[:2]),
#     ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
#                       ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), FEATURES[2:]),
# ])
# # 트리는 이 실습에서 표준화하지 않습니다. 각 후보·fold마다 새 전처리기를 학습합니다.

# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import f1_score, confusion_matrix

# candidates = [("Tree", DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=42)),
#               ("Forest", RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=2,
#                                     max_features="sqrt", random_state=42, n_jobs=1))]
# rows = []
# for name, estimator in candidates:
#     model = Pipeline([("pre", clone(pre)), ("model", estimator)])
#     # TODO A03: model을 train으로 학습한 뒤 valid를 예측하세요.
#     pass
#     prediction = None
#     # TODO 끝
#     tn, fp, fn, tp = confusion_matrix(y_valid, prediction, labels=[0, 1]).ravel()
#     rows.append({"model": name, "train_F1": f1_score(y_train, model.predict(X_train), zero_division=0),
#                  "valid_F1": f1_score(y_valid, prediction, zero_division=0), "FP": fp, "FN": fn})
# result = pd.DataFrame(rows)
# result.to_csv(OUT / "03_forest.csv", index=False, encoding="utf-8-sig")
# print(result.round(3).to_string(index=False))


"""실습 03 응용: 같은 조건의 단일 트리와 숲을 비교합니다."""
from pathlib import Path
import pandas as pd

# 제공: 전날의 분할 소속뿐 아니라 position 순서까지 그대로 읽습니다.
HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
manifest = pd.read_csv(HERE / "data/split_manifest.csv")
train_ids = manifest.loc[manifest["split"] == "train"].sort_values("position")["request_id"]
valid_ids = manifest.loc[manifest["split"] == "valid"].sort_values("position")["request_id"]
train = raw.loc[train_ids]
valid = raw.loc[valid_ids]
FEATURES = ["input_chars", "allowed_tools", "task_type", "priority"]
X_train, X_valid = train[FEATURES], valid[FEATURES]
y_train, y_valid = train["failed"], valid["failed"]
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
# test 20건의 입력·정답은 아래 모델에 전달하지 않습니다.
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone

# 제공: 전날 배운 결측 대체와 문자 변환입니다. 통계는 fit할 때만 배웁니다.
pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), FEATURES[:2]),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), FEATURES[2:]),
])
# 트리는 이 실습에서 표준화하지 않습니다. 각 후보·fold마다 새 전처리기를 학습합니다.

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix

candidates = [("Tree", DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=42)),
              ("Forest", RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=2,
                                    max_features="sqrt", random_state=42, n_jobs=1))]
rows = []
for name, estimator in candidates:
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])
    # TODO A03: model을 train으로 학습한 뒤 valid를 예측하세요.
    model.fit(X_train, y_train) # 바로 윗 줄에 적힌 model을 train으로 학습하라는 힌트에서 X값과 y값에 들어간 train을 확인 후 입력
    prediction = model.predict(X_valid) #prediction이 주어진 것은 다른 실습과 마찬가지로 model을 predict 힌트이므로 값이 존재하는 X_valid를 입력
    # TODO 끝
    tn, fp, fn, tp = confusion_matrix(y_valid, prediction, labels=[0, 1]).ravel()
    rows.append({"model": name, "train_F1": f1_score(y_train, model.predict(X_train), zero_division=0),
                 "valid_F1": f1_score(y_valid, prediction, zero_division=0), "FP": fp, "FN": fn})
result = pd.DataFrame(rows)
result.to_csv(OUT / "03_forest.csv", index=False, encoding="utf-8-sig")
print(result.round(3).to_string(index=False))

""" 출력 결과 값
 model  train_F1  valid_F1  FP  FN
  Tree      0.71     0.333   4   4
Forest      0.80     0.222   2   5 
"""
