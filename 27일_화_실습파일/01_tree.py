# """실습 01: 입력을 질문으로 나누는 트리를 만듭니다. 성능 평가가 아닌 원리 예제입니다."""
# from sklearn.tree import DecisionTreeClassifier

# X = [[100], [200], [300], [700], [800], [900]]
# y = [0, 0, 0, 1, 1, 1]  # 같은 순서의 정답: 성공 0, 실패 1
# model = DecisionTreeClassifier(max_depth=1, random_state=42)
# # TODO B01a: X와 y로 model을 학습하세요.
# pass
# # TODO 끝
# # TODO B01b: 250과 850을 예측하여 prediction에 넣으세요.
# prediction = None
# # TODO 끝
# print("분기 기준:", model.tree_.threshold[0])
# print("250·850 예측:", prediction.tolist())

"""실습 01: 입력을 질문으로 나누는 트리를 만듭니다. 성능 평가가 아닌 원리 예제입니다."""
from sklearn.tree import DecisionTreeClassifier

X = [[100], [200], [300], [700], [800], [900]]
y = [0, 0, 0, 1, 1, 1]  # 같은 순서의 정답: 성공 0, 실패 1
model = DecisionTreeClassifier(max_depth=1, random_state=42)
# TODO B01a: X와 y로 model을 학습하세요.
model.fit(X,y) # model.fit 함수를 사용해 위에 주어진 X 값과 Y값을 받아와 학습
# TODO 끝
# TODO B01b: 250과 850을 예측하여 prediction에 넣으세요.
prediction = model.predict([[250],[850]]) # 250과 850을 예측하라는 힌트와 prediction 함수를 통해 model을 예측하라는 것을 알 수 있으므로
# TODO 끝                       model.predict 입력
print("분기 기준:", model.tree_.threshold[0])
print("250·850 예측:", prediction.tolist())

""" 출력 결과 값
분기 기준 : 500.0
250·850 예측 : [0, 1] """