import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2

EPOCHS = 200
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2
DROPOUT = 0.3

mnist = keras.datasets.mnist
# 레이블은 원 핫 인코딩으로 표현
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train data는 60000개의 행으로 28*28의 값을 가진다. 이를 60000 x 784의 형태로 변환
RESHAPED = 784
# cannot reshape array of size 47040000 into shape 오류 뜨는 이유
# 이미지 사이즈가 차이가 나 계산 불가능 -> 이미지 사이즈 조절해주어야한다.
x_train = x_train.reshape(60000,RESHAPED)
x_test = x_test.reshape(60000,RESHAPED)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# 입력을 [0,1] 사이로 정규화
x_train /= 255
x_test /= 255

print(x_train.shape[0], 'train samples')
print(x_test.shape[0],'test sample')

# 레이블을 이용한 원 핫 인코딩
y_train = tf.keras.utils.to_categorical(y_train, NB_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, NB_CLASSES)

# 마지막 계층은 소프트맥스 함수인 단일 뉴런으로, 시그모이드 함수를 일반화 한것이다.
# 시그 모이드 함수는 입력이 -무한에서 무한일 대 출력은 (0,1) 사이에 존재한다.
# 유사하게 소프트맥스는 임의의 실수값의 K차원 벡터를 밀어 넣어 총합이 1이 되게 한다.

model = tf.keras.models.Sequential()
model.add(keras.layers.Dense(N_HIDDEN, 
                             input_shape=(RESHAPED,),
                             name='Dense_layer0',
                             activation='relu'))
# 은닉층  내부 밀집 신경망에 전파된 값중 일부를 무작위로 제거하자 ( 두번째 개선 )
# 이는 정규화의 한 형태이다.
# 몇 가지 값을 무작위로 제거한다는 아이디어를 통해 성능을 향상시킬 수 있다.
# 무작위 드롭아웃으로 신경망의 일반화를 향상시키는데 도움이 되는 중복 패턴을 학습시킨다는 것이다.
model.add(keras.layers.Dropout(DROPOUT))
# 단순 신경망에서 은닉층으로 개선 ( 첫번째 개선 )
# 1. 신경망에 계층을 추가한다. -> 계층을 추가하게 되면 매개변수가 추가되어 모델이 더 복잡한 패턴을 기억할 수 있게 된다.
# 2. 밀집 (Dense) 계층을 갖게 추가한다. 입력 또는 출력과 직접 연결되지 않기 때문에 은닉된 것으로 간주된다.
model.add(keras.layers.Dense(N_HIDDEN, 
                             name='Dense_layer1',
                             activation='relu'))
model.add(keras.layers.Dropout(DROPOUT))
model.add(keras.layers.Dense(NB_CLASSES,
                             name='Dense_layer2',
                             activation='softmax'))

# 모델 요약
model.summary()

# 모델을 정의하고 난 후 모델을 컴파일 해야한다.
# 1. 모델을 훈련시키는 동안 가중치를 업데이트하는데 사용하는 알고리즘인 최적화기(optimizer) 선택하기
# 2. optimizer가 가중치 공간을 탐색하는데 사용할 목적함수(objective function) 선택
# 목적 함수는 손실(loss)함수 또는 비용(cost)함수 하고 한다.
# 3. 훈련된 모델을 평가해야한다.

model.compile(optimizer='SDG',
              loss= 'categorical_crossentropy',
              metrics=['accuracy'])
# optimizer를 바꿔가며 효율 비교히가 ( RMSProp, Adamm SDG 셋중에 하나 선택)
# SDG : 확률적 그래디언트 하강은 각 훈련 epoch마다 신경망의 오차를 줄이고자 사용된다.
# epochs : 모델이 훈련 집합에 노출된 횟수이다. 목표함수가 최소화 되도록 가중치를 조정하려한다.
# batch_size : 최적화기가 가중치 갱신을 수행하기 전에 관찰한 훈련 인스턴스이다.

# 모델 훈련
model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=VERBOSE,
          validation_split=VALIDATION_SPLIT)

# 훈련에 사용된 예시는 모델 평가에 사용해서는 안되기 때문에 훈련 데이터와 검증데이터를 분리해준다.

#모델 평가
test_loss, test_acc = model.evaluate(x_test,y_test)
print('Tesr accuracy : ', test_acc)

# to_categorical(y_train, NB_CLASSES)는 배열 y_train을 개수만큼의 열을 가진 행렬로 변환
# labels
# to_categorical(labels)

