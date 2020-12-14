from tensorflow import keras
import numpy as np
import cv2
import os
import sys

img_size = (64, 64)
min_size = (32, 32)
test_data_dir = './testpic'
rec_data_dir = './pred_image'
name_dic = {'ran': (0, 0, 255), 'moca': (152, 251, 152), 'himari': (
    193, 182, 255), 'tsugumi': (150, 253, 253), 'tomoe': (128, 0, 128)}
classes = [i for i in name_dic.keys()]


def main():

    model_path = './model/afterglow_model.h5'
    model = keras.models.load_model(model_path)
    test_imagelist = os.listdir(test_data_dir)
    for test_image in test_imagelist:
        file_name = os.path.join(test_data_dir, test_image)
        print(file_name)
        image = cv2.imread(file_name)
        if image is None:
            print("Not open:", file_name)
            continue

        # 顔検出実行
        rec_image = detect_face(image, model)

        # 結果をファイルに保存
        rec_file_name = os.path.join(rec_data_dir, 'pred' + test_image)
        cv2.imwrite(rec_file_name, rec_image)


def detect_face(image, model):

    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cascade_xml = './cascade/lbpcascade_animeface.xml'
    cascade = cv2.CascadeClassifier(cascade_xml)

    # 顔検出の実行
    faces = cascade.detectMultiScale(
        img_gray, scaleFactor=1.11, minNeighbors=2, minSize=min_size)

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            face_img = image[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, img_size)
            # BGR->RGB変換、float型変換
            face_img = cv2.cvtColor(
                face_img, cv2.COLOR_BGR2RGB).astype(np.float32)
            name, score = prediction(face_img, model)
            col = name_dic[name]
            # 認識結果を元画像に表示
            if score >= 0.60:
                cv2.rectangle(image, (x, y), (x+w, y+h), col, 2)
                cv2.putText(image, '%s:%d%%' % (name, score*100),
                            (x+10, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)
            else:
                cv2.rectangle(image, (x, y), (x+w, y+h), (192, 192, 192), 2)
                cv2.putText(image, '%s:%d%%' % ('others', score*100),
                            (x+10, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (192, 192, 192), 2)
    else:
        print('no face')
    return image


def prediction(x, model):
    # 画像データをテンソル整形
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    pred = model.predict(x)[0]

    # 確率が高い上位3キャラを出力
    num = 3
    top_indices = pred.argsort()[-num:][::-1]
    result = [(classes[i], pred[i]) for i in top_indices]
    print(result)
    print('============================================================================')

    # 1番予測確率が高いキャラ名を返す
    return result[0]


if __name__ == '__main__':
    main()
