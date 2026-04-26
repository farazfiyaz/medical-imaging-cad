def start_process(imagepath):
    import os
    from django.conf import settings
    from keras.models import load_model
    from keras.utils import load_img
    import numpy as np
    from PIL import Image

    img_path = os.path.join(settings.MEDIA_ROOT, imagepath)
    model_path = os.path.join(settings.MEDIA_ROOT, 'mammography_model3.h5')

    def getCropImgs(img, needRotations=False):
        z = np.asarray(img, dtype=np.int8)
        c = []
        for i in range(3):
            for j in range(4):
                crop = z[512 * i:512 * (i + 1), 512 * j:512 * (j + 1), :]
                c.append(crop)
                if needRotations:
                    c.append(np.rot90(np.rot90(crop)))
        return c

    def softmaxToProbs(soft):
        z_exp = [np.math.exp(i) for i in soft[0]]
        sum_z_exp = sum(z_exp)
        return [(i / sum_z_exp) * 100 for i in z_exp]

    def predict(img, model_path, showImg=True):
        model = load_model(model_path)
        x = img
        if showImg:
            Image.fromarray(np.array(img, np.float16), 'RGB').show()
        x = np.expand_dims(x, axis=0)
        softMaxPred = model.predict(x)
        probs = softmaxToProbs(softMaxPred)
        maxprob = 0
        maxI = 0
        for j in range(len(probs)):
            if probs[j] > maxprob:
                maxprob = probs[j]
                maxI = j
        return maxI, probs

    def predictImage(img_path=img_path, arrayImg=None, printData=True):
        crops = []
        if arrayImg == None:
            img = load_img(img_path)
            crops = np.array(getCropImgs(img, needRotations=False), np.float16)
            crops = np.divide(crops, 255.)
        Image.fromarray(np.array(crops[0]), "RGB").show()

        classes = ["Benign", "InSitu", "Invasive", "Normal"]
        compProbs = [0, 0, 0, 0]

        for i in range(len(crops)):
            ___, probs = predict(crops[i], model_path, showImg=False)
            for j in range(len(classes)):
                compProbs[j] += probs[j]

        prediction = np.argmax(compProbs)
        prediction = classes[prediction]
        return prediction

    return predictImage()
