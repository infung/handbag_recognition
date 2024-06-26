import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from app.AI_Python import ResizePad


class ClassificationModelInference:
    def __init__(self, ):
        directory = os.path.dirname(os.path.abspath(__file__))

        self.model = torch.load(f'{directory}/handbag_classfication.pt',
                                map_location=torch.device('cpu'))
        self.mapping = {}
        with open(f'{directory}/labels.csv') as mapping_file:
            for row in csv.reader(mapping_file):
                self.mapping[str(row[1])] = row[0]

    @staticmethod
    def data_preprocess(raw_input):
        train_sz = (288, 288)
        norm_stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        transform = transforms.Compose([
            transforms.ToPILImage(),
            ResizePad(max_sz=max(train_sz)),
            transforms.ToTensor(),
            transforms.Normalize(*norm_stats),
        ])
        model_input = transform(raw_input)
        model_input = model_input.unsqueeze(0)
        return model_input

    def raw_model_predict(self, model_input):
        model_output = self.model(model_input)
        return F.softmax(model_output, dim=1)

    def data_post_process(self, model_output, n):
        result = {}
        prediction_scores, prediction_class = torch.topk(model_output, n)
        prediction_scores = prediction_scores.squeeze().tolist()
        prediction_class = prediction_class.squeeze().tolist()
        for k, v in zip(prediction_class, prediction_scores):
            if str(k) in self.mapping:
                result[self.mapping[str(k)]] = float(v)

        return result

    def model_predict_main(self, raw_input, n):
        model_input = ClassificationModelInference.data_preprocess(raw_input)
        model_output = self.raw_model_predict(model_input)
        final_output = self.data_post_process(model_output, n)
        return final_output


if __name__ == '__main__':
    InferenceModule = ClassificationModelInference()

    img_path = r'C:\Users\Administrator\Downloads\msbk2201903_3.jpg'
    X = np.array(Image.open(img_path).convert('RGB'))
    num = 5
    model_results = InferenceModule.model_predict_main(X, num)
    print(model_results)
