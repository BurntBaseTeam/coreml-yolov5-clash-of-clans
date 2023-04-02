import coremltools as ct
import numpy as np
import cv2
import PIL.Image
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from argparse import ArgumentParser


class ObjectDetector:
    REAL_LABELS = {
        "0": "100",
        "1": "air_def",
        "2": "air_sweeper",
        "3": "archer_tower",
        "4": "army",
        "5": "bomb_tower",
        "6": "cannon",
        "7": "cc",
        "8": "champ",
        "9": "dark_mine",
        "10": "dark_storage",
        "11": "eagle",
        "12": "elx_mine",
        "13": "elx_storage",
        "14": "gold_mine",
        "15": "gold_storage",
        "16": "inferno",
        "17": "king",
        "18": "mortar",
        "19": "pet",
        "20": "queen",
        "21": "scatter",
        "22": "th",
        "23": "warden",
        "24": "wiz_tower",
        "25": "xbow"
    }

    def __init__(self, input_folder, output_folder, iou_threshold=0.6, confidence_threshold=0.6):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.model = ct.models.MLModel('models/best.mlmodel')

    def predictMany(self, input_folder):
        for image_name in os.listdir(input_folder):
            extensions = ['.jpg', '.jpeg', '.png']
            if not any(image_name.lower().endswith(ext) for ext in extensions):
                continue
            input_image_path = os.path.join(input_folder, image_name)
            path = self.predict(input_image_path)
            self.visualize(input_image_path, path)

    def predict(self, input_image_path):
        input_image = Image.open(input_image_path).resize((800, 800))
        filenameWithoutExt = os.path.splitext(os.path.basename(input_image_path))[0]

        out_dict = self.model.predict({'image': input_image, "iouThreshold": self.iou_threshold, "confidenceThreshold": self.confidence_threshold})

        out_file_path = os.path.join(self.output_folder, filenameWithoutExt + '.txt')

        with open(out_file_path, "w") as out_file:
            for coordinates, confidence in zip(out_dict["coordinates"], out_dict["confidence"]):
                label_max = confidence.argmax()
                out_file.write("{:d} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(label_max, coordinates[0], coordinates[1], coordinates[2], coordinates[3], confidence[label_max]))

        return out_file_path

    def visualize(self, input_image_path, predictions_file_path):
        img = cv2.imread(input_image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        filenameWithoutExt = os.path.splitext(os.path.basename(input_image_path))[0]

        with open(predictions_file_path, 'r') as f:
            lines = f.readlines()
            predictions = []
            for line in lines:
                data = line.split()
                class_id = int(data[0])
                class_name = self.REAL_LABELS[str(class_id)]
                confidence = float(data[5])
                full_label = f"{class_name} {confidence:.2f}"
                x, y, w, h = map(float, data[1:5])
                x1 = int((x - w / 2) * img.shape[1])
                y1 = int((y - h / 2) * img.shape[0])
                x2 = int((x + w / 2) * img.shape[1])
                y2 = int((y + h / 2) * img.shape[0])
                predictions.append((full_label, x1, y1, x2, y2))

        fig, ax = plt.subplots()
        ax.imshow(img)

        for label, x1, y1, x2, y2 in predictions:
            label = f"{label}"
            bbox = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(bbox)
            plt.text(x1, y1, s=label, color='white', verticalalignment='top',
                    bbox=dict(facecolor='red', alpha=0.5, edgecolor='red', pad=0.5))
            plt.rcParams.update({'font.size': 6})
        plt.savefig(f'output/{filenameWithoutExt}_predict.jpg')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, default='inputs', help='input folder')
    parser.add_argument('--output', type=str, default='output', help='output folder')
    parser.add_argument('--iou_threshold', type=float, default=0.6, help='iou threshold')
    parser.add_argument('--confidence_threshold', type=float, default=0.6, help='confidence threshold')
    args = parser.parse_args()

    detector = ObjectDetector(args.input, args.output, args.iou_threshold, args.confidence_threshold)
    predictions_file_path = detector.predictMany(args.input)
