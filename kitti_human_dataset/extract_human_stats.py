import torchvision
from torchvision import transforms as T
import PIL.Image as pil
import os.path
import pickle
import argparse

data_root = "./data/val/raw/"
human_label = 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--drives_file', default='validation_drives.txt',
                        type=str,
                        help='file containing the drive names to be processed')
    parser.add_argument('--output', default='stats.pickle', type=str,
                        help='output file name')

    args = parser.parse_args()

    with open(args.drives_file) as f:
        drives = f.read().splitlines()

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True).cuda().eval()
    transform = T.Compose([T.ToTensor()])

    stat_dict = {}
    for drive in drives:
        directory = os.path.join(data_root, drive + '_sync', 'image_02',
                                 'data')
        image_names = os.listdir(directory)

        print("Processing drive: ", drive)
        for i in range(len(image_names)):
            img_name = image_names[i]
            print("Reading: ", img_name)

            img = pil.open(os.path.join(directory, img_name))
            img_tensor = transform(img).cuda()

            predictions = model([img_tensor])[0]

            labels = predictions['labels'].cpu().numpy()
            scores = predictions['scores'].cpu().detach().numpy()
            stat_dict[drive + '/' + img_name] = list(
                scores[labels == human_label])

    print(stat_dict)
    with open(args.output, 'wb') as handle:
        pickle.dump(stat_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
