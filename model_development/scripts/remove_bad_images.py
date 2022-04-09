from struct import unpack
from tqdm import tqdm
import os


marker_mapping = {
    0xffd8: "Start of Image",
    0xffe0: "Application Default Header",
    0xffdb: "Quantization Table",
    0xffc0: "Start of Frame",
    0xffc4: "Define Huffman Table",
    0xffda: "Start of Scan",
    0xffd9: "End of Image"
}


class JPEG:
    def __init__(self, image_file):
        with open(image_file, 'rb') as f:
            self.img_data = f.read()
    
    def decode(self):
        data = self.img_data
        while(True):
            marker, = unpack(">H", data[0:2])
            # print(marker_mapping.get(marker))
            if marker == 0xffd8:
                data = data[2:]
            elif marker == 0xffd9:
                return
            elif marker == 0xffda:
                data = data[-2:]
            else:
                lenchunk, = unpack(">H", data[2:4])
                data = data[2+lenchunk:]            
            if len(data)==0:
                break        


bads = []

image_labels = os.listdir("dataset/TRAIN")

images = []

for label in image_labels:
    bads = []
    images = os.listdir("dataset/TRAIN/"+label)

    root_img = f"dataset/TRAIN/{label}/"

    # print(images)

    for img in tqdm(images):
        image = os.path.join(root_img,img)
        image = JPEG(image) 
        try:
            image.decode()   
        except:
            bads.append(img)
    print(label)
    print(bads)
#     # for name in bads:
#     #     os.remove(os.path.join(root_img,name))

# import splitfolders # or import splitfolders
# input_folder = "dataset/TRAIN"
# output = "dataset/TEST" #where you want the split datasets saved. one will be created if it does not exist or none is set

# splitfolders.ratio(input_folder, output=output, seed=42, ratio=(.8, .2))