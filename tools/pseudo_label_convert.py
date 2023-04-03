from os import rename
from glob import glob
import argparse
from PIL import Image
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('files', help="files to convert")

    args = parser.parse_args()

    print(args)

    paths = glob(args.files)

    for path in tqdm(paths):

        img = Image.open(path).convert('L')
        
        img.save(path)