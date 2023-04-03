from os import rename
from glob import glob
import argparse
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('files', help="files to add '_labelTrainIds' extension to")

    args = parser.parse_args()

    print(args)

    paths = glob(args.files)

    for path in tqdm(paths):
        
        new_path = path.replace("_leftImg8bit", "_gtFine_labelTrainIds")

        rename(path, new_path)
