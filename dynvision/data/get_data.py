import argparse
import logging
from pathlib import Path

from torchvision import datasets
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Download dataset")
parser.add_argument("--output", type=Path)
parser.add_argument("--data_name", type=str)
parser.add_argument("--raw_data_path", type=Path)
parser.add_argument("--subset", type=str)
parser.add_argument("--ext", type=str, default="png")


def download_data(output_folder, data_name, train):
    if Path(output_folder).exists() and any(Path(output_folder).iterdir()):
        download = False
    else:
        download = True

    Path(output_folder).mkdir(parents=True, exist_ok=True)

    if hasattr(datasets, data_name):
        dataset = getattr(datasets, data_name)(
            root=output_folder, train=train, download=download
        )
    else:
        raise ValueError(f"Dataset not available in torchvision!")

    return dataset


def store_images(image_folder, dataset, ext="png"):
    if Path(image_folder).exists() and any(Path(image_folder).iterdir()):
        logging.warning(f"Folder {image_folder} already exists and is not empty.")
        return None

    # Create a new folder to store the images
    image_folder.mkdir(parents=True, exist_ok=True)

    logging.info(f"Saving images to: {image_folder}")
    # Iterate through the MNIST dataset and save the images to the new folder
    for i, (image, label) in tqdm(enumerate(dataset), total=len(dataset)):
        label_folder = image_folder / str(label)
        label_folder.mkdir(parents=True, exist_ok=True)

        image_path = label_folder / f"{i}.{ext}"
        image.save(image_path)

    return None


if __name__ == "__main__":

    logger = logging.getLogger(__name__)
    args = parser.parse_args()

    if args.subset in ["train", "test"]:
        train = args.subset == "train"
    else:
        raise ValueError(f"Invalid subset: {args.subset}")

    dataset = download_data(
        output_folder=args.raw_data_path, data_name=args.data_name, train=train
    )

    if args.output.suffix:  # If the output is a file
        args.output = args.output.parent

    store_images(args.output, dataset, ext=args.ext)
