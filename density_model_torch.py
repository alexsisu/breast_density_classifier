import argparse
import torch

import models_torch as models
import utils


def inference(parameters, verbose=True):
    # resolve device
    device = torch.device(
        "cuda:{}".format(parameters["gpu_number"]) if parameters["device_type"] == "gpu"
        else "cpu"
    )

    # load input images
    datum_l_cc = utils.load_images(parameters['image_path'], 'L-CC')
    datum_r_cc = utils.load_images(parameters['image_path'], 'R-CC')
    datum_l_mlo = utils.load_images(parameters['image_path'], 'L-MLO')
    datum_r_mlo = utils.load_images(parameters['image_path'], 'R-MLO')

    print(datum_l_cc.shape)
    print(datum_r_cc.shape)
    print(datum_l_mlo.shape)
    print(datum_r_mlo.shape)

    # construct models and prepare data
    if parameters["model_type"] == 'cnn':
        model = models.BaselineBreastModel(device, nodropout_probability=1.0, gaussian_noise_std=0.0).to(device)
        model.load_state_dict(torch.load(parameters["model_path"]))
        x = {
            "L-CC": torch.Tensor(datum_l_cc).permute(0, 3, 1, 2).to(device),
            "L-MLO": torch.Tensor(datum_l_mlo).permute(0, 3, 1, 2).to(device),
            "R-CC": torch.Tensor(datum_r_cc).permute(0, 3, 1, 2).to(device),
            "R-MLO": torch.Tensor(datum_r_mlo).permute(0, 3, 1, 2).to(device),
        }
    elif parameters["model_type"] == 'histogram':
        model = models.BaselineHistogramModel(num_bins=parameters["bins_histogram"]).to(device)
        model.load_state_dict(torch.load(parameters["model_path"]))
        x = torch.Tensor(utils.histogram_features_generator([
            datum_l_cc, datum_r_cc, datum_l_mlo, datum_r_mlo
        ], parameters)).to(device)
    else:
        raise RuntimeError(parameters["model_type"])

    # run prediction
    with torch.no_grad():
        prediction_density = model(x).cpu().numpy()

    final_result = 'Density prediction:\n' + '\tAlmost entirely fatty (0):\t\t\t' + str(
        prediction_density[0, 0]) + '\n' + '\tScattered areas of fibroglandular density (1):\t' + str(
        prediction_density[0, 1]) + '\n' + '\tHeterogeneously dense (2):\t\t\t' + str(
        prediction_density[0, 2]) + '\n' + '\tExtremely dense (3):\t\t\t\t' + str(prediction_density[0, 3]) + '\n'
    print('Density prediction:\n'
          '\tAlmost entirely fatty (0):\t\t\t' + str(prediction_density[0, 0]) + '\n'
                                                                                 '\tScattered areas of fibroglandular density (1):\t' + str(
        prediction_density[0, 1]) + '\n'
                                    '\tHeterogeneously dense (2):\t\t\t' + str(prediction_density[0, 2]) + '\n'
                                                                                                           '\tExtremely dense (3):\t\t\t\t' + str(
        prediction_density[0, 3]) + '\n')

    return prediction_density[0]


import os
import ntpath
from shutil import copyfile
import shutil


def prepare_image_for_processing(full_path_input_image):
    if not os.path.exists("temp_images"):
        os.mkdir("temp_images")
    else:
        shutil.rmtree("temp_images")
        os.mkdir("temp_images")

    target_names = ["R-MLO", "R-CC", "L-MLO", "L-CC"]
    for end_name in target_names:
        full_end_name = os.path.join("temp_images", end_name + ".png")
        print(f"Copy {full_path_input_image} to {full_end_name}")
        copyfile(full_path_input_image, full_end_name)
    return "temp_images/"


"""
python density_model_torch.py --input-image /Users/alexsisu/work_phd/breast_density_classifier/images2/3_L_MLO.png cnn
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run Inference')
    parser.add_argument('model_type')
    parser.add_argument('--bins-histogram', default=50)
    parser.add_argument('--model-path', default=None)
    parser.add_argument('--device-type', default="cpu")
    parser.add_argument('--image-path', default="images/")
    parser.add_argument('--input-image', default="images/gogu.png")
    args = parser.parse_args()

    full_path_input_image = args.input_image
    destination_folder = prepare_image_for_processing(full_path_input_image)
    args.image_path = destination_folder
    parameters_ = {
        "model_type": args.model_type,
        "bins_histogram": args.bins_histogram,
        "model_path": args.model_path,
        "device_type": args.device_type,
        "image_path": args.image_path,
    }

    if parameters_["model_path"] is None:
        if args.model_type == "histogram":
            parameters_["model_path"] = "saved_models/BreastDensity_BaselineHistogramModel/model.p"
        if args.model_type == "cnn":
            parameters_["model_path"] = "saved_models/BreastDensity_BaselineBreastModel/model.p"

    inference(parameters_)

"""
python density_model_torch.py histogram
python density_model_torch.py cnn
"""
