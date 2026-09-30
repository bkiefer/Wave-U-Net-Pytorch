import argparse
import os

import torch
import numpy as np
import data.utils
import model.utils as model_utils

from model.waveunet import Waveunet

def predict_enhancement(audio, model):
    '''
    Predict sources for a given audio input signal, with a given model. Audio is split into chunks to make predictions on each chunk before they are concatenated.
    :param audio: Audio input tensor, either Pytorch tensor or numpy array
    :param model: Pytorch model
    :return: Source predictions, dictionary with source names as keys
    '''
    if isinstance(audio, torch.Tensor):
        is_cuda = audio.is_cuda()
        audio = audio.detach().cpu().numpy()
        return_mode = "pytorch"
    else:
        return_mode = "numpy"

    expected_outputs = audio.shape[1]

    # Pad input if it is not divisible in length by the frame shift number
    output_shift = model.shapes["output_frames"]
    pad_back = audio.shape[1] % output_shift
    pad_back = 0 if pad_back == 0 else output_shift - pad_back
    if pad_back > 0:
        audio = np.pad(audio, [(0,0), (0, pad_back)], mode="constant", constant_values=0.0)

    target_outputs = audio.shape[1]
    outputs = np.zeros(audio.shape, np.float32)

    # Pad mixture across time at beginning and end so that neural network can make prediction at the beginning and end of signal
    pad_front_context = model.shapes["output_start_frame"]
    pad_back_context = model.shapes["input_frames"] - model.shapes["output_end_frame"]
    audio = np.pad(audio, [(0,0), (pad_front_context, pad_back_context)], mode="constant", constant_values=0.0)

    # Iterate over mixture magnitudes, fetch network prediction
    with torch.no_grad():
        for target_start_pos in range(0, target_outputs, model.shapes["output_frames"]):
            # Prepare mixture excerpt by selecting time interval
            # Since audio was front-padded input of [targetpos:targetpos+inputframes] actually predicts [targetpos:targetpos+outputframes] target range
            curr_input = audio[:, target_start_pos:target_start_pos + model.shapes["input_frames"]]

            # Convert to Pytorch tensor for model prediction
            curr_input = torch.from_numpy(curr_input).unsqueeze(0)

            # Predict
            curr_targets = model(curr_input)
            outputs[:,target_start_pos:target_start_pos+model.shapes["output_frames"]] = curr_targets.squeeze(0).cpu().numpy()

    # Crop to expected length (since we padded to handle the frame shift)
    outputs = outputs[:,:expected_outputs]

    if return_mode == "pytorch":
        outputs = torch.from_numpy(outputs)
        if is_cuda:
            outputs = outputs.cuda()
    return outputs



def enhance_audio(audio_path, model, channels=1, sr=16000, output_duration=2.0):
    '''
    Predicts sources for an audio file for which the file path is given, using a given model.
    Takes care of resampling the input audio to the models sampling rate and resampling predictions back to input sampling rate.
    :param args: Options dictionary
    :param audio_path: Path to mixture audio file
    :param model: Pytorch model
    :return: Source estimates given as dictionary with keys as source names
    '''
    model.eval()

    # Load mixture in original sampling rate
    input_audio, input_sr = data.utils.load(audio_path, sr=None, mono=False)
    input_channels = input_audio.shape[0]
    input_len = input_audio.shape[1]

    sources = predict_enhancement(input_audio, model)

    # output_samples= int(output_duration * sr)

    # # Adapt mixture channels to required input channels
    # if channels == 1:
    #     input_audio = np.mean(input_audio, axis=0, keepdims=True)
    # else:
    #     if input_channels == 1: # Duplicate channels if input is mono but model is stereo
    #         input_audio = np.tile(input_audio, [channels, 1])
    #     else:
    #         assert(input_channels == channels)

    # # resample to model sampling rate
    # input_audio = data.utils.resample(input_audio, input_sr, sr)

    # output_audio = []
    # index = 1
    # while index < input_len:
    #     sources = predict_enhancement(input_audio, model)

    #     inp_len = min(output_samples, input_len - index)

    #     # Resample back to mixture sampling rate in case we had model on different sampling rate
    #     sources = data.utils.resample(sources[index:index + inp_len]
    #                                           , sr, input_sr)
    #     output_audio += sources

    #     diff = sources.shape[1] - inp_len
    #     if diff > 0:
    #         print("WARNING: Cropping " + str(diff) + " samples")
    #         sources = sources[:, :-diff]
    #     elif diff < 0:
    #         print("WARNING: Padding output by " + str(diff) + " samples")
    #         sources = np.pad(sources, [(0,0), (0, -diff)], "constant")

    #     index += sources.shape[1]

    # # In case we had to pad the mixture at the end, or we have a few samples too many due to inconsistent down- and upsamṕling, remove those samples from source prediction now


    # # Adapt channels
    # if input_channels > channels:
    #     assert(channels == 1)
    #     # Duplicate mono predictions
    #     sources = np.tile(sources, [input_channels, 1])
    # elif input_channels < channels:
    #     assert(input_channels == 1)
    #     # Reduce model output to mono
    #     sources = np.mean(sources, axis=0, keepdims=True)

    sources = np.asfortranarray(sources) # So librosa does not complain if we want to save it

    return sources


def main(args):
    # MODEL
    num_features = [args.features*i for i in range(1, args.levels+1)] if args.feature_growth == "add" else \
                   [args.features*2**i for i in range(0, args.levels)]
    target_outputs = int(args.output_size * args.sr)
    model = Waveunet(args.channels, num_features, args.channels, kernel_size=args.kernel_size,
                     target_output_size=target_outputs, depth=args.depth, strides=args.strides,
                     conv_type=args.conv_type, res=args.res)

    if args.cuda:
        model = model_utils.DataParallel(model)
        print("move model to gpu")
        model.cuda()

    print("Loading model from checkpoint " + str(args.load_model))
    state = model_utils.load_model(model, None, args.load_model, args.cuda,
                                   weights_only=False)
    print('Step', state['step'])

    preds = enhance_audio(args.input, model)

    output_folder = os.path.dirname(args.input) if args.output is None else args.output
    data.utils.write_wav(os.path.join(output_folder, os.path.basename(args.input) + "_out.wav"), preds, args.sr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA (default: False)')
    parser.add_argument('--load_model', type=str, default='checkpoints/waveunet/model',
                        help='Reload a previously trained model')
    parser.add_argument('--sr', type=int, default=16000,
                        help="Sampling rate")
    parser.add_argument('--input', type=str, default=os.path.join("audio_examples", "test.wav"),
                        help="Path to input to be enhanced")
    parser.add_argument('--output', type=str, default=None,
                        help="Output path (same folder as input path if not set)")

    parser.add_argument('--levels', type=int, default=6,
                        help="Number of DS/US blocks")
    parser.add_argument('--features', type=int, default=32,
                        help='Number of feature channels per layer')
    parser.add_argument('--channels', type=int, default=1,
                        help="Number of input audio channels")
    parser.add_argument('--output_size', type=float, default=2.0,
                        help="Output duration")
    parser.add_argument('--kernel_size', type=int, default=5,
                        help="Filter width of kernels. Has to be an odd number")
    parser.add_argument('--depth', type=int, default=1,
                        help="Number of convs per block")
    parser.add_argument('--strides', type=int, default=4,
                        help="Strides in Waveunet")
    parser.add_argument('--conv_type', type=str, default="gn",
                        help="Type of convolution (normal, BN-normalised, GN-normalised): normal/bn/gn")
    parser.add_argument('--res', type=str, default="learned",
                        help="Resampling strategy: fixed sinc-based lowpass filtering or learned conv layer: fixed/learned")
    parser.add_argument('--feature_growth', type=str, default="double",
                        help="How the features in each layer should grow, either (add) the initial number of features each time, or multiply by 2 (double)")

    args = parser.parse_args()

    main(args)
