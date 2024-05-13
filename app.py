import numpy as np
import os
import torch
import librosa
import soundfile as sf

from whisper_normal import MyWhisper2Normal
import argparse
import gradio as gr

'''
Whisper to Normal conversion demo
'''

w2n = None

def main(args):
    global w2n
    w2n = MyWhisper2Normal(args)
    
def process(input_wav):
    try:
        print("input_wav:", input_wav)
        sr = input_wav[0]
        whisp_wav = input_wav[1]
        print("whisp_wav shape:", whisp_wav.shape)
        if len(whisp_wav.shape) >=2:
            whisp_wav = whisp_wav[:,0]
        print("whisp_wav2 shape:", whisp_wav.shape)
        target_sr = 16000
        whisp_wav = whisp_wav.astype(np.float32) / 32768.0
        whisp_wav = librosa.resample(whisp_wav,
                                    orig_sr=sr,
                                    target_sr=target_sr,
                                    res_type="kaiser_fast")
        
        #whisp_wav, sr = librosa.load(input_wav, sr=16000)
        #print("input_wav:", input_wav)
        normal_wav, _ = w2n.convert(whisp_wav)
        output_wav = "output.wav"
        sf.write(output_wav, normal_wav, 16000)
        return output_wav
    except Exception as e:
        print("got exception:", e)
        return ""

 

def get_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
#    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
#        return "mps" # M1 mac GPU
    else:
        return "cpu"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input",
                        default = "sample_whisper.wav"
    )

    parser.add_argument("--output",
                        default = "converted.wav"
    )

    parser.add_argument("--preprocess_config",
        default = 'config/my_preprocess16k_LJ.yaml'
    )

    parser.add_argument("--model_config",
        default = 'config/my_model16000.yaml'
    )

    '''
    parser.add_argument("--train_config",
        default = 'config/my_train16k_LJ.yaml'
    )
    '''

    parser.add_argument("--device", 
        default = get_default_device()
    )

    parser.add_argument("--hubert", 
        help="hubert checkpoint path",
        #default="models/hubert/model-layer12-450000.pt"
        default="https://github.com/rkmt/wesper-demo/releases/download/v0.1/model-layer12-450000.pt",

    )

    parser.add_argument("--fastspeech2", 
        help="fastspeech2 checkpoint path",
        #default="models/fastspeech2/lambda_best.tar"
        #default="models/fastspeech2/googletts_neutral_best.tar"        
        default="https://github.com/rkmt/wesper-demo/releases/download/v0.1/googletts_neutral_best.tar",
    )

    parser.add_argument("--hifigan", 
        help="hifigan checkpoint path",
        #default="./hifigan/g_00205000"
        default="https://github.com/rkmt/wesper-demo/releases/download/v0.1/g_00205000",
    )

    args = parser.parse_args()   

    print("### args ###\n", args)
    main(args)
    gr.Markdown(
    """
    # Whisper to Normal Voice Demo   
    """
    )
    demo_inputs = gr.Audio(
            sources=["upload"],
            format="wav",
        )

    demo_outputs = gr.Audio(
            label="Trans-Wav",
            format="wav",
        )

    demo = gr.Interface(
        fn=process,
        inputs=demo_inputs,
        outputs=demo_outputs,
        title="Whisper Transfer Demo",
    )

    if __name__ == "__main__":
        demo.launch(share=False, server_name="0.0.0.0", server_port=80)




