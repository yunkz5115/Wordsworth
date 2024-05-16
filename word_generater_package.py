import os
import numpy as np
from google.cloud import texttospeech
from google.cloud import texttospeech_v1 # version 1
from scipy.io import wavfile as WF
from scipy.io.wavfile import write

# set up environment: provide service aaccount key/credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './healthy-result-378719-3d2d70e5dde1.json'

# Instantiates a client
client = texttospeech_v1.TextToSpeechClient()

voice = {
    'language_code':'en-US',
    'name':'en-US-Wavenet-J'
    }

audio_config = {
      'audio_encoding':'LINEAR16',
      'speaking_rate':1
    }



def word_generater(text4synth,speed=1,name="en-US-Wavenet-J",language_code = "en-US",folder=''):
    audio_config['speaking_rate']=speed
    voice['language_code']=language_code
    voice['name']=name
    input_text = texttospeech.SynthesisInput(text=text4synth)
    response = client.synthesize_speech(
        request={
            "input": input_text, 
            "voice": voice, 
            "audio_config": audio_config
            }
        )
    with open(folder+text4synth+'_speed_'+str(speed)+'_'+name+'_'+'.wav', "wb") as output:
        # Write the response to the output file.
        output.write(response.audio_content)
        print('Word',text4synth,' content written to file',text4synth+'.wav',' Type: ',name)
        
    
def wave_normolize(load_folder,save_folder,text4synth,speed,name="en-US-Wavenet-J"):
    sample_rate, sig = WF.read(load_folder+text4synth+'_speed_'+str(speed)+'_'+name+'_'+'.wav')
    sig_n = sig/np.max(np.abs(sig))
    write(save_folder+text4synth+'_speed_'+str(speed)+'_'+name+'_'+'.wav', sample_rate, sig_n)
    
    
def list_voices():
    """Lists the available voices."""
    from google.cloud import texttospeech

    client = texttospeech.TextToSpeechClient()

    # Performs the list voices request
    voices = client.list_voices()
    
    voice_list = []
    code_list = []

    for voice in voices.voices:
        # Display the voice's name. Example: tpc-vocoded
        #print(f"Name: {voice.name}")

        # Display the supported language codes for this voice. Example: "en-US"
        #for language_code in voice.language_codes:
            #print(f"Supported language: {language_code}")

        #ssml_gender = texttospeech.SsmlVoiceGender(voice.ssml_gender)

        # Display the SSML Voice Gender
        #print(f"SSML Voice Gender: {ssml_gender.name}")

        # Display the natural sample rate hertz for this voice. Example: 24000
        #print(f"Natural Sample Rate Hertz: {voice.natural_sample_rate_hertz}\n")
        
        voice_list = voice_list + [voice.name]
        code_list = code_list + [voice.language_codes]
    
    voice_list = np.array(voice_list)
    code_list = np.array(code_list)[:,0]
        
    return voice_list,code_list


def wave_length_confund_normolize(load_folder,save_folder,text4synth,speed,name="en-US-Wavenet-J",target_length=1):
    sample_rate, sig = WF.read(load_folder+text4synth+'_speed_'+str(speed)+'_'+name+'_'+'.wav')
    sig_n = sig/np.max(np.abs(sig))
    sig_length = len(sig_n)/sample_rate
    if len(sig_n)<target_length*sample_rate:
        sig_n = np.hstack([sig_n,np.zeros(int(target_length*sample_rate)-len(sig_n))])
    write(save_folder+text4synth+'_speed_'+str(speed)+'_'+name+'_'+'.wav', sample_rate, sig_n)
    try:
        os.remove(load_folder+text4synth+'_speed_'+str(speed)+'_'+name+'_'+'.wav')
    except OSError as error:
        print(f"Error deleting the file: {error}")
        
    return sig_length
            
    