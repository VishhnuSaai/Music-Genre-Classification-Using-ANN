import streamlit as st
from pydub import AudioSegment
import make_dataset
from tensorflow import keras
from keras.models import load_model
import os
import numpy as np
import subprocess

def convert_mp3_to_wav(input_file_path, output_file_path):
    ffmpeg_path = 'ffmpeg.exe'  
    command = [ffmpeg_path, '-i', input_file_path, '-y', output_file_path]
    subprocess.run(command)

def convert_aac_to_wav(input_file_path, output_file_path):
    ffmpeg_path = 'ffmpeg.exe'  
    command = [ffmpeg_path, '-i', input_file_path, '-y', output_file_path]
    subprocess.run(command)

def convert_wma_to_wav(input_file_path, output_file_path):
    ffmpeg_path = 'ffmpeg.exe'  
    command = [ffmpeg_path, '-i', input_file_path, '-y', output_file_path]
    subprocess.run(command)
genres = {
    'metal': 0, 'disco': 1, 'classical': 2, 'hiphop': 3, 'jazz': 4, 
    'country': 5, 'pop': 6, 'blues': 7, 'reggae': 8, 'rock': 9
}

def majority_voting(scores, dict_genres):
    preds = np.argmax(scores, axis=1)
    values, counts = np.unique(preds, return_counts=True)
    counts = np.round(counts / np.sum(counts), 2)
    votes = {k: v for k, v in zip(values, counts)}
    votes = {k: v for k, v in sorted(votes.items(), key=lambda item: item[1], reverse=True)}
    return [(get_genres(x, dict_genres), prob) for x, prob in votes.items()]

def get_genres(key, dict_genres):
    labels = []
    tmp_genre = {v: k for k, v in dict_genres.items()}
    return tmp_genre[key]

def run_app(song_path):
    x = make_dataset.make_dataset_dl(song_path)
    
    if x is None:
        print("Error: Unable to load data. Skipping prediction.")
        return None

    # Load the model
    model = load_model("custom_cnn_2d.h5")

    if model is None:
        print("Error: Model not loaded successfully. Skipping prediction.")
        return None

    # Reshape input to match the expected input shape of the model
    x = np.expand_dims(x, axis=-1)  # Add the channel dimension

    # Make predictions
    preds = model.predict(x)

    if preds is None:
        print("Error: Model prediction is None. Skipping prediction.")
        return None

    # Process predictions as needed
    votes = majority_voting(preds, genres)
    print("{} is a {} song".format(song_path, votes[0][0]))
    print("Most likely genres are: {}".format(votes[:3]))
    return votes[0][0]

def main():
    st.title("Music Genre Classification")
    st.write("Upload an Audio file and we will predict this Genre.")

    # First st.file_uploader widget
    song_path1 = st.file_uploader("Choose an Audio file (.wav)", key="uploader1", type=["wav"])

    os.makedirs("temp", exist_ok=True)

    if song_path1 is not None:
        file_path1 = os.path.join("temp", song_path1.name)
        
        with open(file_path1, "wb") as f:
            f.write(song_path1.getbuffer())

        genre1 = run_app(file_path1)
        st.audio(file_path1, format="audio/mp3")

        st.success("The Predicted genre for your audio file is: {}".format(genre1))
    
    # Second st.file_uploader widget
    song_path2 = st.file_uploader("Choose an Audio file (.mp3)", key="uploader2", type=["mp3"])

    os.makedirs("temp2", exist_ok=True)

    if song_path2 is not None:
        file_path2 = os.path.join("temp2", song_path2.name)
        print(file_path2)
        
        with open(file_path2, "wb") as f:
            f.write(song_path2.getbuffer())
        convert_mp3_to_wav(file_path2, 'temp2/output.wav')

        genre2 = run_app('temp2/output.wav')
        st.audio(file_path2, format="audio/mp3")

        st.success("The Predicted genre for your audio file is: {}".format(genre2))
    
    # Third st.file_uploader widget
    song_path3 = st.file_uploader("Choose an Audio file (.aac)", key="uploader3", type=["aac"])
    os.makedirs("temp3", exist_ok=True)

    if song_path3 is not None:
        file_path3 = os.path.join("temp3", song_path3.name)
        
        with open(file_path3, "wb") as f:
            f.write(song_path3.getbuffer())
        convert_aac_to_wav(file_path3,'temp3/output.wav')

        genre3 = run_app('temp3/output.wav')
        st.audio(file_path3, format="audio/aac")

        st.success("The Predicted genre for your audio file is: {}".format(genre3))
    
     # Fourth st.file_uploader widget
    song_path4 = st.file_uploader("Choose an Audio file (.wma)", key="uploader4", type=["wma"])
    os.makedirs("temp4", exist_ok=True)

    if song_path4 is not None:
        file_path4 = os.path.join("temp4", song_path4.name)
        
        with open(file_path4, "wb") as f:
            f.write(song_path4.getbuffer())
        convert_wma_to_wav(file_path4,'temp4/output.wav')

        genre4 = run_app('temp4/output.wav')
        st.audio(file_path4, format="audio/aac")

        st.success("The Predicted genre for your audio file is: {}".format(genre4))
    

    

if __name__ == "__main__":
    main()
