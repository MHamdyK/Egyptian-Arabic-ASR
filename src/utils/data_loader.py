import pandas as pd

def load_dataset(csv_path):
    """
    Load a CSV dataset containing at least two columns: 'path' (audio file path) and 'transcript' (text).
    Returns lists of file paths and transcripts.
    """
    df = pd.read_csv(csv_path)
    # Expect columns 'path' and 'transcript'
    audio_paths = df["path"].tolist()
    transcripts = df["transcript"].tolist()
    return audio_paths, transcripts
