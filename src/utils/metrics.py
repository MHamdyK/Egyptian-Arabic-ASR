from jiwer import wer

def compute_wer(predictions, references):
    """
    Compute the Word Error Rate (WER) between two lists of strings.
    Returns a float (percentage).
    """
    return wer(references, predictions) * 100.0
