import jiwer

def calculate_metrics(ground_truth, predicted):
    # Normalize text: remove newlines, extra spaces, and lowercase
    gt = " ".join(ground_truth.lower().split())
    pred = " ".join(predicted.lower().split())
    
    if not gt: return 0, 0 # Handle empty ground truth
    
    cer = jiwer.cer(gt, pred)
    wer = jiwer.wer(gt, pred)
    
    return cer, wer