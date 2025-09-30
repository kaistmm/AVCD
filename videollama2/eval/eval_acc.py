from difflib import SequenceMatcher
import json
import glob
import os

import ast
import time
import argparse
import traceback
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed



def main(args):
    file= args.pred_path
    correct_count = 0
    total_questions =0
    
    print("file", file)
    file = open(file)
    new_pred_contents = [eval(i.strip()) for i in file.readlines()]

    for number, entry in enumerate(new_pred_contents):
        answer = entry["answer"].strip().lower()
        pred = entry["pred"].strip().lower()

        if "yes" in pred or "no" in pred:
            if answer.lower() in pred.lower():
                correct_count+=1
        
    total_questions += len(new_pred_contents)
    print("total question", total_questions)

    # Calculate accuracy
    accuracy = (correct_count) / total_questions
    print("correct num", correct_count)
    print("ACC is ", accuracy)







if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
    parser.add_argument("--pred-path", required=False, default="AVH_validation.json")
    args = parser.parse_args()
  
    main(args)