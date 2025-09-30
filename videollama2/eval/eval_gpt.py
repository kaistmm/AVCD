import os
import ast
import json
import time
import argparse
import traceback
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI


def interaction(message_text):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=message_text
    )
    
    return response.choices[0].message.content.strip() #response

def prompt_gpt(question, answer, pred, key, qa_set, output_dir):
    message = [
        {
            "role": "system",
            "content":
                "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                "------"
                "##INSTRUCTIONS: "
                "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                "- Consider synonyms or paraphrases as valid matches.\n"
                "- Evaluate the correctness of the prediction compared to the answer."
        },
        {
            "role": "user",
            "content":
                "Please evaluate the following video-based question-answer pair:\n\n"
                f"Question: {question}\n"
                f"Correct Answer: {answer}\n"
                f"Predicted Answer: {pred}\n\n"
                "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                "For example, your response should look like this: {'pred': 'yes', 'score': 4}."
        }
    ]
    completion = interaction(message)
    response_message = completion
    response_dict = ast.literal_eval(response_message)
    result_qa_pair = [response_dict, qa_set]
    
    # Save the question-answer pairs to a JSON file.
    with open(f"{output_dir}/{key}.json", "w") as f:
        json.dump(result_qa_pair, f)

def annotate(task_arg):
    """
    Evaluates question and answer pairs using GPT-3
    Returns a score for correctness.
    """
    prediction_set, caption_files, output_dir, args = task_arg

    for file in tqdm(caption_files):
        key = int(file[:-5])  # Strip file extension
       
        qa_set = prediction_set[key]
        question = qa_set['q']
        answer = qa_set['a']
        pred = qa_set['p']

        try:
            prompt_gpt(question, answer, pred, key, qa_set, output_dir)
        except Exception as e:
            prompt_gpt(question, answer, pred[:50], key, qa_set, output_dir)
            traceback.print_exc()
    time.sleep(1)

def main(args):

    file = open(args.pred_path)
    
    new_pred_contents = [eval(i.strip()) for i in file.readlines()]

    id_list = [x['id'] for x in new_pred_contents]
    caption_files = [f"{id}.json" for id in id_list]

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Preparing dictionary of question-answer sets
    prediction_set = {}
    for sample in new_pred_contents:
        id = sample['id']
        question = sample['question']
        answer = sample['answer']
        pred = sample['pred']
        qa_set = {"q": question, "a": answer, "p": pred}
        prediction_set[id] = qa_set

        print(qa_set)


    num_tasks = args.num_tasks

    # While loop to ensure that all captions are processed.
    while True:
        try:
            # Files that have not been processed yet.
            completed_files = os.listdir(output_dir)
            print(f"completed_files: {len(completed_files)}")

            # Files that have not been processed yet.
            incomplete_files = [f for f in caption_files if f not in completed_files]
            print(f"incomplete_files: {len(incomplete_files)}")

            # Break the loop when there are no incomplete files
            if len(incomplete_files) == 0:
                break
            if len(incomplete_files) <= num_tasks:
                num_tasks = 1

            # Split tasks into parts.
            part_len = len(incomplete_files) // num_tasks
            all_parts = [incomplete_files[i:i + part_len] for i in range(0, len(incomplete_files), part_len)]
            task_args = [(prediction_set, part, args.output_dir, args) for part in all_parts]

            # Use a pool of workers to process the files in parallel.
            with ThreadPoolExecutor(max_workers=args.num_tasks) as executor:
                list(tqdm(executor.map(annotate, task_args), total=len(task_args)))

        except Exception as e:
            print(f"Error: {e}")

    # multiprocessing to combine json files
    def combine_json(file_name):
        file_path = os.path.join(output_dir, file_name)
        with open(file_path, "r") as json_file:
            content = json.load(json_file)
            return (file_name[:-5], content)

    files = os.listdir(output_dir)
    with ThreadPoolExecutor(max_workers=64) as executor:
        combined_contents = list(tqdm(executor.map(combine_json, files), total=len(files)))

    # Calculate average score and accuracy
    score_sum = 0
    count = 0
    yes_count = 0
    no_count = 0
    for key, result in tqdm(combined_contents):
        try:
            # Computing score
            count += 1
            score_match = result[0]['score']
            score = int(score_match)
            score_sum += score

            # Computing accuracy
            pred = result[0]['pred']
            if "yes" in pred.lower():
                yes_count += 1
            elif "no" in pred.lower():
                no_count += 1
        except:
            print(result)

    average_score = score_sum / count
    accuracy = yes_count / (yes_count + no_count)
    print("Yes count:", yes_count)
    print("No count:", no_count)
    print("Accuracy:", accuracy)
    print("Average score:", average_score)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
    parser.add_argument("--pred-path", required=False, default= "filepath.json")
    parser.add_argument("--output-dir", required=False, default="./dir/",help="The path to save annotation json files.")
    parser.add_argument("--num-tasks", required=False,default=8, type=int, help="Number of splits.")
    parser.add_argument("--api-key", required=True, type=str, help="API key.")
    args = parser.parse_args()
    client = OpenAI(
    api_key=args.api_key,
    )


    main(args)
