import os
import datetime
import json
import multiprocessing
from tqdm import tqdm

from args import parse_args
from libs.data_loader import load_dataset, extract_answer_yesno
from libs.utils import compare_answer_with_groundtruth
from prompt import sys_single_sol_prompt, pubmed_prompt_0shot, format_prompt_yesno
from agent import actor_agent

DATA_BATCH_SIZE = 1


def get_solve(
    temperature, max_tokens, rank, batched_input_data, log_dir, correct_count
):
    correct = 0

    os.makedirs(os.path.join(log_dir, "sol"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "correct"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "wrong"), exist_ok=True)

    for batch in tqdm(batched_input_data, desc=str(rank), position=rank):
        item = batch[0]
        question = item["question"]
        groundtruth = item["groundtruth"]
        context = item["context"]
        index = item["index"]

        sol_path = os.path.join(log_dir, "sol", f"{index}_sol.jsonl")
        correct_path = os.path.join(log_dir, "correct", f"{index}_correct.jsonl")
        wrong_path = os.path.join(log_dir, "wrong", f"{index}_wrong.jsonl")

        if os.path.exists(sol_path):
            print(f"Problem {index} already solved.")
            continue

        user_prompt = pubmed_prompt_0shot.format(
            context=context, question=question, format_prompt=format_prompt_yesno
        )

        single_log = actor_agent.call_agent(
            sys_prompt=sys_single_sol_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if single_log is None:
            print(f"Failed to get response for {index}, skipping.")
            continue

        actor_response = single_log["messages"][2]["content"]
        # Extract answer from last 15 chars (where "Answer: yes" format appears)
        answer = extract_answer_yesno(actor_response[-15:])
        score = compare_answer_with_groundtruth(answer, *groundtruth)

        info = item.copy()
        info["answer"] = answer
        info["single_log"] = single_log
        info["score"] = score

        json_line = json.dumps(info, ensure_ascii=False)
        with open(sol_path, "w", encoding="utf-8") as f_sol:
            f_sol.write(json_line + "\n")

        if score:
            correct += 1
            with open(correct_path, "w", encoding="utf-8") as f_correct:
                f_correct.write(json_line + "\n")
        else:
            print(f"Wrong: {index}")
            with open(wrong_path, "w", encoding="utf-8") as f_wrong:
                f_wrong.write(json_line + "\n")

    correct_count[rank] = correct


if __name__ == "__main__":
    args = parse_args()
    print(args)

    start_time = datetime.datetime.now()
    input_data = load_dataset(args)

    batched_data = [
        input_data[i : i + DATA_BATCH_SIZE]
        for i in range(0, len(input_data), DATA_BATCH_SIZE)
    ]

    log_dir = f"logs/solve_{args.subject}_{actor_agent.model}"
    os.makedirs(log_dir, exist_ok=True)
    print(f"Output dir: {log_dir}")
    print(f"Total problems: {len(batched_data)}")

    manager = multiprocessing.Manager()
    correct_count = manager.dict()

    processes = []
    for i in range(min(64, len(batched_data))):
        p = multiprocessing.Process(
            target=get_solve,
            args=(
                args.temperature,
                args.max_tokens,
                i,
                batched_data[i::64],
                log_dir,
                correct_count,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    total_correct = sum(correct_count.values())
    end_time = datetime.datetime.now()
    print(f"Total correct: {total_correct} / {len(batched_data)}")
    print(f"Time cost: {round((end_time - start_time).total_seconds() / 60, 2)} mins")
