import argparse


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--n_generate_sample", type=int, default=1)
    args.add_argument("--temperature", type=float, default=0.0)
    args.add_argument("--max_tokens", type=int, default=4096)

    args.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo-0125",
        choices=[
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0125",   # exact snapshot required by fine-tuning API
            "gpt-3.5-turbo-1106",   # exact snapshot required by fine-tuning API
            "gpt-4o-mini-2024-07-18",
            "gpt-4.1-mini-2025-04-14",
            "gpt-4.1-nano-2025-04-14",
            "gpt-4o-2024-08-06",
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Reference",
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        ],
    )
    args.add_argument(
        "--prompt_type",
        type=str,
        default="multi_agent",
        choices=["single_agent_zeroshot", "multi_agent"],
    )

    args.add_argument("--ft_round", type=int, default=0)
    args.add_argument("--sol_round", type=int, default=0)
    args.add_argument("--fd_round", type=int, default=0)
    args.add_argument("--re_round", type=int, default=0)

    args.add_argument(
        "--input_file", type=str, default="", help="Path to input JSONL file"
    )
    args.add_argument("--mode", type=str, default="eval", choices=["eval", "generate"])
    args.add_argument("--subject", type=str, default="PubMedQA", choices=["PubMedQA"])
    args.add_argument(
        "--only_agents",
        type=str,
        default="",
        help="Comma-separated list of agents to fine-tune (e.g. verifier_a,verifier_b). "
             "If empty, all agents are submitted.",
    )

    args = args.parse_args()

    return args
