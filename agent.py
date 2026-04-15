import json
import openai
from openai import OpenAI
import time
import datetime
from args import parse_args


class Agent:
    def __init__(self, api_type, name, model, next_agent, pre_agent):
        self.name = name
        self.model = model
        self.next_agent = next_agent
        self.pre_agent = pre_agent
        self.api_type = api_type
        self.client = api_type

    def call_agent(
        self, sys_prompt, user_prompt, temperature=0.0, max_tokens=5500, stop=None, n=1
    ):
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]
        client = self.client
        attempt = 0
        while attempt < 50:
            try:
                completion = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    stop=stop,
                    temperature=temperature,
                    n=n,
                )
                assistant_message = {
                    "role": "assistant",
                    "content": completion.choices[0].message.content,
                }
                messages.append(assistant_message)
                log = {"messages": messages}
                return log

            except openai.OpenAIError as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                attempt += 1
                if attempt < 50:
                    time.sleep(10)
                else:
                    raise
        return None

    def fine_tune(self, training_dataset_filename):
        client = self.client

        if isinstance(self.api_type, OpenAI):
            file_object = client.files.create(
                file=open(training_dataset_filename, "rb"), purpose="fine-tune"
            )
            file_id = file_object.id
            print(f"File uploaded, ID: {file_id}")

            print(f"Polling file status until processed...")
            max_wait = 300
            waited = 0
            while waited < max_wait:
                file_status = client.files.retrieve(file_id)
                if file_status.status == "processed":
                    print(f"File {file_id} is ready.")
                    break
                print(
                    f"  status={file_status.status}, waited {waited}s, retrying in 10s..."
                )
                time.sleep(10)
                waited += 10
            else:
                raise RuntimeError(
                    f"File {file_id} not processed after {max_wait}s. Aborting."
                )

            # Extra buffer: OpenAI's storage sometimes reports "processed" before
            # the file is fully propagated across their internal network.
            # 30s wait eliminates the "trouble accessing your files" race condition.
            print(f"Waiting 30s for file to fully propagate before submitting job...")
            time.sleep(30)

            fine_tuned_object = client.fine_tuning.jobs.create(
                training_file=file_id, model=self.model
            )

        return fine_tuned_object

    # def fine_tune(self, training_dataset_filename):
    #     client = self.client

    #     if isinstance(self.api_type, OpenAI):
    #         file_object = client.files.create(
    #             file=open(training_dataset_filename, "rb"), purpose="fine-tune"
    #         )

    #         file_id = file_object.id
    #         print(f"File uploaded, ID: {file_id}")

    #         # Wait for OpenAI's backend to process the file before starting the job.
    #         # Without this, you get "trouble accessing your files right now" server_error.
    #         print("Waiting 15 seconds for file to be processed by OpenAI...")
    #         time.sleep(15)

    #         fine_tuned_object = client.fine_tuning.jobs.create(
    #             training_file=file_id, model=self.model
    #         )
    #     return fine_tuned_object


# Initialize agents
args = parse_args()
api_type = OpenAI()
model = args.model


actor_agent = Agent(
    name="actor", model=model, next_agent=None, pre_agent=None, api_type=api_type
)

judge_agent = Agent(
    name="judge", model=model, next_agent=None, pre_agent=None, api_type=api_type
)

critic_agent = Agent(
    name="critic", model=model, next_agent=None, pre_agent=None, api_type=api_type
)
