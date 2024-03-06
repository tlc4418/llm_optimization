from torch.utils.data import Dataset


class RMPromptDataset(Dataset):
    def __init__(self, dataset: Dataset, output_alpaca: bool = False):
        super().__init__()
        self.data = dataset
        self.output_alpaca = output_alpaca

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[str, str, list[str], list[str]]:
        entry = self.data[index]
        start_prompt, answers, end_token = _parse_entry(entry, self.output_alpaca)
        rm_prompts = [f"{start_prompt}{answer}{end_token}" for answer in answers]
        return entry, rm_prompts


def _parse_entry(entry: dict, output_alpaca: bool = False):
    instruction = entry["instruction"]
    input = entry.get("input", "")
    answers = entry["answers"]
    if output_alpaca:
        input_ = f"### Input:\n{input}\n\n" if len(input) > 0 else ""
        start_prompt = "Below is an instruction that describes a task, paired with an "
        "input that provides further context. Write a response that appropriately "
        "completes the request.\n\n### Instruction:\n{instruction}\n\n{input_}### "
        "Response:\n"
        end_token = ""
    else:
        input_ = f"\n{input}" if len(input) > 0 else ""
        start_prompt = f"<|prompter|>{instruction}{input_}<|endoftext|><|assistant|>"
        end_token = "<|endoftext|>"
    return start_prompt, answers, end_token
