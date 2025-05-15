from copy import deepcopy


class SATAPromptBuilder:
    def __init__(self, prompt_template: str, choices: dict, answer: str, dataset: str):
        """
        Initialize the MultiLabelPrompt object.
        Note the V1 prompt template is a basic version w/o prompt engineering because
        our goal is to focus on the SATA debias strategies.

        Args:
            prompt_template (str): input prompt template
            choices (dict): Dictionary of choices (e.g., {"A": "Paris", "B": "Berlin"})
            answer (str): Ground truth answers (e.g., "ACD")
            dataset (str): SATA-Bench subsdataset (e.g., "d3")
        """
        self.prompt_template = prompt_template
        self.choices = choices
        self.original_choices = deepcopy(choices)
        self.original_choices_vk = {str(v): k for k, v in choices.items()}
        self.original_answer = answer
        self.dataset = dataset

    def reset(self):
        self.choices = deepcopy(self.original_choices)

    def build_prompt(self, choices=None, additional_requirements=""):
        """
        Build the full prompt string by plugging in the choices (e.g., {"A": "Paris", "B": "Berlin"})

        Returns:
            str: The formatted prompt string.
        """
        if not choices:
            choices = self.choices

        formatted_choices = "\n".join(
            [f"{key}. {value}" for key, value in choices.items()]
        )
        prompt = self.prompt_template.format(choices=formatted_choices, additional_requirements=additional_requirements)
        return prompt

    def build_prompt_yesno(self, choice: str, additional_requirements=""):
        """
        Returns:
            str: The formatted prompt string.
        """
        prompt = self.prompt_template.format(choice=choice, additional_requirements=additional_requirements)
        return prompt

    def remove_choice(self, choice: str, rebalance: bool = True):
        """
        Remove a choice from the available choices.
        Useful for SATA choice by choice selection process.

        Args:
            rebalance (bool): whether to rebalance the options e.g., ACD(removed B) -> ABC
        """
        if choice not in self.choices:
            raise ValueError(f"choice: {choice} does not exist in the available choices")
        self.choices.pop(choice)

        if rebalance:
            new_choices = {}
            new_texts = self.choices.values()
            choice = "A"
            for text in new_texts:
                new_choices[choice] = text
                choice = chr(ord(choice) + 1)
            self.choices = new_choices

    def append_choice(self, choice: str):
        next_label = chr(ord(list(self.choices.keys())[-1]) + 1)
        self.choices[next_label] = choice

    def prepend_choice(self, choice: str):
        # Shift existing keys
        new_choices = {}
        for key, value in self.choices.items():
            new_label = chr(ord(key) + 1)
            if ord(new_label) > ord('Z'):
                raise ValueError("Cannot add more choices, keys limited to 'A-Z'")
            new_choices[new_label] = value
        new_choices['A'] = choice
        self.choices = dict(sorted(new_choices.items()))

    def get_original_choice_id(self, cur_id: str):
        text = self.choices[cur_id]
        return self.original_choices_vk[text]

    def generate_permutation_prompts(self, method="cyclic"):
        """
        Perform cyclic permutation of the choices and create N new versions of the prompt.
        Each version has permuted choices and adjusted answers.

        Returns:
            list[MultiLabelPrompt]: List of new MultiLabelPrompt objects with permuted choices and updated answers.
        """

        prompts_list = []
        if method == "cyclic":
            permutated_choices, permutated_answers = self._cycle_choices(self.choices, self.original_answer)
    
            for c, a in zip(permutated_choices, permutated_answers):
                prompts_list.append(SATAPromptBuilder(self.prompt_template, c, a, self.dataset))
    
            return prompts_list
        raise NotImplementedError()

    def _cycle_choices(self, choices, answer):
        texts = list(choices.values())
        n = len(texts)
        all_new_choices = []
        all_new_answers = []

        for i in range(n):
            new_texts = texts[i:] + texts[:i]
            new_choices = {}
            choice = "A"
            for text in new_texts:
                new_choices[choice] = text
                choice = chr(ord(choice) + 1)
            all_new_choices.append(new_choices)

            # shift the answers by i to get new answers
            new_answer = ""
            all_new_answers.append(answer)
            for label in answer:
                if label == 'A':
                    new_answer += chr(ord(label) + (n-1))
                else:
                    new_answer += chr(ord(label) -1)
            answer = new_answer

        return all_new_choices, all_new_answers

    def validate_prediction(self, prediction):
        """
        Validate a model's prediction against the ground truth answer.

        Args:
            prediction (str): The model's predicted answers (e.g., "ABC").

        Returns:
            bool: True if the prediction matches the ground truth answer, False otherwise.
        """
        return set(prediction) == set(self.original_answer)