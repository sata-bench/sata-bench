from typing import List

from satabench.methods.utils.metrics import *
from satabench.methods.utils.debiasing import *
from satabench.methods.choice_funnel.sata_model_runner import SATAModelRunner
from satabench.methods.choice_funnel.sata_prompt_builder import SATAPromptBuilder
from satabench.methods.choice_funnel.sata_prompt_provider import prompt_provider


class SATAScoringExecutor:
    """
    SATAJobExecutor is the main orchestrator for making
    predictions for SATA datasets in SATAPromptBuilder format
    """
    def __init__(self,
                 prompt_list: List[SATAPromptBuilder],
                 model: SATAModelRunner):
        self.prompt_list = prompt_list
        self.model = model

        # states
        self.all_observed_probs = []
        self.all_observed_logits = []
        self.all_priors = []
        self.all_priors_dict = {}
        self.predictions = []
        self.prediction_probs = []
        self.answers = []
        self.num_choices = []
        self.dataset_used = []
        self.final_prior_dict = {}
        self.is_predictions_thresholded = False

        # metrics
        self.total_permutation = 0
        self.prior_miss = 0
        self.inf_count = 0
        self.break_on_idk = 0
        self.break_on_prob_drop = 0
        self.break_on_exhaustion = 0
        self.break_on_prob_limit = 0

    def reset(self):
        self.all_priors = []
        self.all_priors_dict = {}
        self.prior_miss = 0
        self.inf_count = 0
        self.predictions = []
        self.prediction_probs = []
        self.answers = []
        self.num_choices = []
        self.dataset_used = []
        self.final_prior_dict = {}
        self.is_predictions_thresholded = False
        self.break_on_idk = 0
        self.break_on_prob_drop = 0
        self.break_on_exhaustion = 0
        self.break_on_prob_limit = 0

        for p in self.prompt_list:
            p.reset()

    def run_inference(self, prefix_sample_ratio: float=1.0):
        num_sample_for_prior = int(len(self.prompt_list) * prefix_sample_ratio)
        for prompt in self.prompt_list[:num_sample_for_prior]:
            observed_probs = []
            permutated_prompts = prompt.generate_permutation_prompts()

            for p in permutated_prompts:
                model_input = p.build_prompt()
                num_options = len(p.choices)

                sorted_option_prob_dict, prediction = self.model.generate(model_input, num_options)
                probs = np.array([sorted_option_prob_dict.get(option) for option in list(p.choices.keys())])
                observed_probs.append(probs)

            self.all_observed_probs.append(np.array(observed_probs))
            self.total_permutation += len(permutated_prompts)

    # Baseline Method #1: "first token" in SATA-Bench Paper Table4
    def baseline_scoring(self):
        self.reset()

        predictions = []
        answers = []
        num_choices = []
        dataset_used = []
        for idx, observed_probs in enumerate(self.all_observed_probs):
            prompt = self.prompt_list[idx]
            original_choices = list(prompt.original_choices.keys())
            predicted_labels = [(label,prob) for prob, label in sorted(zip(observed_probs[0], original_choices), reverse=True)]
            predictions.append(predicted_labels)
            answers.append(prompt.original_answer)
            num_choices.append(len(prompt.original_choices))
            dataset_used.append(prompt.dataset)

            self.inf_count += 1

        self.predictions = predictions
        self.answers = answers
        self.num_choices = num_choices
        self.dataset_used = dataset_used

    # Baseline Method #2: "first token debiasing" in SATA-Bench Paper Table4
    def debias_scoring(self, prefix_sample_ratio: float):
        self.reset()

        # compute prior for first prefix_sample_ratio samples
        len_prior_dict = defaultdict(list)
        num_sample_for_prior = int(len(self.prompt_list) * prefix_sample_ratio)
        for idx, observed_probs in enumerate(self.all_observed_probs[:num_sample_for_prior]):
            # cyclic permutation debias
            observed, debiased, prior = compute_prior_token_bias(observed_probs)
            self.all_priors.append(prior)
            self.inf_count += len(observed)

        # for SATA setting we need to compute prior for each sub lengths to make
        # sure each mean prior are computed from the same probability distribution
        for prior in self.all_priors:
            n = len(prior)
            for i in range(2, n + 1):
                prefix = prior[:i]
                normalized_prefix = prefix / prefix.sum()
                len_prior_dict[i].append(normalized_prefix)
        self.final_prior_dict = {k: np.mean(np.array(v), axis=0) for k, v in len_prior_dict.items()}

        # scoring using prior for debiasing for rest of samples
        for idx, observed_probs in enumerate(self.all_observed_probs):
            observed_probs = np.array(observed_probs[0])
            if len(observed_probs) in self.final_prior_dict:
                prior = self.final_prior_dict[len(observed_probs)]
                # PriDe debias
                debiased = np.log(observed_probs + 1e-10) - np.log(prior + 1e-10)
            else:
                self.prior_miss += 1
                debias = observed_probs[0]

            prompt = self.prompt_list[idx]
            original_choices = list(prompt.original_choices.keys())
            predicted_labels = [(label, prob) for prob, label in sorted(zip(debiased, original_choices), reverse=True)]
            self.predictions.append(predicted_labels)
            self.answers.append(prompt.original_answer)
            self.num_choices.append(len(prompt.original_choices))
            self.dataset_used.append(prompt.dataset)
            self.inf_count += 1

    # Baseline Method #3: "yes/no" in SATA-Bench Paper Table4
    def yesno_scoring(self, confidence_threshold: float=0.5):
        self.reset()

        predictions = []
        answers = []
        for p in self.prompt_list:
            predicted_labels = []
            for choice_id, choice in p.original_choices.items():
                try:
                    model_input = p.build_prompt_yesno(choice)
                except Exception:
                    print(choice)
                    print(p.prompt_template)
                is_correct_choice, confidence = self.model.generate_yesno(model_input)
                if is_correct_choice and confidence >= confidence_threshold:
                    predicted_labels.append(choice_id)
                    # debug: all_option_probs.append(option_probs)
                self.inf_count += 1

            predictions.append("".join(predicted_labels))
            # bebug: predictions.append(("".join(predicted_labels), all_option_probs))
            answers.append(p.original_answer)
            self.num_choices.append(len(p.original_choices))
            self.dataset_used.append(p.dataset)

        self.predictions = predictions
        self.answers = answers
        self.is_predictions_thresholded = True # for metrics computation

    # Proposed method "Choice Funnel" in SATA-Bench Paper Table4
    def debias_scoring_cbc_thresholding(self,
                                        prefix_sample_ratio: float,
                                        idk_choice: str="I don't know",
                                        additional_requirements: str="",
                                        do_debias: bool=True,
                                        break_on_prob_drop: bool=True,
                                        break_on_prob_drop_range: float= 0.0,
                                        break_on_prob_limit: float = None,
                                        break_on_num_gt_labels: bool=False,
                                        run_until_exhaustion: bool=False):
        """
        Choice Funnel algorithm:
        For each question we iteratively select one choice at a time until:
            1. model selects "idk_choice", or
            2. current pick's log prob is higher than break_on_prob_limit (90% in paper)
        """
        self.reset()

        # compute prior using first prefix_sample_ratio (10% in paper) of samples
        len_prior_dict = defaultdict(list)
        num_sample_for_prior = int(len(self.prompt_list) * prefix_sample_ratio)
        for _, observed_probs in enumerate(self.all_observed_probs[:num_sample_for_prior]):
            # cyclic permutation debias
            observed, debiased, prior = compute_prior_token_bias(observed_probs)
            self.all_priors.append(prior)
            self.inf_count += len(observed)

        # for SATA setting we need to compute prior for each sub lengths to make
        # sure each mean prior are computed from the same probability distribution
        for prior in self.all_priors:
            n = len(prior)
            for i in range(2, n + 1):
                prefix = prior[:i]
                normalized_prefix = prefix / prefix.sum()
                len_prior_dict[i].append(normalized_prefix)
        self.final_prior_dict = {k: np.mean(np.array(v), axis=0) for k, v in len_prior_dict.items()}

        for p in self.prompt_list:
            prediction = []
            prediction_prob = []

            picked_first = False
            prev_prob = 0.0
            prev_selected_choices = []

            # Choice Funnel iterative selection starts
            while True:
                if not p.choices:
                    self.break_on_exhaustion += 1
                    break

                if not picked_first:
                    model_input = p.build_prompt()
                else:
                    cur_additional_requirements = additional_requirements
                    model_input = p.build_prompt(additional_requirements=cur_additional_requirements)

                num_options = len(p.choices)
                sorted_option_prob_dict, choices = self.model.generate(model_input,
                                                                       num_options)
                self.inf_count += 1
                observed_probs = np.array([sorted_option_prob_dict.get(option) for option in list(p.choices.keys())])

                # Remove token debias
                if do_debias and len(observed_probs) in self.final_prior_dict:
                    prior = self.final_prior_dict[len(observed_probs)]
                    # PriDe debias
                    debiased = np.log(observed_probs + 1e-10) - np.log(prior + 1e-10)
                    cur_choice = (list(p.choices.keys())[np.argmax(debiased)], np.max(debiased)) # (debias_option_id, debiased_prob)
                    original_max_prob = max(observed_probs)
                else:
                    self.prior_miss += 1
                    cur_choice = next(iter(sorted_option_prob_dict.items()))
                    original_max_prob = cur_choice[1]
                prediction_prob.append(original_max_prob)

                # Stopping Condition #1: model selects "idk_choice"
                if not break_on_num_gt_labels and idk_choice:
                    idk_choice_id = next(iter(reversed(p.choices.items())))[0]
                    # stop if model picks "I don't know"
                    if picked_first and cur_choice[0] == idk_choice_id:
                        self.break_on_idk += 1
                        if not run_until_exhaustion:
                            break

                # debugging only
                if not break_on_num_gt_labels and break_on_prob_drop and cur_choice[1] + break_on_prob_drop_range < prev_prob:
                    self.break_on_prob_drop += 1
                    if not run_until_exhaustion:
                        break

                original_choice = p.get_original_choice_id(cur_choice[0])
                prediction.append(original_choice)
                prev_selected_choices.append(p.choices[cur_choice[0]])
                if break_on_num_gt_labels and len(prev_selected_choices) == len(p.original_answer):
                    break

                p.remove_choice(cur_choice[0], rebalance = True)
                if not p.choices:
                    self.break_on_exhaustion += 1
                    break
                if not break_on_num_gt_labels and not picked_first and idk_choice:
                    p.append_choice(idk_choice)

                # Stopping Condition #2: current pick's log prob is higher than break_on_prob_limit (90% in paper)
                if not break_on_num_gt_labels and break_on_prob_limit and picked_first and cur_choice[1] < break_on_prob_limit:
                    if not run_until_exhaustion: # debugging only
                        self.break_on_prob_limit += 1
                        break

                picked_first = True
                prev_prob = original_max_prob

            self.predictions.append("".join(prediction))
            self.prediction_probs.append(prediction_prob)
            self.answers.append(p.original_answer)
            self.num_choices.append(len(p.original_choices))
            self.dataset_used.append(p.dataset)
            self.is_predictions_thresholded = True

    def compute_final_metrics_by_dataset(self, first_token_threshold: float=None, skip_empty_prediction: bool=False):
        for dataset in ["d1", "d2", "d3", "d4", "d5", "d6"]: 
            print(f"## Compute metrics for sub-dataset:{dataset}##")
            self.compute_final_metrics(first_token_threshold, skip_empty_prediction, dataset)
        self.compute_final_metrics(first_token_threshold, skip_empty_prediction)

    def compute_final_metrics(self, first_token_threshold: float=None, skip_empty_prediction: bool=False, select_dataset_used: "str"=None):
        total_count = 0
        no_pred_count = 0
        exact_match_count = []
        length_exact_match_count = []
        jaccard_acc = []
        recall_acc = []
        precision_acc = []
        f1_acc = []
        num_choices_acc = []
        hamming_acc = []
        length_dif_acc = []
        length_std_acc = []
        length_abs_acc = []
        num_predicted_label_count = 0
        num_correct_label_count = 0
        
        # Prepare for RCKLD
        all_preds = []
        all_labels = []
        per_label_recalls = {choice: [] for choice in "ABCDEFGHIJKLMNO"}  # Track recalls for RSTD


        for pred, label, num_choices, dataset_used in zip(self.predictions, self.answers, self.num_choices, self.dataset_used):
            if select_dataset_used and select_dataset_used != dataset_used:
                continue
            answer_set = set(label)
            answer_length = len(label)
            num_correct_label_count += answer_length
    
            # Split into prediction and probability if using tuples
            probs = []
            preds = []
            if isinstance(pred, list):
                for p in pred:
                    preds.append(p[0])
                    probs.append(p[1])
    
            if not self.is_predictions_thresholded:
                if first_token_threshold and probs:
                    pred_set = set()
                    for idx, p in enumerate(preds):
                        if probs[idx] >= first_token_threshold:
                            pred_set.add(p)
                else:
                    # Extract the prefix of the same length as 'answer'
                    pred_set = set(pred[:answer_length])
            else:
                pred_set = set(pred)
    
            # Convert to string for new metrics
            pred_str = "".join(pred_set)
            label_str = "".join(answer_set)
            all_preds.append(pred_str)
            all_labels.append(label_str)
    
            # Update prediction count
            num_predicted_label_count += len(pred_set)
            if not pred_set and skip_empty_prediction:
                no_pred_count += 1
                total_count += 1
                continue
    
            # Compute core metrics
            exact_match = 1 if answer_set == pred_set else 0
            length_exact_match = 1 if len(answer_set) == len(pred_set) else 0
            jaccard_index = len(answer_set & pred_set) / len(answer_set | pred_set) if len(answer_set | pred_set) > 0 else 0
            recall = len(answer_set & pred_set) / len(answer_set) if len(answer_set) > 0 else 0
            precision = len(answer_set & pred_set) / (len(pred_set) + 1e-12)  # avoid division by zero
            f1_score = (2 * precision * recall) / (precision + recall + 1e-12)  # avoid division by zero
    
            # Compute new metrics
            hamming = hamming_score([pred_str], [label_str])
            length_dif = len(pred_str) - len(label_str)
            length_std = length_dif ** 2
            length_abs = abs(length_dif)

            # Collect per-label recalls for RSTD
            for choice in "ABCDEFGHIJKLMNO":
                if choice in label_str:
                    if choice in pred_str:
                        per_label_recalls[choice].append(1)  # Correct recall
                    else:
                        per_label_recalls[choice].append(0)  # Missed recall
    
            # Append to accumulators
            exact_match_count.append(exact_match)
            length_exact_match_count.append(length_exact_match)
            jaccard_acc.append(jaccard_index)
            recall_acc.append(recall)
            precision_acc.append(precision)
            f1_acc.append(f1_score)
            hamming_acc.append(hamming)
            length_dif_acc.append(length_dif)
            length_std_acc.append(length_std)
            length_abs_acc.append(length_abs)
            num_choices_acc.append(num_choices)
            total_count += 1
    
        # Compute CKLD
        rckld_val = rckld(all_preds, all_labels)

        # Compute RSTD
        recalls = []
        for choice, recall_list in per_label_recalls.items():
            if recall_list:
                recall_mean = np.mean(recall_list)  # Mean recall for the choice
                recalls.append(recall_mean * 100)
        rstd_val = np.std(recalls) if recalls else -1  # Avoid error if no recalls

        # Compute RSD
        rsd_acc, rsd_f1 = rsd(all_preds, all_labels)

        print(f"""
        Metrics Summary:
        EM (Exact Match): {(float(sum(exact_match_count)) / float(total_count)):.6f} ↑
        Precision: {sum(precision_acc)*100 / total_count:.2f}% ↑
        Recall: {sum(recall_acc)*100 / total_count:.2f}% ↑
        JI (Jaccard Index): {sum(hamming_acc)*100 / total_count:.2f}% ↑
        SPD: {rckld_val:.2f} ↓
        RStd: {rstd_val:.2f} ↓
        RSD: {rsd_acc:.2f} ↓
        CtDif (Count Difference): {np.mean(length_dif_acc):.2f}
        CtDifAbs (Absolute Count Difference): {np.mean(length_abs_acc):.2f} ↓
        CtAcc (Count Accuracy): {sum(length_exact_match_count) / total_count:.2f} ↑
        InfCost (Inference Cost): {self.inf_count} ↓
        """)


if __name__ == "__main__":
    import os
    import random
    import time
    
    print("Starting SATA-Bench evaluation...")
    
    random.seed(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Get start time for performance tracking
    start_time = time.time()

    # HF Model Configs
    model_names = ["mistralai/Ministral-8B-Instruct-2410",      #0
                    "bigscience/bloomz-7b1",                    #1
                    "microsoft/Phi-3-small-8k-instruct",        #2
                    "meta-llama/Meta-Llama-3-8B-instruct",      #3
                    "google/gemma-7b-it",                       #4
                    "Qwen/Qwen2.5-14B-Instruct",                #5
                    "microsoft/Phi-4-mini-reasoning",           #6
                    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"] #7
    # Depends on how tokenizer is designed, generated option 
    # will have a prefix before the option_ids e.g. "ĠA"
    label_prefix = ["Ġ", "Ġ", b" ", "Ġ", "", "Ġ", "Ġ", "Ġ"]
    model_name = os.getenv('MODEL_NAME')
    
    print(f"Evaluating model: {model_name}")
    
    model_idx = model_names.index(model_name)
    # phi3 (idx 2) need to encode to bytes
    encode_option = False if model_idx != 2 else True

    print(f"Loading model {model_name}...")
    # Loading model
    model = SATAModelRunner(model_names[model_idx],
                            option_prefix=label_prefix[model_idx],
                            encode_option=encode_option)
    print(f"Model loaded successfully.")

    method = os.getenv('METHOD')
    print(f"Using evaluation method: {method}")
    
    if method == "choice_funnel":
        # Loading SATA-Bench dataset
        print("Loading SATA-Bench dataset for choice funnel evaluation...")
        sata_promptv1_list = prompt_provider("choice_funnel")
        sample_count = os.environ.get("SAMPLE_COUNT")
        if sample_count:
            sample_count = min(int(sample_count), len(sata_promptv1_list))
            sata_promptv1_list = random.sample(sata_promptv1_list, k=sample_count)
            print(f"Sampled {sample_count} examples from dataset.")
        else:
            print(f"Using full dataset with {len(sata_promptv1_list)} examples.")

        # Initializing scoring executior
        print("Initializing scoring executor...")
        job_executor = SATAScoringExecutor(sata_promptv1_list, model)

        # Here we select 10% to compute first token prior bias
        # following PriDe algorithm arxiv.org/pdf/2309.03882
        print("Running inference with 10% prefix sampling ratio...")
        job_executor.run_inference(prefix_sample_ratio = 0.1)
        
        print("Applying debiasing with CBC thresholding...")
        job_executor.debias_scoring_cbc_thresholding(prefix_sample_ratio = 0.1,
                                                    idk_choice="None of the above",
                                                    additional_requirements="",
                                                    break_on_prob_drop=False,
                                                    do_debias=True,
                                                    break_on_prob_drop_range=0.0,
                                                    break_on_prob_limit=0.9,
                                                    run_until_exhaustion=False)
        print("Computing final metrics...")
        job_executor.compute_final_metrics()

    elif method == "first_token":
        print("Loading SATA-Bench dataset for first token evaluation...")
        sata_promptv1_list = prompt_provider("choice_funnel")
        sample_count = os.environ.get("SAMPLE_COUNT")
        if sample_count:
            sample_count = min(int(sample_count), len(sata_promptv1_list))
            sata_promptv1_list = random.sample(sata_promptv1_list, k=sample_count)
            print(f"Sampled {sample_count} examples from dataset.")
        
        print("Initializing scoring executor...")
        job_executor = SATAScoringExecutor(sata_promptv1_list, model)
        
        print("Running inference...")
        job_executor.run_inference(1.0)
        
        print("Applying baseline scoring...")
        job_executor.baseline_scoring()
        
        print("Computing final metrics...")
        job_executor.compute_final_metrics(first_token_threshold=0.1)

    elif method == "first_token_debiasing":
        print("Loading SATA-Bench dataset for first token debiasing evaluation...")
        sata_promptv1_list = prompt_provider("choice_funnel")
        sample_count = os.environ.get("SAMPLE_COUNT")
        if sample_count:
            sample_count = min(int(sample_count), len(sata_promptv1_list))
            sata_promptv1_list = random.sample(sata_promptv1_list, k=sample_count)
            print(f"Sampled {sample_count} examples from dataset.")
        
        print("Initializing scoring executor...")
        job_executor = SATAScoringExecutor(sata_promptv1_list, model)
        
        print("Running inference...")
        job_executor.run_inference(1.0)
        
        print("Applying debiasing...")
        job_executor.debias_scoring(prefix_sample_ratio = 0.1)
        
        print("Computing final metrics...")
        job_executor.compute_final_metrics(first_token_threshold=0.1)

    elif method == "yesno":
        print("Loading SATA-Bench dataset for yes/no evaluation...")
        sata_promptv1_list = prompt_provider("yesno")
        sample_count = os.environ.get("SAMPLE_COUNT")
        if sample_count:
            sample_count = min(int(sample_count), len(sata_promptv1_list))
            sata_promptv1_list = random.sample(sata_promptv1_list, k=sample_count)
            print(f"Sampled {sample_count} examples from dataset.")
        
        print("Initializing scoring executor...")
        job_executor = SATAScoringExecutor(sata_promptv1_list, model)
        
        print("Applying yes/no scoring...")
        job_executor.yesno_scoring(confidence_threshold=0.0)
        
        print("Computing final metrics...")
        job_executor.compute_final_metrics()
    
    # Calculate and print execution time
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\nEvaluation completed!")
    print(f"Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")