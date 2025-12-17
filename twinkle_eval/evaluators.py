import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from .dataset import Dataset
from .evaluation_strategies import EvaluationStrategy
from .logger import log_error
from .models import LLM


class RateLimiter:
    def __init__(self, calls_per_second):
        self.no_limit = calls_per_second == -1
        self.interval = 1.0 / calls_per_second if not self.no_limit else 0
        self.last_call_time = 0

    def wait(self):
        if self.no_limit:
            return
        current_time = time.time()
        time_to_wait = self.interval - (current_time - self.last_call_time)
        if time_to_wait > 0:
            time.sleep(time_to_wait)
        self.last_call_time = time.time()


class Evaluator:
    def __init__(
        self,
        llm: LLM,
        evaluation_strategy: EvaluationStrategy,
        config: dict,
        eval_method: str,
        system_prompt_enabled: bool = True,
        samples_per_question: int = 1,
        pass_k: int = 1,
        shuffle_options: bool = False,
        model_overrides: dict | None = None,
    ):
        self.llm = llm
        self.evaluation_strategy = evaluation_strategy
        self.eval_method = eval_method
        self.system_prompt_enabled = system_prompt_enabled
        self.config = config
        self.rate_limiter = RateLimiter(calls_per_second=self.config["llm_api"]["api_rate_limit"])
        self.samples_per_question = int(samples_per_question)
        self.pass_k = int(pass_k)
        self.shuffle_options = bool(shuffle_options)
        self.model_overrides = model_overrides or {}

    def shuffle_question_options(self, question_data):
        options = []
        for key in ["A", "B", "C", "D"]:
            if key in question_data:
                options.append((key, question_data[key]))

        if not options:
            return question_data

        correct_ans = question_data["answer"]
        correct_option_text = question_data.get(correct_ans)

        random.shuffle(options)

        new_data = {"question": question_data["question"]}

        for (old_key, text), (new_key, _) in zip(
            options, [("A", ""), ("B", ""), ("C", ""), ("D", "")]
        ):
            new_data[new_key] = text
            if text == correct_option_text:
                new_data["answer"] = new_key

        return new_data

    def evaluate_file(
        self, file_path: str, timestamp: str, prompt_lang: str = "zh", dataset_label: str = ""
    ):
        dataset = Dataset(file_path)
        shuffle_enabled = self.shuffle_options

        total_correct_samples = 0
        total_samples = 0
        detailed_results = []
        question_stats = {}  # question_id -> {"correct": int, "total": int}

        with ThreadPoolExecutor() as executor:
            future_tasks = []
            future_to_data = {}

            for idx, q in enumerate(tqdm(dataset, desc="處理題庫中")):
                if shuffle_enabled:
                    q = self.shuffle_question_options(q)

                option_lines = [f"{k}: {q[k]}" for k in ["A", "B", "C", "D"] if k in q]
                question_text = q["question"] if not option_lines else q["question"] + "\n" + "\n".join(option_lines)

                try:
                    correct_answer = self.evaluation_strategy.normalize_answer(q["answer"])
                except (KeyError, AttributeError) as e:
                    log_error(f"\n Error processing question {idx + 1}: {str(e)}")
                    continue

                self.rate_limiter.wait()
                future = executor.submit(
                    self.llm.call,
                    question_text,
                    prompt_lang,
                    self.eval_method,
                    self.system_prompt_enabled,
                    self.samples_per_question,
                    self.model_overrides,
                )
                future_tasks.append(future)
                future_to_data[future] = (question_text, correct_answer, idx, self.samples_per_question)

            for future in tqdm(
                as_completed(future_tasks), total=len(future_tasks), desc="處理回應中"
            ):
                llm_chat_completion = future.result()

                usage = llm_chat_completion.usage
                question_text, correct_answer, question_id, expected_samples = future_to_data[future]

                for sample_id, choice in enumerate(llm_chat_completion.choices[:expected_samples]):
                    message = choice.message
                    content = message.content
                    reasoning_content = getattr(message, "reasoning_content", None)

                    predicted_answer_raw = self.evaluation_strategy.extract_answer(content)
                    predicted_answer = (
                        None
                        if predicted_answer_raw is None
                        else self.evaluation_strategy.normalize_answer(predicted_answer_raw)
                    )

                    is_correct = (
                        False
                        if predicted_answer is None
                        else self.evaluation_strategy.is_correct(predicted_answer, correct_answer)
                    )
                    question_stats.setdefault(question_id, {"correct": 0, "total": 0})
                    if is_correct:
                        question_stats[question_id]["correct"] += 1
                    question_stats[question_id]["total"] += 1
                    if is_correct:
                        total_correct_samples += 1
                    total_samples += 1

                    detailed_results.append(
                        {
                            "question_id": question_id,
                            "sample_id": sample_id,
                            "question": question_text,
                            "correct_answer": correct_answer,
                            "predicted_answer": predicted_answer,
                            "is_correct": is_correct,
                            "llm_output": content,
                            "llm_reasoning_output": reasoning_content,
                            "usage_completion_tokens": usage.completion_tokens,
                            "usage_prompt_tokens": usage.prompt_tokens,
                            "usage_total_tokens": usage.total_tokens,
                        }
                    )

            accuracy = total_correct_samples / total_samples if total_samples else 0
            pass_at_k_values = []
            for stats in question_stats.values():
                c = stats["correct"]
                n = stats["total"]
                k = self.pass_k
                if n == 0 or k > n:
                    pass_at_k_values.append(0)
                    continue
                # pass@k = 1 - comb(n-c, k) / comb(n, k)
                from math import comb

                if c == 0:
                    pass_at_k_values.append(0)
                else:
                    pass_at_k_values.append(1 - comb(n - c, k) / comb(n, k))

            pass_at_k = sum(pass_at_k_values) / len(pass_at_k_values) if pass_at_k_values else 0

        base_results_dir = os.path.join("results", "details")
        model_name = self.config.get("model", {}).get("name", "model")
        safe_model = re.sub(r"[^\w.-]+", "_", str(model_name))
        results_dir = os.path.join(base_results_dir, f"{safe_model}_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)

        safe_dataset = dataset_label or os.path.basename(os.path.dirname(file_path))
        results_path = os.path.join(results_dir, f"eval_results_{safe_dataset}_{timestamp}.jsonl")

        record = {
            "timestamp": timestamp,
            "dataset_label": safe_dataset,
            "file": os.path.relpath(file_path),
            "accuracy": accuracy,
            "pass_at_k": pass_at_k,
            "details": detailed_results,
        }

        # 追加到同一檔案（同一 run 的多個檔案會寫在同一 jsonl）
        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"✅ 評測完成，結果已追加至 {results_path}")
        return file_path, {"accuracy": accuracy, "pass_at_k": pass_at_k}, results_path
