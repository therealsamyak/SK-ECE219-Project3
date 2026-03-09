---
title: FAQ
---

Q:
Hi Instructor,

For Question 7, I scaled the training data and evaluated on the same 100-question GSM8K subset. To better align the training data format with the model’s expected output structure, I first rewrote the GSM8K training answers using GPT-5-mini with a few-shot prompt to enforce a consistent:

```
Step 1: ...
Step 2: ...
...
Final Answer: <number>
```

format (removing raw calculation markers like <<...>> and ####, and standardizing intermediate arithmetic expressions).

Using this rewritten training dataset, I obtained the following accuracies:

Baseline: 42%
1,000 examples: 43%
3,000 examples: 47%
Full 7,473 examples: 55%
From these results, performance continues to improve quite noticeably as the dataset scales, especially from 3,000 to the full dataset. I therefore do not clearly observe diminishing returns yet within this range.

My question is:

Is it acceptable to conclude that diminishing returns are not yet visible at this data scale, and that we may still be in the “efficient scaling” regime?

A:
yes, please go ahead with your insights and if you can look to scale even more, find more data relevant to the case and see how far we can push this!!

---

Q:
Hi, Instructor.

For question 5, when I tried to train the LoRA SFT model,  I met with the out of memory error for GPU after few iteration. I already free the GPU by resetting conversation but this still happens. I am using Google colab Pro with T4 GPU runtime.

The complete error code is this: OutOfMemoryError: CUDA out of memory. Tried to allocate 3.02 GiB. GPU 0 has a total capacity of 14.56 GiB of which 2.60 GiB is free. Including non-PyTorch memory, this process has 11.96 GiB memory in use. Of the allocated memory 9.43 GiB is allocated by PyTorch, and 2.39 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_ALLOC_CONF=expandable_segments:True to avoid fragmentation. See documentation for Memory Management

A:

I had to change two parameters in the training arg for it to work for the 1k, the 2 parameters are: per_device_train_batch_size=4 and gradient_accumulation_steps=8. I asked the TA during office hour he said we are allowed to change the hyperparameter.

---

Q:
I'm having trouble with Question 1. The assignment says that we should expect about 35-40% accuracy. However, the best I have been able to get is 30%. I believe the issue is that I am not able to get Qwen to format the final answer as \boxed{<final_answer>} as reliably as I would like.  Does anyone have any tips for how to do better in Question 1?

Here is the system prompt that gave us the most success, in case that is helpful in suggesting fixes:

SYSTEM_PROMPT = (
"You need to solve math problems."
"Clearly present your solution in a step by step manner."
"Show intermediate calculations."
"The final answer should be an integer."
"Clearly label the final answer in the following format:"
"Final answer = \\boxed{<final_answer>}"
"Do not include anything other than the integer which is the final answer in your <final_answer>"
"If there are units involved in the final answer, do no include the units."
"Only include the integer portion of your answer."
"If the last portion of your response is not \'\\boxed{<final_answer>}\', then you failed."
)

A:
in the 100 answers, check how many errors are because of bad parsing. Because I tried with a similar looking prompt and i got 35.6%

Check temperature and inference code and if all that looks good, 30% should be acceptable too. ask AI to make a better prompt :)

---

Q:
Hi, Instructor

I have a clarification question regarding Project 3 – Question 13.

The write-up mentions that “we were able to obtain a model in the low-to-mid 70% accuracy range on this evaluation setup. This number serves as a practical reference point.” It also states that “if you surpass the staff reference number, you receive a bonus on top of the project score.”

I’m a bit confused about what specific performance threshold counts as surpassing the reference number. Since “low-to-mid 70%” is somewhat approximate, could you clarify what exact accuracy (e.g., 73%, 75%, etc.) would be considered above the staff reference?

In particular, achieving 75%+ pass@1 seems extremely challenging, so I want to make sure I understand the benchmark correctly.

Thank you very much for the clarification!

A:
it was 74% to be exact, but anything even touching 70% will be awarded full bonus.
you arent forced to just use pass@1 try different ways to beat that.

Having said that, 74% was in fact achieved in pass@1  but you have the full freedom to explore different sampling methods.

---

Q:

I get 68% accuracy with the base model if I comment out attn_implementation="eager", and 35% if I don't (the eager also seems to be a bit slower), why is that?

A:

The difference is mainly coming from how attention is implemented (obviously) and the numerical precision that gets carried forward. A major factor is batching under the eager implementation, which further reduces its performance. Since we don’t have a lot of compute, batching was recommended, but that seems to hurt eager more noticeably.

I can explain some performance improvement when not using eager, but this magnitude of jump is too large to feel fully comfortable with. I’m not entirely sure what to conclude from it yet.

To see sensible gains with the sft, i would say you stick with eager for now, but this becomes a good angle to explore in your bonus question and as a research topic too.

I will update you if i find some more convincing reasons.

---

Q:
I was wondering what behavior other groups have seen for task 3 of part a? Both the baseline model and SFT did a lot worse (8 - 10%) with 3-shot prompting (prompts generated by ChatGPT). I am wondering if this a similar experience for others, or if the way I am implementing k-shot prompting is incorrect and that's why test accuracy is bad.

A:
I did manage to get some improvements, but they weren't very significant jumps. Perhaps your formatting of the answers generated from the model isn't consistent across your approaches? I would double check and take a look at the SYSTEM_PROMPT (and how you are extracting the answer) for the model as well as the raw model response, comparing that to the raw model response with 3-shot prompting. Make sure the 3-shot examples you generate from ChatGPT is consistent with your SYSTEM_PROMPT and not confusing the model. (i.e. system prompts for a latex boxed answer vs the 3-shot examples uses "Answer: xxx").

P.S. I've fine-tuned and experimented with many different things towards the last part of Part A, and even then, my results seem very random and there were a lot of unexpected outcomes.

---

Q:
For the two questions, it was asked for both saving the csv files, and how do we need to report these two questions? Do we also need to attach the csv files when submitting the hw?

A:

no it is just for the agent pipeline to load csv, no need to submit

---

Q:
My baseline model got 43% with a good SYSTEM_PROMPT and an extraction method that grabs the last number if the model doesn't output the wanted format. After the SFT process, which I took from the helper code, the accuracy doesn't change at all. If I make the extraction only look at the proper output format and return "" for everyting else baseline model drops to 41% and SFT drops to 34%. Is there an expected accuracy improvement from the 1k example SFT?

EDIT: I re-ran the code again from a different IDE with all the same settings and got 44% with the SFT. I don't really know what causes this stochasticity

EDIT2: I loaded the same checkpoint and now the accuracy is 35%

Instructor Answer:
ok i would advise you to check the temperature and inference setting on the qwen model. but even the default of greedy or the 0.7 temp probably wont give this much of a difference. best way is to go through the q-a pairs and see what the errors are being encountered to best debug this. and ofcourse check the basic code logic also.

Student1 Answer:
I think it is because of the system prompt. The resulting SFT improvement heavily depends on it. If I change the prompt to make it more specific like saying write intermediate steps in bullet point form, for example, the baseline results will be better but the SFT improvement will be very little or cause harm even. I found that a simple system prompt works better than a complicated one to show the improvement in the fine tuning. Also make sure to load the data in the expected format to not accidentally train it on the prompt, the trl library explains the desired format for the train only on response to work because one format doesn't work. You can kind of tell by how high your initial loss is.

EDIT: I also removed the weird <<>> notation in the training data which hurt the training a lot

Student2 Answer:
Thank you! Student1 has a great suggestion, and it seems to work on our side! For others who struggle with this, you can check the prompt that the original prompt that Qwen developers used when they evaluate their model against gsm8k.

---

Q:
Dear TAs,

For project3 Q18, it says to "demonstrate (with 5 different prompts) ...". Can we select any prompts we want like the demo in the helper code (shown below)? We are not supposed to be playing with the dataset in q17 yet, correct?

```
# --- Demo: basic generation ---
resp = generate([{"role": "user", "content": "What is 2 + 2?"}])
print("Content:", resp.content)
```

A:

Yes any prompt is fine

---

Q: Can we use the Qwen model itself to assess the accuracy of the ReAct agent? I did not pull the keywords from the common_answers column. I instructed the planner to output the final answer in the form [[keyword, value], ...] where it should choose the keyword itself. Even in this very similar format, I'm not sure how to create a grading function that would score these (and others) as correct:

true: [['importance_score_std', '0.01'], ['importance_score_mean', '0.0']]

vs

generated: [['mean', 0.0026525198938991555], ['standard_deviation', 0.0063718028700038044]]

true: [['percentage_cases_min', '36.45'], ['percentage_deaths_max', '38.79']]

vs

generated: [['missing_values_no_of_cases_min', '36.45'], ['missing_values_no_of_deaths_max', '38.79'], ['comparison', 'no_of_deaths_max_has_higher_missing_values']]

 true:  [['normality_status', 'not normal']]

vs

generated: [['distribution', 'non-normal'], ['test_statistic', '0.9414'], ['p_value', '1.43e-33']]

A:
In this case yes! 
But in the report please clearly state the pipeline: what you do (why) and how is the LLM as judge is being used. 

Maybe use some simple test cases to make sure the LLM-judge is functioning as expected.

Again, THIS IS NOT MANDATORY IF YOU HAVE A BETTER APPROACH

---

Q:
For project3 Task 2.2 (Q5–7),

If we want to slightly revise the training hyperparameters (for example, increasing the number of epochs from 1 to 2), is it acceptable as long as we keep the hyperparameters consistent across Q5–Q7 and only vary the number of training examples?

Or should we strictly keep all hyperparameters exactly as specified in the original setup?

A:
yes, you are allowed to change whatever you want in the hypers. just state it clearly in your answers / report though

---

Q:

Is there a certain performance target we should be aiming for with our ReAct agents for part 2, or should we focus more on efficiency + error case analysis?

A:
no specific target, but in our experiments we reached 100%, just as a fyi

---

Q:
I tried batch_size = 16 and 32.
The accuracy of base model rise from 0.35 to 0.45. Is it normal?
Also, do you need to select the first 100 questions as evaluation set?

A:
no need to select the first 100 questions, but keep them fixed throughout the eval.

I think we can fix batch size of 16 here.

---

Q:
Is it expected to see (31%, 46%, 45%) for 0, 1k, 3k SFT respectively? I ran my script again, and instead got (31%, 35%, 45%).

A:
well to be honest, we can find justification for both the plots if they were given separately.
but you shouldnt be having different values for the same lora adapter unless you are doing random batching or have using a top_p sampling.

So, please check your implementation first, to keep all the things same during both evaluations and you have to get the same accuracies everytime (not that you are required to do this multiple times but since you arent getting the same accs means there is an element of randomization there, you should probably fix that.)

---
