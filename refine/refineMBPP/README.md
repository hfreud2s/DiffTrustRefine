# Clarifying Questions Pipeline & Data

This folder contains the code for running the complete clarifying questions experiment on the MBPP benchmark (currently only MBBP is supported). It also contains a dummy example (.MBPP-example-mini) as well as the results from ten runs on the complete MBPP-sanitized benchmark.
The experiment measures wether incoherence can detect ambiguous task descriptions and help to detect sensible clarifying questions to improve code correctness. 

## Structure

```
.
├── .MBPP-example-mini              # example that can be recreated following the instuctions below
│   ├── baseline                     
│   │   ├── candidates              # baseline candidates (sampled from the original task description)
│   │   │   ├── mbpp_106
│   │   │   ├── ...
│   │   │   └── mnpp_780
│   │   ├── baseline_stats.py
│   │   ├── batch_request.jsonl
│   │   └── batch_result.json
│   ├── refined
│   │   ├── batch_results           # contains all results from the various API requests
│   │   │   ├── batch_result_oracle  
│   │   │   ├── batch_result_refined_descriptions
│   │   │   ├── batch_result_questions    
│   │   │   └── refined_candidates
│   │   ├── candidates              # refined candidates (for each question, description and true answer)
│   │   │   ├── mbpp_106
│   │   │   │   ├── q1_desc1
│   │   │   │   ├── q1_desc2
│   │   │   │   ├── q1_oracle
│   │   │   │   ├── q2_desc1
│   │   │   │   ├── q2_desc2
│   │   │   │   ├── q2_oracle
│   │   │   │   ├── q3_desc1
│   │   │   │   ├── q3_desc2
│   │   │   │   └── q3_oracle
│   │   │   ├── ...
│   │   │   └── mnpp_780
│   │   ├── batch_request.jsonl
│   │   ├── questions_and_descriptions.json
│   │   └── refined_stats.json
├── .MBPP-experiment                    # experiment with the complete MBPP dataset (10 runs)
│   ├── baseline
│   └── refined
├── batch_processing.py                 # This file contains all relevant functions to work with large batch requests and results
├── build_batch.py                      # This file contains the methods to build batch requests for efficient API calls
├── compute_stats.py                    # Functions to compute incoherence and error for the baseline and the refined candidates
├── refine_descriptions.py              # All necessary prompts to generate questions and refined descriptions
└── README.md
```


> [!NOTE]
> Since this experiment requires a lot of API calls the most efficient way to handle this is via batch requests. It is possible to generate all program candidates via single API calls using the original difftrust code (for example when models are used that don't provide the option to process batches). If batch processing shall be used, just follow the guide below. We use OpenAI to generate the program candidates and Anthropic to generate the questions and refinements so API keys for both are required. If other APIs are used it might be necessary to adjust the format of the request messages.

## Step by Step Guide (using batch requests for efficiency)

Just un-comment the relevant code snippets in the main method and provide the required paths.

1. In `build_batch.py` call `build_baseline_batch()` to generate a jsonl-file with batch requests to the OpenAI API. This will generate a file with individual requests for each task x the number of program candidates. For smaller experiments provide an additional list of MBPP task ids that should be evaluated.

2. In `batch_processing.py` call `create_batch()` with the path to the just created jsonl-file. This will send the just generated batch to Open AI.

3. Wait until the batch is completed and download it from Open AI.

4. If not all requests could be completed (e.g. due to server issues), re-send the failed requests and merge the result with the results from the original requests. To do so:
    - Save the file with the completed requests and the file with the failed requests
    - Call `recreate_batch()` with the path of the original batch file and the path of the failed batches -> creates a new file containing only the requests that previously failed
    - Re-send this batch
    - Download and merge with the already completed requests using `merge_batches()`

5. In `batch_processing.py` call `postprocess_baseline_candidates()`. This converts the baseline batch results into per-task cloudpickle files (just like in the original difftrust code). 

6. In `compute_stats.py` run `compute_baseline_stats()` to compute incoherence and error of all tasks.

7. In `refine_descriptions.py` run `generate_questions()`. This will check, which tasks have a non-zero incoherence. For those tasks, a batch request for num_questions binary questions will be send to Anthropic. Currently only binary questions are supported to reduce the possible answer space. Again, wait for the results, download them, and save them somewhere in the project.

8. In `batch_processing.py` run `postprocess_questions()`. This will store all questions for all tasks in one json-file. This file will be expanded in the next steps to generate one file containing all questions and descriptions needed for the final evaluation.

9. To generate the refined task descriptions for the yes and no answer for all questions, first run `generate_refined_descriptions()` in `refine_descriptions.py` then run `postprocess_refined_descriptions()` in `batch_processing.py`.

10. To generate the true task descriptions (answered by the oracle) for all questions, first run `generate_oracle_description()` in `refine_descriptions.py` then run `postprocess_oracle_descriptions()` in `batch_processing.py`.

11. The final step is to build and send the batch requests for all program candidates. To build the request, run `build_refinement_batch()` in `build_batch.py`. Send the request with `create_batch()` in `batch_processing.py` and wait for the results. Save them and post-process them with `postprocess_refined_candidates()` in `batch_processing.py`.

12. Now compute the final results using the code from `compute_stats.py`. 
