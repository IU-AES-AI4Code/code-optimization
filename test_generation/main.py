import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

from pydantic import BaseModel
from typing import List
from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer, AutoModelForCausalLM
import subprocess
from radon.metrics import mi_visit

# tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-7b-base-v1.5", trust_remote_code=True, device_map = "auto")
# model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-7b-base-v1.5", trust_remote_code=True, device_map = "auto")

app = FastAPI()

def form_prompt_for_method(method_code: str) -> str:
    prefix = "import pytest"
    test_comment = "# test for the method above\n# those tests cover each possible branch just once, no excessive repeats\n\n"
    prompt = "\n\n".join([prefix, method_code, test_comment])
    return prompt

def generate_code_tests(method_code: str, tokenizer: AutoTokenizer, model: AutoModelForCausalLM) -> str:
    prompt = form_prompt_for_method(method_code)
    tokenized_prompt = tokenizer(prompt, return_tensors="pt")
    tokenized_output = model.generate(**tokenized_prompt, max_new_tokens=200, do_sample=False, use_cache=True)
    whole_code = tokenizer.decode(tokenized_output[0], skip_special_tokens=True)

    prompt_length = len(tokenizer.tokenize(prompt))
    tests = tokenizer.decode(tokenized_output[0][prompt_length:], skip_special_tokens=True)    

    return whole_code, tests

def fix_interrupted_gen(whole_code: str) -> str:
    function_split = whole_code.split("def")
    function_split_count = len(function_split)
    # if there are more than two functions the last one could be interrupted in the middle
    # we want to get rid of such a function so that code is interpretable
    # if there are just two functions last line could be interrupted
    if function_split_count > 3:
        valid_parts = function_split[:-1]
        working_code = 'def'.join(valid_parts)
    else:
        line_split = whole_code.split("\n")
        valid_lines = line_split[:-1]
        working_code = '\n'.join(valid_lines)
    
    return working_code

def write_code_tests(code: str, path: str = "code_test.py") -> None:
    with open(path, 'w') as file:
        file.write(code)

def get_total_file_coverage(path: str = "code_test.py") -> int:
    test_command = f"pytest {path} --cov={path.split('.')[0]}"
    test_result = subprocess.run(test_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
    out = test_result.stdout

    try:
        # identify total result line
        linesplit = out.split('\n')
        for line in linesplit:
            if line[:5] == 'TOTAL':
                total_line = line
                break
        
        # retrieve just the coverage info
        coverage_string = total_line.split()[-1]
        coverage_percent = int(coverage_string[:-1])
    except:
        return 0
    return coverage_percent

def clean_tests(tests_code: str) -> str:
    # Split the tests code into lines
    lines = tests_code.split("\n")
    
    # Find the first line that starts with 'def'
    first_def_index = next((i for i, line in enumerate(lines) if line.strip().startswith("def")), None)
    
    # If there is no 'def' line, return the original tests_code
    if first_def_index is None:
        cleaned_tests_code = tests_code
    
    # Remove any lines before the first 'def' line that do not start with '#'
    cleaned_lines = [line for line in lines[first_def_index:] if line.strip().startswith("#") or line.strip().startswith("def")]
    
    # Join the cleaned lines back into a single string
    cleaned_tests_code = "\n".join(cleaned_lines)
    
    fixed_tests_code = fix_interrupted_gen(cleaned_tests_code)

    return fixed_tests_code

def calculate_maintainability_index(code: str) -> float:
    try:
        maintainability_index = mi_visit(code, multi=True)
        return maintainability_index
    except Exception as e:
        print(f"Error calculating maintainability index: {e}")
        return None

def run_tests(code_path: str = "code_test.py") -> bool:
    try:
        test_command = f"pytest {code_path}"
        test_result = subprocess.run(test_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
        return test_result.returncode == 0
    except Exception as e:
        print(f"Error running tests: {e}")
        return False

class CodeModel(BaseModel):
    code: List[str]

class CodeTestModel(BaseModel):
    code: List[str]
    tests: List[str]

@app.post("/gen_test")
async def generate_tests(data: CodeModel):
    ret_list = []
    for i in range(len(data.code)):
        try:
            whole_code, tests = generate_code_tests(data.code, tokenizer, model)
            # tests = clean_tests(tests)
            working_code = fix_interrupted_gen(whole_code)
            write_code_tests(working_code)
            test_coverage = get_total_file_coverage()
            ret_list.append({"tests": tests, "coverage": test_coverage})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        
    return ret_list

@app.post("/compute_coverage")
async def compute_coverage(data: CodeTestModel):
    ret_list = []
    for i in range(len(data.code)):
        try:
            with open("code_test.py", 'w') as file:
                file.write(data.code[i] + "\n" + data.tests[i])
            test_coverage = get_total_file_coverage()
            ret_list.append({"coverage": test_coverage})
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    return ret_list
    
@app.post("/score_code_test")
async def score_code_test(data: CodeTestModel):
    ret_list = []
    for i in range(len(data.code)):
        try:
            # Write the code and tests to a file
            with open("code_test.py", 'w') as file:
                file.write(data.code[i] + "\n" + data.tests[i])
            
            # Calculate maintainability index
            maintainability_index = calculate_maintainability_index(data.code[i])
            
            # Run the tests
            tests_passed = run_tests()
            ret_list.append({"maintainability_index": maintainability_index, "tests_passed": tests_passed})
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        
    # Return the scores
    return ret_list

@app.post("/compute_mi")
async def compute_mi(data: CodeModel):
    ret_list = []
    for i in range(len(data.code)):
        if data.code: # non-empty
            try:
                maintainability_index = calculate_maintainability_index(data.code[i])
                ret_list.append({"maintainability_index": maintainability_index})
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        else: # if empty
            ret_list.append({"maintainability_index": 0})
    return ret_list

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
