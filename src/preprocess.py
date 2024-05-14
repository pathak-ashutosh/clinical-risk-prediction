def get_disease_name(code):
    if code=="oud":
        code_text = "Opioid Use Disorder"
    elif code == "sud":
        code_text = "Substance Use Disorder"
    elif code == "diabetes":
        code_text = "Diabetes"
    else:
        print("Error in the code")
        code_text = Null
        
    return code_text
    
def create_test_prompt(examples, code_text, text_column, label_column):
    # Initialize static strings for the prompt template
    INTRO_BLURB = "Given a patient's past medical history, predict whether the patient will have a future diagnosis of " + code_text + ". Return 'Yes' or 'No' after the XML tag <Diagnosis>."
    INSTRUCTION_KEY = "### Instruction:"
    INPUT_KEY = "### Input:"
    RESPONSE_KEY = "### Response:"
    END_KEY = "### End"
    
    instruction = f"{INSTRUCTION_KEY}\n{INTRO_BLURB}"
    input_context = f"{INPUT_KEY}\n{examples[text_column]}" if examples[text_column] else None

    high_low_label = examples[label_column]
    if high_low_label == "High":
        t_label = "Yes"
    elif high_low_label == "Low":
        t_label = "No"
    else:
        print("There is some error with the label")
        
    response_ground_truth = f"{RESPONSE_KEY}\n<Diagnosis>"
    
    end = f"{END_KEY}"

    # Create a list of prompt template elements
    parts_ground_truth = [part for part in [instruction, input_context, response_ground_truth] if part]
    # Join prompt template elements into a single string to create the prompt template
    formatted_prompt_ground_truth = "\n\n".join(parts_ground_truth)

    # # Store the formatted prompt template in a new key "text"
    # examples["prompt"] = formatted_prompt

    return formatted_prompt_ground_truth