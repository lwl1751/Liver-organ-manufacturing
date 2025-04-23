schema = {
    "Seeded cell density": {
      "description": "(list of string) Density of cells seeded onto the scaffold, to form spheroids, or on the chip at the start of the experiment. The value should include a numerical value with a unit. Acceptable units include [cells per well, cells/mL, cells/cm³, cells/cm², cells per scaffold], or a description like '4 × 10^5 cells in 40 μL in a 96-well plate'. This should not be confused with the coefficient of variation of cell density.",
      "example": "1.2 × 10^5 cells/cm2, 5 × 106 cells per 25 cm2 "
    },
    "Material in cell culture": {
      "description": "(list of string) Denotes materials used as scaffolds or extracellular matrix (ECM) substitutes to support cell growth and development in vitro. These bioactive substances, typically organic polymers, promote cell adhesion, proliferation, and differentiation by mimicking native tissue environments. They are often biodegradable and selected to provide structural support that resembles the native cellular matrix.",
      "example": "PLGA, PLLA, silk fibroin, chitosan"
    },
    "Concentration of material": {
      "description": "(list of string) Specifies the concentration or dilution of the 'Material in cell culture' used to prepare the cell culture environment. This concentration impacts the bioactivity and cell compatibility of the material. If not explicitly stated, it may be indicated through terms like 'dissolved' or 'dispersed' in specific media.",
      "example": "0.5% (w/v), aqueous solution of 10% gelatin, 10% GAA solution in 0.1 M borax and 10% gelatin in distilled water"
    },
    "Chip material": {
      "description": "(list of string) Refers to the materials used in constructing organ-on-chip or biochip devices. These materials must be biocompatible, chemically inert, and possess appropriate mechanical properties to create a stable microenvironment for cell culture. Optical transparency is often a key requirement for imaging and observing cell behavior in microfluidic applications.",
      "example": "polydimethylsiloxane (PDMS), glass, polycarbonate"
    },
    "Cross-linking agent": {
      "description": "(list of string) The type of cross-linking agent used to stabilize the scaffold.",
      "example": "CaCl2, glutaraldehyde, EDC/NHS"
    },
    "Pore size": {
      "description": "(list of string) The size of the pores in the scaffold. Multiple values may be mentioned in the text; however, only the most pertinent value should be extracted. Prioritize extraction by the following order: mean > maximum > range. If a mean value is not explicitly stated, refrain from calculating or inferring it.  A numerical value and unit is expected.",
      "example": "50 µm"
    },
    "Diameter": {
      "description": "(list of string) The diameter of the scaffold. Multiple values may be mentioned in the text; however, only the most pertinent value should be extracted. Prioritize extraction by the following order: mean > maximum > range. If a mean value is not explicitly stated, refrain from calculating or inferring it.  A numerical value and unit is expected.",
      "example": "200 µm"
    },
    "Manufacturing method": {
      "description": "(list of string) The approach used to design or fabricate the scaffold.",
      "example": "lyophilization, freeze-drying"
    },
    "Perfusion rate": {
      "description": "(list of string) The rate at which the medium flows through the chip. Multiple values may be mentioned in the text; however, only the most pertinent value should be extracted. Prioritize extraction by the following order: mean > maximum > range. If a mean value is not explicitly stated, refrain from calculating or inferring it.  A numerical value and appropriate unit is expected. Possible units include [µL/min, μL/hr, mL/min, mL/h].",
      "example": "200 µL/min"
    },
    "Channel width": {
      "description": "(list of string) The width of the channels in the microfluidic chip. This should be given as a numerical value with units. Do not confuse with channel height.",
      "example": "500 µm"
    },
    "Channel height": {
      "description": "(list of string) In microfluidic chips, channel height refers to the height or depth of a microchannel, which is the vertical dimension along which fluid flows. This should be given as a numerical value with units. Do not confuse with channel width.",
      "example": "100 µm"
    }
}

system_prompt = '''You are an expert assistant in liver tissue engineering.'''

## 实体逻辑关系处理
scaffold_needed_para_list = ['Seeded cell density', 'Material in cell culture', 'Concentration of material', 'Cross-linking agent', 'Pore size', 'Diameter', 'Manufacturing method']
chip_needed_para_list = ['Seeded cell density', 'Material in cell culture', 'Concentration of material', 'Chip material', 'Perfusion rate', 'Channel width', 'Channel height']

scaffold_needed_para = {scaffold_needed_para_list[k]:schema[scaffold_needed_para_list[k]] for k in range(len(scaffold_needed_para_list))}
chip_needed_para = {chip_needed_para_list[k]:schema[chip_needed_para_list[k]] for k in range(len(chip_needed_para_list))}

schema_para = {
    "Cell Culture Conditions": {
        "Cell type": [],
        "Seeded cell density": []
      },
    "Scaffold and Chip Materials": {
        "Material in cell culture": [],
        "Concentration of material": [],
        "Chip material": []
      },
    "Culture Methods": {
        "3D scaffold": {
            "Cross-linking agent": [],
            "Pore size": [],
            "Diameter": [],
            "Manufacturing method": []
          },
        "liver-on-a-chip": {
            "Perfusion rate": [],
            "Channel width": [],
            "Channel height": []
          }
      }
}

method_2_schema = {
    "3D scaffold": {
        "Description": "Liver 3D scaffold culture is a technique that utilizes three-dimensional scaffold materials, combined with hepatocytes and other cell types, to mimic the liver microenvironment in vitro and construct liver tissues or organoids. By providing a microenvironment for cell attachment, growth, and functional expression, 3D scaffold culture facilitates the study of liver development, disease modeling, and drug screening.",
        "Cell Culture Conditions": schema_para["Cell Culture Conditions"],
        "Scaffold and Chip Materials": schema_para["Scaffold and Chip Materials"],
        "Culture Methods": {"3D scaffold": schema_para["Culture Methods"]["3D scaffold"]},
    },
    "liver-on-a-chip": {
        "Description": "A liver-on-a-chip is a specific type of liver microfluidic device that reconstructs a simplified liver model on a chip by mimicking the microstructures and physiological functions of the liver.",
        "Cell Culture Conditions": schema_para["Cell Culture Conditions"],
        "Scaffold and Chip Materials": schema_para["Scaffold and Chip Materials"],
        "Culture Methods": {"liver-on-a-chip": schema_para["Culture Methods"]["liver-on-a-chip"]},
    }
}

method_2_needed_para = {
    "3D scaffold": scaffold_needed_para,
    "liver-on-a-chip": chip_needed_para
}

summary_1 = '''Your task is to determine whether the following text is related to liver tissue engineering.
- If the text is relevant, summarize the methods and cell types used in liver tissue engineering mentioned in the text.
    * If no cell types are explicitly mentioned, omit them from your summary.
    * Provide a detailed explanation of how you identified the methods and cell types, clearly stating the textual evidence.
- If the text is unrelated to liver tissue engineering, briefly explain the reasoning behind your conclusion.

Requirements:
1. Rely solely on the information provided in the text. Avoid making assumptions or adding information not explicitly stated.
2. Use specific and detailed terms for methods and cell types whenever possible. Avoid vague descriptions unless the text lacks specificity.
3. Return the result in the following format:
{
    "is_review_article": true/false, // If the text is a review article, set this to true.
    "relevant": true/false,  // Determine if the text is related to liver tissue engineering.
    "methods": ["<list of methods>"], // If no methods are mentioned, return an empty list []
    "cell types": ["<list of cell types>"], // If no cell types are mentioned, return an empty list []
    "explanation": "<detailed explanation>"
}

The text is as follows:
<text>
'''

summary_2 = '''Based on your previous analysis, extract the primary cell culture method and cell types used in the text. 
Then, classify the culture method into one of the following categories:
    - liver-on-a-chip: A liver-on-a-chip is a specific type of liver microfluidic device that reconstructs a simplified liver model on a chip by mimicking the microstructures and physiological functions of the liver. Liver-on-a-chip typically contains multiple cell types, such as hepatocytes, endothelial cells, and Kupffer cells, and simulates liver sinusoids and central veins through microchannels to study liver diseases, drug toxicity, liver regeneration, and more. Requirements: it must be clear in the txt file, such as microfluidics, chip, organ chip and other related words appear, in order to judge belong to liver-on-a-chip.
    - 3D scaffold: Liver 3D scaffold culture is a technique that utilizes three-dimensional scaffold materials, combined with hepatocytes and other cell types, to mimic the liver microenvironment in vitro and construct liver tissues or organoids. By providing a microenvironment for cell attachment, growth, and functional expression, 3D scaffold culture facilitates the study of liver development, disease modeling, and drug screening.
    - 2D sandwich culture: Utilizes a double-layered structure where hepatocytes or other cell types are cultured between two layers of extracellular matrix or gel.
    - others: If the method does not fit any of the above categories or the text is a review article, classify it as "others." Note: Review articles, or texts that describe general principles or surveys of methods without focusing on specific experimental details, should always be classified as "others."

Requirements:
1. Primary Method Selection:
    - Classify the method into one of the categories, following these priority rules:
        * liver-on-a-chip > 3D scaffold > 2D sandwich culture > others.
        * If a scaffold is used within a chip, classify the method as liver-on-a-chip.
    - If the text is a review article, always select "others" regardless of the methods mentioned.
    - If the text is unrelated to liver tissue engineering,, always select "others" regardless of the methods mentioned.
2. Cell Types:
    - Extract the most relevant cell types cultured using the selected method:
        * Single-cell cultures: Use specific names (e.g., "primary human hepatocytes").
        * Co-cultures: List all explicitly mentioned cell types (e.g., "primary human hepatocytes, Kupffer cells").
        * If no cell types are mentioned for the method, indicate this explicitly (acceptable outcome).
    - Criteria for determining significance:
        * Contribution to the study’s findings.
        * Alignment with the article's main focus.
        * Most detailed description in the text.
    - Use detailed and specific terms; avoid vague descriptors like "hepatocyte-like cells" unless no further detail is provided.
    - Ensure the extracted cell types are explicitly supported by the text.
3. Source Explanation:
    - Provide a succinct explanation of how the selected method and cell type(s) were identified from the article text.
4. Reasoning Steps:
    - Provide a step-by-step explanation of how the method and cell type(s) were identified and selected.
5. Strict Text Reliance:
    - Base all extractions solely on the provided article text, avoiding assumptions or inferences not explicitly supported by the text.
6. Output Format:
    - Explanation: provide a succinct explanation of how the selected method and cell type(s) were identified from the article text.
    - Reasoning： Provide a step-by-step explanation of how the method and cell type(s) were identified and selected:
        * Step 1: Identification of all methods and cell types.
        * Step 2: Application of the priority order.
        * Step 3: Verification of alignment with the article’s focus.
        * Step 4: Validation of textual support for extracted entities.
        * Step 5: Final selection rationale.
    - Summariz: Return the result as a Python JSON object:
    ```json
    {
        "method": "<method>",
        "cell type": "<list of cell types>", // If no cells are cultured, return an empty list []
    }
    ```
'''

paragraph_check_prompt = '''Your task is to identify, verify, and evaluate the entities and their values extracted from a given text. 
Focus on determining their explicit mention, contextual relevance, and formatting accuracy, particularly within the domain of organ manufacturing.
Follow the steps below to ensure precision and consistency.

Special Note:
The provided Extracted Sentence-Level Entity Information is based on multiple paragraphs. However, your task is limited to confirming whether these entities are explicitly mentioned within the specific paragraph provided and whether their values are reasonable.

Instructions:
1. Focus on Mention: 
    - Verify Explicit Mentions: For each extracted entity, confirm its explicit mention in the text.
        * Ensure numerical values are formatted correctly (e.g., reformat "5 106 cells/mL" to "5 x 10^6 cells/mL").
        * Do not add new values; your task is to evaluate the provided values only.
        * Base your verification solely on the provided text—avoid any assumptions or inferences.
    - If an entity or its value is not mentioned, exclude it from the output.
    - Prioritize Detail: Ensure that when two parameters represent the same concept, the most detailed and precise value is retained. For example:
        * Use "poly(lactic-co-glycolic acid)" instead of "PLGA."
        * Prefer "100 μg/mL collagen" over "collagen."
        * Choose "wet electrospinning" over "wet electrospun" and "0.5 mL/h" over "0.5 ml/h."
2. Contextual Evaluation:
   - For explicitly mentioned entities:
        * Assess whether the value is accurate, reasonable, and contextually appropriate based on the text and the given requirements.
        * Confirm the value aligns with the entity type and its description.
3. Handling Absences:
    - If no entities or values are mentioned in the text, return an empty result.

Output Requirements:
1. Mentioned Entities
    - List all entities and values that are explicitly mentioned in the text:
    ###
    entity_type: <entity type>
    value: <value mentioned in the text>
    mention: <return specific sentence in the text mentioning this entity>
    ###
2. Reasonableness Assessment
    - Independently evaluate whether each mentioned value is reasonable:
    ###
    entity_type: <entity type>
    value: <value mentioned in the text>  
    reasonable: <true or false, concise explanation of why the entity is valid based on the provided text>
    ###
3. Final Verified Output
    - Provide a consolidated JSON array of entities that are both mentioned and reasonable:
    ```json
    [
        {
        "entity_type": entity type,
        "value": entity value,
        "reason": <providing specific sentence in the text mentioning this entity, concise explanation of why the entity is valid>
        },
    ...
    ]
    ```

Inputs:
1. Text Content:
<text>

2. Entity Requirements:
<entity requirements>

3. Extracted Sentence-Level Entity Information:
<extracted sentence-level entity information>
'''

paragraph_fix_prompt = '''Your task is to check whether your previous output is correct and return the final output.
Instructions:
1. Focus on Mention: 
    - Verify Explicit Mentions: For each extracted entity, confirm its explicit mention in the text.
        * Base your verification solely on the provided text—avoid any assumptions or inferences.
    - If an entity or its value is not mentioned, exclude it from the output.
2. Contextual Evaluation:
   - For explicitly mentioned entities:
        * Assess whether the value is accurate, reasonable, and contextually appropriate based on the text and the given requirements.
        * Confirm the value aligns with the entity type and its description.
3. Handling Absences:
    - If no entities or values are mentioned in the text, return an empty result.

Output Requirements: Provide a consolidated JSON array of entities that are both mentioned and reasonable:
```json
[
    {
    "entity_type": entity type,
    "value": entity value,
    "reason": <providing specific sentence in the text mentioning this entity, concise explanation of why the entity is valid>
    },
...
]
```
'''

match_prompt = '''You are tasked with completing a fill-in-the-blank task by identifying and selecting the correct values of the specified target entities from a provided text. 
The task focuses exclusively on one specified method and its associated cell type. 
Each entity has multiple potential values along with justifications, which needs to be carefully analyzed to select the most appropriate value. 
Ensure that all selected entity values strictly adhere to the Entity Requirements and belong to the same method and cell type as described in the text.

Detailed Task Instructions:
1. Understand the Focus:
    - You are working with a single specified method and its corresponding cell type.
    - All selected values must align exclusively with the method and cell type described.
    - If the text describes multiple methods or cell types, filter out any information unrelated to the specified method and cell type.
2. Entity Value Selection:
    - Each entity may have multiple potential values with justifications.  
      - Carefully evaluate each justification and its alignment with the criteria.  
      - Identify any errors in the justifications, and provide reasoning for rejecting incorrect values.
    - Use the Entity Requirements and the available options as guides to make your selection.
3. When to Leave Values Blank:
    - If an entity's value cannot be determined with absolute certainty, leave the field blank ([]).
    - If no matching value exists in the text or options, leave the field blank ([]).
4. Available Options Only:
    - Select values only from the available options for each entity.
    - Avoid making assumptions or selecting values not explicitly provided.
5. Entity-Specific Rules:
    - Numeric Entities (e.g., Perfusion Rate, Pore Size, Porosity): 
        - Select the value based on this priority:
            * Mean value (if explicitly provided).
            * Maximum value.
            * Range value (if explicitly mentioned).
        - Avoid calculations or assumptions; only include values directly provided in the text.
    - Material in Cell Culture and Concentration:
        * If a concentration is specified, ensure the corresponding material is also included.
        * Material in cell culture can exist without concentration of material.
    - Material in cell culture and Chip material:
        * Materials in "Material in Cell Culture" must differ from "Chip Material". For example, scaffold materials like "PLGA" are distinct from chip materials like "PDMS."
6. Detail-Oriented Selection:
    - If two parameters represent the same concept, please select and retain the more detailed one. For example:
        * Use "poly(lactic-co-glycolic acid)" instead of "PLGA."
        * Prefer "100 μg/mL collagen" over "collagen."
        * Choose "wet electrospinning" over "wet electrospun" and "0.5 mL/h" over "0.5 ml/h."
    - Justify why one value is selected over the other, focusing on specificity and alignment with the Entity Requirements.
7. Strict Text Alignment:
    - Extract information strictly from the provided text.
    - Avoid assumptions or extrapolations; rely solely on explicit mentions.
8. Output Format:
    - Analysis Process: Provide a step-by-step analysis for each entity, detailing:
        * The potential values identified from the text.
        * The reasoning for selecting or rejecting each value, including addressing any errors in the provided justifications.
        * Why the selected value aligns best with the Entity Requirements and the specified method and cell type.
        * If the value is left blank, explain why.
    - After completing the analysis, provide a summary JSON schema capturing the selected values in the following format:
        ```json
            <JSON schema>
        ```
9. Output Requirements:  
    - Provide both the analysis process and the JSON schema. Ensure clarity and completeness for each entity.

Supporting Information:
1. Entity Requirements:
<Entity requirements>

2. Provided Text:
<text>

3. Available Options for Each Entity:
<Options for each entity>
'''

reflect_prompt = '''Your task is to evaluate and refine the results of a fill-in-the-blank task. 
The evaluation focuses on verifying whether the selected values in the provided JSON schema adhere strictly to the Entity Requirements and accurately reflect the information in the provided text.
If errors, inconsistencies, or missing values are identified, provide corrections with clear justifications based solely on the source text and entity requirements.

Detailed Evaluation Instructions:
1. Understand the Scope:
    - Ensure all selected values align exclusively with the specified method and associated cell type.
    - Exclude values that are inconsistent with the text or unrelated to the specified method and cell type.
2. Entity Value Evaluation:
    - For each entity, evaluate the selected value(s) for:
        * Alignment with the Entity Requirements.
        * Explicit mention or clear derivation from the provided text.
        * The selected value must align exclusively with the method and cell type described.
    - Assess the reasoning behind each value (if provided) for accuracy and sufficiency.
3. When to Make Corrections:
    - Make corrections if:
        * The selected value is unsupported by the text.
        * The justification is incorrect or incomplete.
        * The field contains an incorrect or less precise value.
        * The value is missing but can be derived with certainty from the text.
        * The value is align exclusively with the method and cell type described.
    - If a value cannot be determined with certainty, leave the field blank ([]).
4. Corrections with Justifications:
    - For each correction, provide a detailed explanation, addressing:
        * Why the original selection was incorrect or incomplete.
        * Why the new value is more appropriate, referencing specific text excerpts.
5. Strict Text Alignment:
    - Extract information strictly from the provided text.
    - Avoid assumptions or extrapolations; rely solely on explicit mentions.
6. Entity-Specific Rules:
    - Numeric Entities (e.g., Perfusion Rate, Pore Size, Porosity): 
        - Select the value based on this priority:
            * Mean value (if explicitly provided).
            * Maximum value.
            * Range value (if explicitly mentioned).
        - Avoid calculations or assumptions; only include values directly provided in the text.
    - Material in Cell Culture and Concentration:
        * If a concentration is specified, ensure the corresponding material is also included.
        * Material in cell culture can exist without concentration of material.
    - Material in cell culture and Chip material:
        * Materials in "Material in Cell Culture" must differ from "Chip Material" . For example, scaffold materials like "PLGA" are distinct from chip materials like "PDMS."
7. Output Format: 
    - For each correction or addition, provide a clear explanation in the following format:
        ###
        Field Name: <entity>
        Original Value: <value>
        Revised Value: <value>
        Justification: <reason for correction or addition based on text or lack of correct options>

Supporting Information:
1. Entity Requirements:
<Entity requirements>

2. Provided Text:
<text>

3. Original JSON Output:
<JSON schema>
'''

reflect_check_prompt = '''Your task is to check whether your previous output is correct and return the final output.
Pay special attention to whether the values for each entity adhere strictly to the Entity Requirements provided below:  
<Entity requirements>  

Entity-Specific Rules:
- Numeric Entities (e.g., Perfusion Rate, Pore Size, Porosity): 
    - Select the value based on this priority:
        * Mean value (if explicitly provided).
        * Maximum value.
        * Range value (if explicitly mentioned).
    - Avoid calculations or assumptions; only include values directly provided in the text.
- Material in Cell Culture and Concentration:
    * If a concentration is specified, ensure the corresponding material is also included.
    * Material in cell culture can exist without concentration of material.
- Material in cell culture and Chip material:
    * Materials in "Material in Cell Culture" must differ from "Chip Material" . For example, scaffold materials like "PLGA" are distinct from chip materials like "PDMS."

Your output must be in the following Python JSON format, with no additional explanations:
```json
    <method>
```
'''


extract_prompt = '''
Your task is to extract specific parameters related to a cell culture method from the given article text. 
The extraction process should strictly follow the entity requirements and include only values explicitly stated in the article. 
Do not infer any values or make assumptions. 
Each parameter should be mapped correctly, and the values should match the precise phrasing used in the article.

Extraction Instructions:
1. Text-Based Reliance:
    - Only include information explicitly mentioned in the article.
    - Do not infer or assume any values. All extracted entities must have a direct source in the text.
2. Logical Grouping:
    - Ensure that all extracted parameters belong to the same culture method. Do not mix parameters from unrelated methods.
3. Entity Dependence:
    - The Concentration of material must correspond to the Material in cell culture listed. If one is provided, the other must also be present.
4. Numeric Entity Handling:
    - For numeric entities (e.g., Perfusion rate, Pore size, Porosity), extract one value based on the following priority:
        * Mean value (if explicitly provided).
        * Maximum value.
        * Range value (if explicitly mentioned).
    - Avoid calculations or assumptions; only include values directly provided in the text.
5. Detail Selection:
    - Choose the most detailed value for each entity. For example:
        * "poly(lactic-co-glycolic acid)" instead of "PLGA".
        * "100 μg/mL collagen" instead of just "collagen".
        * "wet electrospinning" instead of "wet electrospun".
6. Output Format:
    - Your output must be in the following Python JSON format, with no additional explanations:
        ```json
        <method>
        ```

The article is as follows:
<text>

The entity requirements are as follows:
<entity requirements>
'''

extract_fix_prompt = '''
Your task is to evaluate the reasonableness and logical consistency of previously extracted JSON Schema for a cell culture method. 
This evaluation focuses on verifying whether each entity value is appropriate, accurate, and logically aligned with the relationships between entities as described in the text.

Here is how you should approach the evaluation, providing a detailed analysis for each step:
1. Verification: 
    - Using the specified Cell type as the basis, verify that each entity is correctly identified and matches the exact phrasing and detail from the article. 
    - Ensure that no external assumptions or guesses have been introduced.
2. Entity Dependence: 
    - 'Concentration of material' specifies the concentration or dilution of the 'Material in cell culture'. If 'Concentration of material' has a value, then 'Material in cell culture' must also have a value. However, the reverse is not necessarily true. Additionally, 'Concentration of material' must correspond to the specific 'Material in cell culture' listed.
3. Numeric Entity Handling:
    - For numeric entities (e.g., Perfusion rate, Pore size, Porosity), extract one value based on the following priority:
        * Mean value (if explicitly provided).
        * Maximum value.
        * Range value (if explicitly mentioned).
    - Avoid calculations or assumptions; only include values directly provided in the text.
4. Detail Selection:
    - Ensure that when two parameters represent the same concept, the most detailed and precise value is retained. For example:
        * Use "poly(lactic-co-glycolic acid)" instead of "PLGA."
        * Prefer "100 μg/mL collagen" over "collagen."
        * Choose "wet electrospinning" over "wet electrospun" and "0.5 mL/h" over "0.5 ml/h."
5. Logical Alignment: 
    - Confirm that all entities and values logically belong to the same culture method.
    - Avoid mixing parameters from unrelated methods.
6. Text-Based Reliance:
    - Ensure every entity and its value are supported by the article text.
    - Exclude any entity or value not explicitly mentioned in the text.
    - Do not infer or assume values; include only explicitly stated information.
7. Output Revision:
    - After completing the analysis, provide the final Python JSON format, with no additional explanations:
    ```json
        <method>
    ```

Key Considerations:
    - Strict Text Reliance: Ensure all evaluations are strictly based on the article without making unsupported assumptions or inferences.
    - Entity Requirements: Adhere to the definitions and dependencies provided for each entity.
    - Logical Grouping: Confirm that all parameters belong to the same culture method and do not mix unrelated methods.

Please perform the evaluation based on the steps and key considerations outlined above.
'''
