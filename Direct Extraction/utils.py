

def get_info(method):
    if method == '3D scaffold':
        cur_schema = {
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
                }
            }
        }
        entity_requirements = {
            "Cell type": {
                "desciption": "Extract the most relevant cell types cultured using the selected method",
                "example": "primary human hepatocytes"
            },
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
            }
        }
    else:
        cur_schema = {
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
                "liver-on-a-chip": {
                    "Perfusion rate": [],
                    "Channel width": [],
                    "Channel height": []
                }
            }
        }
        entity_requirements = {
            "Cell type": {
                "desciption": "Extract the most relevant cell types cultured using the selected method",
                "example": "primary human hepatocytes"
            },
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

    return cur_schema, entity_requirements


system_prompt = '''You are an expert assistant in liver tissue engineering.'''
extract_prompt = '''
Your task is to extract specific parameters related to a cell culture method from the given article text. 
Extract only the most central parameter combination of the study, typically the one associated with key findings or optimal outcomes.
Do not extract multiple parameter groups; prioritize the one that the study emphasizes most.
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
