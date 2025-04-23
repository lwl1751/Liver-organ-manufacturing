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

sft_prompt = f'''Your task is to identify and extract the relevant entities and information from a research paper in the field of liver tissue engineering.
The extracted information should follow the provided schema in a Python dictionary format, where keys represent the entities of interest and values describe the required information and format.

The schema is as follows:
{schema}

Please follow these guidelines:
1. Entity-Specific Rules:
    - Seeded Cell Density and Number of Cells: Do not infer one from the other. If only one is provided in the text, include it as-is.
    - Numeric Entities (e.g., Perfusion Rate, Pore Size, Porosity): 
        - Select the value based on this priority:
            * Mean value (if explicitly provided).
            * Maximum value.
            * Range value (if explicitly mentioned).
        - Avoid calculations or assumptions; only include values directly provided in the text.
    - Material in Cell Culture and Concentration:
        * If a concentration is specified, ensure the corresponding material is also included.
        * Material in cell culture can exist without concentration of material.
2. Do not fill in any information that is not explicitly in the text fragment. 
3. If you don’t know something from the context, just leave that spot blank (don’t guess!).
4. Please provide the JSON output in the specified format without additional explanation.

Here is the research text: 
'''

inference_prompt = f'''Your task is to identify and extract the relevant entities and information from a research paper in the field of liver tissue engineering.
The extracted information should follow the provided schema in a Python dictionary format, where keys represent the entities of interest and values describe the required information and format.

The schema is as follows:
{schema}

Please follow these guidelines:
1. Entity-Specific Rules:
    - Seeded Cell Density and Number of Cells: Do not infer one from the other. If only one is provided in the text, include it as-is.
    - Numeric Entities (e.g., Perfusion Rate, Pore Size, Porosity): 
        - Select the value based on this priority:
            * Mean value (if explicitly provided).
            * Maximum value.
            * Range value (if explicitly mentioned).
        - Avoid calculations or assumptions; only include values directly provided in the text.
    - Material in Cell Culture and Concentration:
        * If a concentration is specified, ensure the corresponding material is also included.
        * Material in cell culture can exist without concentration of material.
2. Do not fill in any information that is not explicitly in the text fragment. 
3. If you don’t know something from the context, just leave that spot blank (don’t guess!).
4. Please provide the JSON schema output in the specified format without additional explanation. 

Here are several examples:
<Example>

Here is the research text: 
<text>

Output:
'''

