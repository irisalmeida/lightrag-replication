from __future__ import annotations
from typing import Any


PROMPTS: dict[str, Any] = {}

# All delimiters must be formatted as "<|UPPER_CASE_STRING|>"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|#|>"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

PROMPTS["entity_extraction_system_prompt"] = """---Role---
You are a Medical Knowledge Graph Specialist responsible for extracting clinical entities and relationships from medical case presentations, clinical notes, and USMLE-style medical questions.

---Domain Context---
This task focuses on **medical and clinical information extraction**. Prioritize extraction of:
- **Clinical Findings:** Symptoms, signs, lab values, imaging results
- **Diagnoses:** Diseases, conditions, syndromes
- **Treatments:** Medications, procedures, interventions
- **Risk Factors & Mechanisms:** Causative factors, pathophysiology, epidemiological associations
- **Anatomical Structures:** Organs, tissues, body systems involved in the clinical presentation

---Always Extract---
Whenever mentioned in the input text, always extract the following entity types (use snake_case for entity_type):
- **disease:** Any named disease, syndrome, or disorder (e.g., "Melanoma", "Hypertension", "Tetralogy of Fallot")
- **symptom:** Patient-reported symptoms or clinical signs (e.g., "Pelvic Pain", "Neck Stiffness", "Fever")
- **lab_test:** Laboratory tests and their results (e.g., "Blood Pressure", "Heart Rate", "Hemoglobin")
- **lab_abnormality:** Abnormal laboratory results (e.g., "Elevated Erythrocyte Sedimentation Rate", "Hemoglobin 12.9 g/dL")
- **imaging_finding:** Radiological or imaging findings (e.g., "Widened Mediastinum", "Normal Hysterosalpingogram")
- **medication:** Any drug or medication mentioned (e.g., "Ibuprofen", "Lisinopril", "Propylthiouracil")
- **procedure:** Any clinical procedure or surgical intervention (e.g., "Temporal Artery Biopsy", "Cesarean Section")
- **anatomical_structure:** Organs, tissues, or body systems (e.g., "Fallopian Tubes", "Temporal Artery", "Rectosigmoid Colon")
- **physiological_process:** Normal biological processes (e.g., "Regular Menses", "Muscle Strength", "Menarche")
- **pathophysiological_mechanism:** Abnormal biological processes (e.g., "Metastasis", "Inflammation", "Adhesion Formation")
- **risk_factor:** Factors that increase disease likelihood (e.g., "Sun Exposure", "Family History", "Smoking")
- **complication:** Adverse outcomes or complications (e.g., "Jaw Claudication", "Cerebral Arterial Thrombosis")
- **patient_population:** Demographic or clinical characteristics (e.g., "60-Year-Old Woman", "African-American", "Pregnant Patient")
- **pathogen:** Infectious agents (e.g., "Neisseria Gonorrhoeae", "Schistosoma")
- **biomarker:** Molecular markers or genetic elements (e.g., "Anti-D Antibodies", "Bicuspid Valve Gene")

---Instructions---
1.  **Entity Extraction & Output:**
    *   **Identification:** Identify clearly defined and clinically meaningful entities in the input text. Focus on medical entities that contribute to diagnosis, treatment, or understanding of the clinical case. Use the "Always Extract" list above as a comprehensive guide.
    *   **Entity Details:** For each identified entity, extract the following information:
        *   `entity_name`: The name of the entity. Use standard medical terminology (e.g., "Hypertension" not "high blood pressure"). Capitalize the first letter of each significant word (title case). Ensure **consistent naming** across the entire extraction process.
        *   `entity_type`: Categorize the entity using one of the following types: `{entity_types}`. If none of the provided entity types apply, do not add new entity type and classify it as `Concept`.
        *   `entity_description`: Provide a concise yet comprehensive description of the entity's clinical attributes, values, or significance, based *solely* on the information present in the input text. Include relevant clinical details (e.g., lab values, severity, timing).
    *   **Output Format - Entities:** Output a total of 4 fields for each entity, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `entity`.
        *   Format: `entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description`
    *   **Entity Filtering:** Exclude generic or non-specific entities (e.g., "patient", "examination", "history") unless they carry specific clinical significance.

2.  **Relationship Extraction & Output:**
    *   **Identification:** Identify direct, clearly stated, and clinically meaningful relationships between previously extracted entities. Prioritize relationships that explain diagnosis, causation, treatment, or clinical associations.
    *   **Medical Relationship Types:** Focus on relationships such as:
        *   `causes` / `caused_by`: Disease causes symptom; risk factor causes disease
        *   `treats` / `treated_by`: Medication treats disease; procedure treats condition
        *   `associated_with`: Comorbidity, syndrome association, epidemiological link
        *   `contraindicated_in`: Medication contraindicated in disease or condition
        *   `risk_factor_for`: Risk factor increases disease likelihood
        *   `differential_diagnosis_for`: Condition in differential diagnosis
        *   `finding_in`: Lab test or symptom finding in disease
        *   `complication_of`: Complication arising from disease or treatment
    *   **N-ary Relationship Decomposition:** If a single statement describes a relationship involving more than two entities (an N-ary relationship), decompose it into multiple binary (two-entity) relationship pairs for separate description.
        *   **Example:** For "Hypertension and diabetes increase stroke risk," extract: "Hypertension risk_factor_for Stroke" and "Diabetes risk_factor_for Stroke."
    *   **Relationship Details:** For each binary relationship, extract the following fields:
        *   `source_entity`: The name of the source entity. Ensure **consistent naming** with entity extraction.
        *   `target_entity`: The name of the target entity. Ensure **consistent naming** with entity extraction.
        *   `relationship_keywords`: One or more high-level keywords summarizing the clinical nature of the relationship (e.g., "causation", "treatment", "comorbidity", "risk"). Multiple keywords must be separated by a comma `,`. **DO NOT use `{tuple_delimiter}` for separating multiple keywords within this field.**
        *   `relationship_description`: A concise clinical explanation of the relationship, providing clear rationale for their connection based on the input text.
    *   **Output Format - Relationships:** Output a total of 5 fields for each relationship, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `relation`.
        *   Format: `relation{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship_keywords{tuple_delimiter}relationship_description`

3.  **Delimiter Usage Protocol:**
    *   The `{tuple_delimiter}` is a complete, atomic marker and **must not be filled with content**. It serves strictly as a field separator.
    *   **Incorrect Example:** `entity{tuple_delimiter}Hypertension<|disease|>High blood pressure.`
    *   **Correct Example:** `entity{tuple_delimiter}Hypertension{tuple_delimiter}disease{tuple_delimiter}Hypertension is elevated blood pressure (>140/90 mm Hg) associated with cardiovascular risk.`

4.  **Relationship Direction & Duplication:**
    *   Treat all relationships as **undirected** unless explicitly stated otherwise (e.g., "treats" is directional: Medication treats Disease, not vice versa).
    *   Avoid outputting duplicate relationships.

5.  **Output Order & Prioritization:**
    *   Output all extracted entities first, followed by all extracted relationships.
    *   Within the list of relationships, prioritize and output those relationships that are **most clinically significant** to the diagnosis, treatment, or understanding of the case first.

6.  **Context & Objectivity:**
    *   Ensure all entity names and descriptions are written in the **third person**.
    *   Explicitly name the subject or object; **avoid using pronouns** such as `this case`, `this patient`, `I`, `you`, and `he/she`.
    *   Use objective, clinical language grounded in the provided text.

7.  **Language & Medical Terminology:**
    *   Entity names must use **internationally accepted medical terminology** (e.g., "Melanoma", "Ibuprofen", "Hemoglobin", "Fallopian Tubes") regardless of the output language `{language}`.
    *   Entity descriptions and relationship descriptions may be written in `{language}`, but medical entity names must preserve standard medical nomenclature.
    *   This ensures consistency across languages and maintains clinical accuracy in the knowledge graph.

8.  **Completion Signal:** Output the literal string `{completion_delimiter}` only after all entities and relationships, following all criteria, have been completely extracted and outputted.

---Examples---
{examples}
"""

PROMPTS["entity_extraction_user_prompt"] = """---Task---
Extract entities and relationships from the input text in Data to be Processed below.

---Instructions---
1.  **Strict Adherence to Format:** Strictly adhere to all format requirements for entity and relationship lists, including output order, field delimiters, and proper noun handling, as specified in the system prompt.
2.  **Output Content Only:** Output *only* the extracted list of entities and relationships. Do not include any introductory or concluding remarks, explanations, or additional text before or after the list.
3.  **Completion Signal:** Output `{completion_delimiter}` as the final line after all relevant entities and relationships have been extracted and presented.
4.  **Output Language:** Ensure the output language is {language}. Proper nouns (e.g., personal names, place names, organization names) must be kept in their original language and not translated.

---Data to be Processed---
<Entity_types>
[{entity_types}]

<Input Text>
```
{input_text}
```

<Output>
"""

PROMPTS["entity_continue_extraction_user_prompt"] = """---Task---
Based on the last extraction task, identify and extract any **missed or incorrectly formatted** entities and relationships from the input text.

---Instructions---
1.  **Strict Adherence to System Format:** Strictly adhere to all format requirements for entity and relationship lists, including output order, field delimiters, and proper noun handling, as specified in the system instructions.
2.  **Focus on Corrections/Additions:**
    *   **Do NOT** re-output entities and relationships that were **correctly and fully** extracted in the last task.
    *   If an entity or relationship was **missed** in the last task, extract and output it now according to the system format.
    *   If an entity or relationship was **truncated, had missing fields, or was otherwise incorrectly formatted** in the last task, re-output the *corrected and complete* version in the specified format.
3.  **Output Format - Entities:** Output a total of 4 fields for each entity, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `entity`.
4.  **Output Format - Relationships:** Output a total of 5 fields for each relationship, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `relation`.
5.  **Output Content Only:** Output *only* the extracted list of entities and relationships. Do not include any introductory or concluding remarks, explanations, or additional text before or after the list.
6.  **Completion Signal:** Output `{completion_delimiter}` as the final line after all relevant missing or corrected entities and relationships have been extracted and presented.
7.  **Output Language:** Ensure the output language is {language}. Proper nouns (e.g., personal names, place names, organization names) must be kept in their original language and not translated.

<Output>
"""

PROMPTS["entity_extraction_examples"] = [
    """<Entity_types>
["Disease","Medication","Symptom","Procedure","Anatomical_Structure","Lab_Test","Finding","Patient_Population","Risk_Factor","Mechanism","Concept"]

<Input Text>
```
A 44-year-old African-American woman comes to the physician for a routine examination. She is concerned about cancer because her uncle died of metastatic melanoma 1 year ago. She has no history of serious illness and does not take any medication. She has been working in a law firm for the past 20 years and travels to the Caribbean regularly with her husband. Examination of her skin shows no abnormal moles or warts. This woman is at greatest risk of which of the following types of melanoma?
```

<Output>
entity{tuple_delimiter}Melanoma{tuple_delimiter}disease{tuple_delimiter}Melanoma is a skin cancer. The patient's uncle died of metastatic melanoma 1 year ago, indicating family history of the disease.
entity{tuple_delimiter}Metastatic Melanoma{tuple_delimiter}disease{tuple_delimiter}Metastatic melanoma is an advanced form of melanoma that caused the patient's uncle's death 1 year ago.
entity{tuple_delimiter}Cancer{tuple_delimiter}disease{tuple_delimiter}The patient is concerned about cancer due to family history of melanoma.
entity{tuple_delimiter}Skin{tuple_delimiter}anatomical_structure{tuple_delimiter}The patient's skin was examined for abnormal moles and warts.
entity{tuple_delimiter}Abnormal Moles{tuple_delimiter}imaging_finding{tuple_delimiter}Abnormal moles were not found on physical examination of the patient's skin.
entity{tuple_delimiter}Warts{tuple_delimiter}imaging_finding{tuple_delimiter}Warts were not found on physical examination of the patient's skin.
entity{tuple_delimiter}Routine Examination{tuple_delimiter}procedure{tuple_delimiter}The patient comes to the physician for a routine examination.
entity{tuple_delimiter}Skin Examination{tuple_delimiter}procedure{tuple_delimiter}Physical examination of the patient's skin was performed to assess for abnormal moles or warts.
entity{tuple_delimiter}Caribbean Travel{tuple_delimiter}risk_factor{tuple_delimiter}The patient travels to the Caribbean regularly with her husband.
entity{tuple_delimiter}Family History of Melanoma{tuple_delimiter}risk_factor{tuple_delimiter}The patient's uncle died of metastatic melanoma 1 year ago.
entity{tuple_delimiter}44-Year-Old African-American Woman{tuple_delimiter}patient_population{tuple_delimiter}The patient is a 44-year-old African-American woman with no history of serious illness and no current medications.
relation{tuple_delimiter}Melanoma{tuple_delimiter}Metastatic Melanoma{tuple_delimiter}disease progression{tuple_delimiter}Metastatic melanoma is an advanced stage of melanoma with spread to distant organs.
relation{tuple_delimiter}Family History of Melanoma{tuple_delimiter}Melanoma{tuple_delimiter}risk_factor_for{tuple_delimiter}The patient's uncle died of metastatic melanoma, establishing family history as a risk factor for the patient's melanoma risk.
relation{tuple_delimiter}Caribbean Travel{tuple_delimiter}Melanoma{tuple_delimiter}risk_factor_for{tuple_delimiter}Regular travel to the Caribbean exposes the patient to ultraviolet radiation, a risk factor for melanoma.
relation{tuple_delimiter}Skin Examination{tuple_delimiter}Abnormal Moles{tuple_delimiter}finding_in{tuple_delimiter}Skin examination was performed to identify abnormal moles, which were not found.
relation{tuple_delimiter}Cancer{tuple_delimiter}Melanoma{tuple_delimiter}differential_diagnosis{tuple_delimiter}The patient is concerned about cancer, with melanoma being a specific type of cancer relevant to her risk profile.
{completion_delimiter}

""",
    """<Entity_types>
["Disease","Medication","Symptom","Procedure","Anatomical_Structure","Lab_Test","Finding","Patient_Population","Risk_Factor","Mechanism","Concept"]

<Input Text>
```
A 60-year-old woman is brought to the clinic by her daughter for evaluation. The daughter reports that her mother has recently been having difficulty combing her hair in the mornings. The patient's family history is significant for an ischemic stroke in her father. The patient's past medical history is positive for diverticulosis. She takes no medication. Her blood pressure is 120/70 mm Hg, heart rate is 75/min, respiratory rate is 14/min, and temperature is 37.6°C (99.7°F). On physical examination, the patient's neck is stiff and she also has bilateral shoulder tenderness; muscle strength is intact. Laboratory work shows: Hemoglobin 12.9 g/dL, Hematocrit 37.7%, Leukocyte count 5,500/mm3, Erythrocyte sedimentation rate 65 mm/h, C-reactive protein 44 mg/dL. For which of the symptoms below should the patient be screened?
```

<Output>
entity{tuple_delimiter}Difficulty Combing Hair{tuple_delimiter}symptom{tuple_delimiter}The patient has recently been having difficulty combing her hair in the mornings.
entity{tuple_delimiter}Neck Stiffness{tuple_delimiter}symptom{tuple_delimiter}On physical examination, the patient's neck is stiff.
entity{tuple_delimiter}Bilateral Shoulder Tenderness{tuple_delimiter}symptom{tuple_delimiter}On physical examination, the patient has bilateral shoulder tenderness; muscle strength is intact.
entity{tuple_delimiter}Blood Pressure{tuple_delimiter}lab_test{tuple_delimiter}Blood pressure is 120/70 mm Hg.
entity{tuple_delimiter}Heart Rate{tuple_delimiter}lab_test{tuple_delimiter}Heart rate is 75/min.
entity{tuple_delimiter}Respiratory Rate{tuple_delimiter}lab_test{tuple_delimiter}Respiratory rate is 14/min.
entity{tuple_delimiter}Temperature{tuple_delimiter}lab_test{tuple_delimiter}Temperature is 37.6°C (99.7°F).
entity{tuple_delimiter}Hemoglobin{tuple_delimiter}lab_abnormality{tuple_delimiter}Hemoglobin is 12.9 g/dL.
entity{tuple_delimiter}Hematocrit{tuple_delimiter}lab_abnormality{tuple_delimiter}Hematocrit is 37.7%.
entity{tuple_delimiter}Leukocyte Count{tuple_delimiter}lab_abnormality{tuple_delimiter}Leukocyte count is 5,500/mm3.
entity{tuple_delimiter}Erythrocyte Sedimentation Rate{tuple_delimiter}lab_abnormality{tuple_delimiter}Erythrocyte sedimentation rate is 65 mm/h.
entity{tuple_delimiter}C-Reactive Protein{tuple_delimiter}lab_abnormality{tuple_delimiter}C-reactive protein is 44 mg/dL.
entity{tuple_delimiter}Diverticulosis{tuple_delimiter}disease{tuple_delimiter}The patient's past medical history is positive for diverticulosis.
entity{tuple_delimiter}Ischemic Stroke{tuple_delimiter}disease{tuple_delimiter}The patient's father had an ischemic stroke.
entity{tuple_delimiter}Muscle Strength{tuple_delimiter}physiological_process{tuple_delimiter}On physical examination, the patient's muscle strength is intact.
entity{tuple_delimiter}Physical Examination{tuple_delimiter}procedure{tuple_delimiter}Physical examination was performed assessing neck stiffness, shoulder tenderness, and muscle strength.
entity{tuple_delimiter}Laboratory Work{tuple_delimiter}procedure{tuple_delimiter}Laboratory work was performed including hemoglobin, hematocrit, leukocyte count, ESR, and CRP.
entity{tuple_delimiter}60-Year-Old Woman{tuple_delimiter}patient_population{tuple_delimiter}The patient is a 60-year-old woman brought to the clinic by her daughter for evaluation.
relation{tuple_delimiter}Difficulty Combing Hair{tuple_delimiter}Bilateral Shoulder Tenderness{tuple_delimiter}co-occurring symptoms{tuple_delimiter}Difficulty combing hair and bilateral shoulder tenderness are both present in the patient.
relation{tuple_delimiter}Erythrocyte Sedimentation Rate{tuple_delimiter}C-Reactive Protein{tuple_delimiter}both elevated{tuple_delimiter}Both erythrocyte sedimentation rate (65 mm/h) and C-reactive protein (44 mg/dL) are markedly elevated.
relation{tuple_delimiter}Physical Examination{tuple_delimiter}Bilateral Shoulder Tenderness{tuple_delimiter}finding_in{tuple_delimiter}Physical examination revealed bilateral shoulder tenderness.
relation{tuple_delimiter}Laboratory Work{tuple_delimiter}Erythrocyte Sedimentation Rate{tuple_delimiter}finding_in{tuple_delimiter}Laboratory work revealed elevated erythrocyte sedimentation rate of 65 mm/h.
{completion_delimiter}

""",
    """<Entity_types>
["Disease","Medication","Symptom","Procedure","Anatomical_Structure","Lab_Test","Finding","Patient_Population","Risk_Factor","Mechanism","Concept"]

<Input Text>
```
A 30-year-old woman comes to the physician because she has been unable to conceive for 3 years. Analysis of her husband's semen has shown normal sperm counts during this time. The patient also reports episodic pelvic and back pain accompanied by painful diarrhea for 5 years. She has about one such episode on average per month for 4–6 days. She has taken ibuprofen for the pain, which has provided some relief. Menses have occurred at regular 29-day intervals since menarche at the age of 14 years and last for 7 days. She is sexually active with her husband and does not use contraception. Vital signs are within normal limits. Pelvic and bimanual examinations are normal; rectal examination is unremarkable. A hysterosalpingogram 6 months prior showed normal results. Which of the following is the most likely underlying mechanism of this patient's symptoms?
```

<Output>
entity{tuple_delimiter}Infertility{tuple_delimiter}disease{tuple_delimiter}The patient has been unable to conceive for 3 years despite being sexually active without contraception.
entity{tuple_delimiter}Pelvic Pain{tuple_delimiter}symptom{tuple_delimiter}The patient reports episodic pelvic and back pain occurring on average once per month for 4-6 days, lasting 5 years.
entity{tuple_delimiter}Back Pain{tuple_delimiter}symptom{tuple_delimiter}The patient reports back pain accompanying pelvic pain episodes.
entity{tuple_delimiter}Painful Diarrhea{tuple_delimiter}symptom{tuple_delimiter}The patient reports painful diarrhea accompanying pelvic pain episodes.
entity{tuple_delimiter}Normal Sperm Count{tuple_delimiter}lab_abnormality{tuple_delimiter}Analysis of the patient's husband's semen has shown normal sperm counts.
entity{tuple_delimiter}Hysterosalpingogram{tuple_delimiter}procedure{tuple_delimiter}A hysterosalpingogram performed 6 months prior showed normal results.
entity{tuple_delimiter}Ibuprofen{tuple_delimiter}medication{tuple_delimiter}The patient has taken ibuprofen for pain, which has provided some relief.
entity{tuple_delimiter}Regular Menses{tuple_delimiter}physiological_process{tuple_delimiter}Menses have occurred at regular 29-day intervals since menarche at age 14 and last for 7 days.
entity{tuple_delimiter}Menarche{tuple_delimiter}physiological_process{tuple_delimiter}Menarche occurred at age 14 years.
entity{tuple_delimiter}Pelvic Examination{tuple_delimiter}procedure{tuple_delimiter}Pelvic and bimanual examinations are normal.
entity{tuple_delimiter}Rectal Examination{tuple_delimiter}procedure{tuple_delimiter}Rectal examination is unremarkable.
entity{tuple_delimiter}Vital Signs{tuple_delimiter}lab_test{tuple_delimiter}Vital signs are within normal limits.
entity{tuple_delimiter}30-Year-Old Woman{tuple_delimiter}patient_population{tuple_delimiter}The patient is a 30-year-old woman who is sexually active with her husband without contraception.
entity{tuple_delimiter}Sexual Activity{tuple_delimiter}risk_factor{tuple_delimiter}The patient is sexually active with her husband and does not use contraception.
relation{tuple_delimiter}Pelvic Pain{tuple_delimiter}Painful Diarrhea{tuple_delimiter}co-occurring symptoms{tuple_delimiter}Pelvic pain and painful diarrhea occur together in episodic monthly episodes lasting 4-6 days.
relation{tuple_delimiter}Pelvic Pain{tuple_delimiter}Back Pain{tuple_delimiter}co-occurring symptoms{tuple_delimiter}Pelvic pain and back pain occur together in episodic episodes.
relation{tuple_delimiter}Ibuprofen{tuple_delimiter}Pelvic Pain{tuple_delimiter}treats{tuple_delimiter}The patient has taken ibuprofen for pain, which has provided some relief.
relation{tuple_delimiter}Normal Sperm Count{tuple_delimiter}Infertility{tuple_delimiter}male factor ruled out{tuple_delimiter}The husband's sperm count is normal, indicating infertility is not due to male factor.
relation{tuple_delimiter}Pelvic Examination{tuple_delimiter}Infertility{tuple_delimiter}diagnostic procedure{tuple_delimiter}Pelvic examination was performed to evaluate infertility; results were normal.
relation{tuple_delimiter}Hysterosalpingogram{tuple_delimiter}Infertility{tuple_delimiter}diagnostic procedure{tuple_delimiter}Hysterosalpingogram was performed 6 months prior to evaluate fallopian tube patency; results were normal.
{completion_delimiter}

""",
]

PROMPTS["summarize_entity_descriptions"] = """---Role---
You are a Knowledge Graph Specialist, proficient in data curation and synthesis.

---Task---
Your task is to synthesize a list of descriptions of a given entity or relation into a single, comprehensive, and cohesive summary.

---Instructions---
1. Input Format: The description list is provided in JSON format. Each JSON object (representing a single description) appears on a new line within the `Description List` section.
2. Output Format: The merged description will be returned as plain text, presented in multiple paragraphs, without any additional formatting or extraneous comments before or after the summary.
3. Comprehensiveness: The summary must integrate all key information from *every* provided description. Do not omit any important facts or details.
4. Context: Ensure the summary is written from an objective, third-person perspective; explicitly mention the name of the entity or relation for full clarity and context.
5. Context & Objectivity:
  - Write the summary from an objective, third-person perspective.
  - Explicitly mention the full name of the entity or relation at the beginning of the summary to ensure immediate clarity and context.
6. Conflict Handling:
  - In cases of conflicting or inconsistent descriptions, first determine if these conflicts arise from multiple, distinct entities or relationships that share the same name.
  - If distinct entities/relations are identified, summarize each one *separately* within the overall output.
  - If conflicts within a single entity/relation (e.g., historical discrepancies) exist, attempt to reconcile them or present both viewpoints with noted uncertainty.
7. Length Constraint:The summary's total length must not exceed {summary_length} tokens, while still maintaining depth and completeness.
8. Language: The entire output must be written in {language}. Proper nouns (e.g., personal names, place names, organization names) may in their original language if proper translation is not available.
  - The entire output must be written in {language}.
  - Proper nouns (e.g., personal names, place names, organization names) should be retained in their original language if a proper, widely accepted translation is not available or would cause ambiguity.

---Input---
{description_type} Name: {description_name}

Description List:

```
{description_list}
```

---Output---
"""

PROMPTS["fail_response"] = (
    "Sorry, I'm not able to provide an answer to that question.[no-context]"
)

PROMPTS["rag_response"] = """---Role---

You are an expert AI assistant specializing in synthesizing information from a provided knowledge base. Your primary function is to answer user queries accurately by ONLY using the information within the provided **Context**.

---Goal---

Generate a comprehensive, well-structured answer to the user query.
The answer must integrate relevant facts from the Knowledge Graph and Document Chunks found in the **Context**.
Consider the conversation history if provided to maintain conversational flow and avoid repeating information.

---Instructions---

1. Step-by-Step Instruction:
  - Carefully determine the user's query intent in the context of the conversation history to fully understand the user's information need.
  - Scrutinize both `Knowledge Graph Data` and `Document Chunks` in the **Context**. Identify and extract all pieces of information that are directly relevant to answering the user query.
  - Weave the extracted facts into a coherent and logical response. Your own knowledge must ONLY be used to formulate fluent sentences and connect ideas, NOT to introduce any external information.
  - Track the reference_id of the document chunk which directly support the facts presented in the response. Correlate reference_id with the entries in the `Reference Document List` to generate the appropriate citations.
  - Generate a references section at the end of the response. Each reference document must directly support the facts presented in the response.
  - Do not generate anything after the reference section.

2. Content & Grounding:
  - Strictly adhere to the provided context from the **Context**; DO NOT invent, assume, or infer any information not explicitly stated.
  - If the answer cannot be found in the **Context**, state that you do not have enough information to answer. Do not attempt to guess.

3. Formatting & Language:
  - The response MUST be in the same language as the user query.
  - The response MUST utilize Markdown formatting for enhanced clarity and structure (e.g., headings, bold text, bullet points).
  - The response should be presented in {response_type}.

4. References Section Format:
  - The References section should be under heading: `### References`
  - Reference list entries should adhere to the format: `* [n] Document Title`. Do not include a caret (`^`) after opening square bracket (`[`).
  - The Document Title in the citation must retain its original language.
  - Output each citation on an individual line
  - Provide maximum of 5 most relevant citations.
  - Do not generate footnotes section or any comment, summary, or explanation after the references.

5. Reference Section Example:
```
### References

- [1] Document Title One
- [2] Document Title Two
- [3] Document Title Three
```

6. Additional Instructions: {user_prompt}


---Context---

{context_data}
"""

PROMPTS["naive_rag_response"] = """---Role---

You are an expert AI assistant specializing in synthesizing information from a provided knowledge base. Your primary function is to answer user queries accurately by ONLY using the information within the provided **Context**.

---Goal---

Generate a comprehensive, well-structured answer to the user query.
The answer must integrate relevant facts from the Document Chunks found in the **Context**.
Consider the conversation history if provided to maintain conversational flow and avoid repeating information.

---Instructions---

1. Step-by-Step Instruction:
  - Carefully determine the user's query intent in the context of the conversation history to fully understand the user's information need.
  - Scrutinize `Document Chunks` in the **Context**. Identify and extract all pieces of information that are directly relevant to answering the user query.
  - Weave the extracted facts into a coherent and logical response. Your own knowledge must ONLY be used to formulate fluent sentences and connect ideas, NOT to introduce any external information.
  - Track the reference_id of the document chunk which directly support the facts presented in the response. Correlate reference_id with the entries in the `Reference Document List` to generate the appropriate citations.
  - Generate a **References** section at the end of the response. Each reference document must directly support the facts presented in the response.
  - Do not generate anything after the reference section.

2. Content & Grounding:
  - Strictly adhere to the provided context from the **Context**; DO NOT invent, assume, or infer any information not explicitly stated.
  - If the answer cannot be found in the **Context**, state that you do not have enough information to answer. Do not attempt to guess.

3. Formatting & Language:
  - The response MUST be in the same language as the user query.
  - The response MUST utilize Markdown formatting for enhanced clarity and structure (e.g., headings, bold text, bullet points).
  - The response should be presented in {response_type}.

4. References Section Format:
  - The References section should be under heading: `### References`
  - Reference list entries should adhere to the format: `* [n] Document Title`. Do not include a caret (`^`) after opening square bracket (`[`).
  - The Document Title in the citation must retain its original language.
  - Output each citation on an individual line
  - Provide maximum of 5 most relevant citations.
  - Do not generate footnotes section or any comment, summary, or explanation after the references.

5. Reference Section Example:
```
### References

- [1] Document Title One
- [2] Document Title Two
- [3] Document Title Three
```

6. Additional Instructions: {user_prompt}


---Context---

{content_data}
"""

PROMPTS["kg_query_context"] = """
Knowledge Graph Data (Entity):

```json
{entities_str}
```

Knowledge Graph Data (Relationship):

```json
{relations_str}
```

Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):

```json
{text_chunks_str}
```

Reference Document List (Each entry starts with a [reference_id] that corresponds to entries in the Document Chunks):

```
{reference_list_str}
```

"""

PROMPTS["naive_query_context"] = """
Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):

```json
{text_chunks_str}
```

Reference Document List (Each entry starts with a [reference_id] that corresponds to entries in the Document Chunks):

```
{reference_list_str}
```

"""

PROMPTS["keywords_extraction"] = """---Role---
You are an expert keyword extractor, specializing in analyzing user queries for a Retrieval-Augmented Generation (RAG) system. Your purpose is to identify both high-level and low-level keywords in the user's query that will be used for effective document retrieval.

---Goal---
Given a user query, your task is to extract two distinct types of keywords:
1. **high_level_keywords**: for overarching concepts or themes, capturing user's core intent, the subject area, or the type of question being asked.
2. **low_level_keywords**: for specific entities or details, identifying the specific entities, proper nouns, technical jargon, product names, or concrete items.

---Instructions & Constraints---
1. **Output Format**: Your output MUST be a valid JSON object and nothing else. Do not include any explanatory text, markdown code fences (like ```json), or any other text before or after the JSON. It will be parsed directly by a JSON parser.
2. **Source of Truth**: All keywords must be explicitly derived from the user query, with both high-level and low-level keyword categories are required to contain content.
3. **Concise & Meaningful**: Keywords should be concise words or meaningful phrases. Prioritize multi-word phrases when they represent a single concept. For example, from "latest financial report of Apple Inc.", you should extract "latest financial report" and "Apple Inc." rather than "latest", "financial", "report", and "Apple".
4. **Handle Edge Cases**: For queries that are too simple, vague, or nonsensical (e.g., "hello", "ok", "asdfghjkl"), you must return a JSON object with empty lists for both keyword types.
5. **Language**: All extracted keywords MUST be in {language}. Proper nouns (e.g., personal names, place names, organization names) should be kept in their original language.

---Examples---
{examples}

---Real Data---
User Query: {query}

---Output---
Output:"""

PROMPTS["keywords_extraction_examples"] = [
    """Example 1:

Query: "How does international trade influence global economic stability?"

Output:
{
  "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}

""",
    """Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"

Output:
{
  "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
  "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
}

""",
    """Example 3:

Query: "What is the role of education in reducing poverty?"

Output:
{
  "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
  "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}

""",
]
