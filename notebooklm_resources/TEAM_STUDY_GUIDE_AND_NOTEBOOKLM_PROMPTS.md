# OCEAN-SHIELD: Team Study Guide & NotebookLM Prompts

## 1. Team Role Breakdown & Presentation Split

For a winning hackathon or client defense, split presentation duties clearly across team members based on technical specialization:

| Team Role | Member Responsibility | Core Talking Points to Master | Key Reference Doc |
| :--- | :--- | :--- | :--- |
| **Speaker 1: Team Lead / Operational Vision** | Hook, Problem Statement, NTRO Context, System Architecture | 80% crude transit through Indian Ocean, 2.37M km² EEZ, SIH26143 requirements (a, b, c), end-to-end architecture, Coast Guard interdiction impact. | `01_PROJECT_OVERVIEW` & `06_PITCH_SCRIPT` |
| **Speaker 2: AI & Computer Vision Lead** | Satellite Ingestion, Super-Resolution, PyTorch U-Net Segmentation | Sentinel-1 SAR C-band all-weather capability, Sentinel-2 Optical NDOI index, ESPCN 2X sub-pixel convolution, Dice + BCE loss formula, look-alike rejection. | `02_DATASETS` & `05_MATHEMATICAL_FOUNDATIONS` |
| **Speaker 3: Physics & Ocean Modeling Lead** | Hydrodynamic Drift, Environmental Data, ADIOS Weathering | HYCOM 4D current vectors, NOAA GFS wind grids, 4th-Order Runge-Kutta (RK4) integration vs Euler, Mackay ADIOS kinetics (evaporation, emulsification, Mooney viscosity). | `03_TECH_STACK` & `05_MATHEMATICAL_FOUNDATIONS` |
| **Speaker 4: Maritime Intelligence & Full-Stack Lead** | AIS Dark Ship Forensics, C2 Dashboard, Automated PDF Reporting | MarineCadastre AIS schema, Closest Point of Approach (CPA), 4-factor composite risk scoring formula, vanilla JS 60 FPS GIS cockpit, ReportLab PDF generation. | `04_UI_ELEMENTS` & `06_PITCH_SCRIPT` |

---

## 2. How to Use Google NotebookLM with This Bundle

Google NotebookLM is an AI research assistant powered by Google Gemini that turns your project documentation into an interactive knowledge base, flashcards, study guides, and an **Audio Overview (AI Podcast)**.

### Step-by-Step Setup:
1. **Unzip the Bundle**:
   - Extract `OCEAN_SHIELD_NOTEBOOKLM_BUNDLE.zip` on your computer.
2. **Open NotebookLM**:
   - Navigate to [https://notebooklm.google.com/](https://notebooklm.google.com/) and sign in with your Google account.
3. **Create a New Notebook**:
   - Click **"New Notebook"** and name it `OCEAN-SHIELD SIH26143 Master Defense`.
4. **Upload Sources**:
   - In the "Add Sources" modal, select **Upload Files** or drag and drop all `.md` files from the unzipped folder:
     - `01_PROJECT_OVERVIEW_AND_PROBLEM_STATEMENT.md`
     - `02_DATASETS_AND_PROVENANCE.md`
     - `03_TECH_STACK_AND_ARCHITECTURE.md`
     - `04_UI_ELEMENTS_AND_OPERATIONAL_WORKFLOW.md`
     - `05_MATHEMATICAL_AND_ALGORITHMIC_FOUNDATIONS.md`
     - `06_PITCH_SCRIPT_DEMO_AND_JUDGE_QA.md`
     - `TEAM_STUDY_GUIDE_AND_NOTEBOOKLM_PROMPTS.md`
     - `PITCH_AND_JUDGE_GUIDE.md`
5. **Generate the "Deep Dive" Audio Overview (AI Podcast)**:
   - In the top right of the notebook, locate the **Audio Overview** tile.
   - Click **"Generate"**.
   - NotebookLM will automatically create a 10-15 minute conversational podcast where two AI hosts discuss your project, explain the math, review the architecture, and hype the real-world impact.
   - **Share this audio file with all your teammates** so they can listen and absorb the pitch on the go!

---

## 3. Top 20 Prompts to Run in NotebookLM for Team Mastery

Copy and paste these exact prompts into NotebookLM's chat box to prepare for judge scrutiny:

### A. General Understanding & System Architecture
1. *"Give me a comprehensive executive summary of project OCEAN-SHIELD, explaining why it was built for NTRO and how it solves problem statement SIH26143."*
2. *"List all technologies used in the frontend, backend, AI models, physics simulation, and reporting engines, along with the specific justification for choosing each one."*
3. *"Explain the end-to-end data flow from the moment an ESA Sentinel satellite captures a radar pass over the Indian Ocean to the moment an ICG Dornier aircraft is dispatched."*
4. *"Create a high-yield study sheet comparing Sentinel-1 SAR imagery and Sentinel-2 Optical imagery in the context of oil spill detection."*

### B. Mathematical & Algorithmic Drills
5. *"Explain the hybrid Soft Dice + BCE loss function to me as if I were a junior machine learning engineer. Why can't we just use standard Binary Cross-Entropy?"*
6. *"Break down the 4th-Order Runge-Kutta (RK4) advection differential equations step-by-step. Show how $k_1, k_2, k_3,$ and $k_4$ are computed from ocean current and wind vectors."*
7. *"Explain the Mackay ADIOS chemical weathering model. What is the Mooney equation and why does increasing water emulsion make oil impossible to pump?"*
8. *"Explain how the 4-factor composite culprit risk score is calculated for suspect vessels, including the exact formulas for $S_{\text{spatial}}$, $S_{\text{temporal}}$, and $S_{\text{anomaly}}$."*
9. *"What is ESPCN sub-pixel convolution and how does it achieve 2X super-resolution without the latency of deconvolutional layers?"*

### C. UI & Operational Workflow Practice
10. *"Describe every major visual component of the OCEAN-SHIELD Command and Control cockpit, explaining what each gauge, map layer, and panel indicates to a naval watchstander."*
11. *"Walk me through the exact 4-step guided button workflow in the left control panel. What happens under the hood when each button is clicked?"*
12. *"What happens if a user clicks 'Step 3: Intercept' before clicking Step 1 or Step 2? How does the auto-cascade feature prevent system crashes?"*

### D. Mock Judge Grilling & Role-Playing
13. *"Act as a tough, skeptical SIH hackathon judge with expertise in Synthetic Aperture Radar and oceanography. Ask me 5 brutal questions to test if our team actually built this."*
14. *"How does OCEAN-SHIELD catch a 'dark ship' that turned off its AIS transponder before dumping oil? Give me the exact mathematical and kinematic defense."*
15. *"A judge asks: 'Why didn't you just use Euler's method for drift forecasting instead of overcomplicating it with Runge-Kutta 4th-order?' What is my bulletproof 30-second rebuttal?"*
16. *"A judge claims: 'Algal blooms look identical to oil spills in radar imagery.' How do I prove them wrong using physics and multi-spectral infrared bands?"*
17. *"What makes our generated PDF report legally admissible in international maritime courts under MARPOL Annex I?"*

### E. Team Rehearsal & Flashcards
18. *"Generate a 15-question multiple choice quiz with answers and explanations to test my team's knowledge of the OCEAN-SHIELD codebase."*
19. *"Condense the entire 2-minute pitch script into 5 concise bullet points that our lead presenter can memorize for the opening statement."*
20. *"Create a table of the top 10 numerical metrics of our project (e.g., latency, accuracy, resolution, EEZ size, loss weights) that every team member must have memorized."*
